import math
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from scipy.spatial.distance import cdist


class Utilities:
    def __init__(self):
        pass

        
    def find_image_center(self, img):
        """Returns the center coordinates (x, y) of an image."""
        return img.shape[1] // 2, img.shape[0] // 2   


    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def determine_quadrant(self, center_x, center_y, qr_x, qr_y):
        if qr_x >= center_x and qr_y < center_y:
            return 'Q1'

        elif qr_x < center_x and qr_y < center_y:
            return 'Q2'

        elif qr_x < center_x and qr_y >= center_y:
            return 'Q3'

        elif qr_x >= center_x and qr_y >= center_y:
            return 'Q4'


    def calculate_center(self, coordinates):
        x_coords = [pt[0] for pt in coordinates]
        y_coords = [pt[1] for pt in coordinates]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return center_x, center_y


    def find_closest_match_sequence(self, part_number, candidates):
        """Find the closest matching part number from a list of candidates."""
        if not candidates:
            return None  # Handle empty candidate list gracefully

        def similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()

        # Convert candidates to strings and compute similarities
        closest_match = max(map(str, candidates), key=lambda candidate: similarity(part_number, candidate), default=None)

        return closest_match


    def update_part_no_with_closest_match(self, csv_path, pairs, image_center_x):
        """
        Match part number and box number with CSV data, handling two cases:
        1. When invoice number is present: match all three fields.
        2. When invoice is "missing invoice": match only part number and box number.
        """
        # Load and clean CSV data
        df = pd.read_csv(csv_path, keep_default_na=False)
        
        # Ensure data types are consistent
        df[['part_number', 'box_number', 'invoice_number', 'box_quantity']] = df[
            ['part_number', 'box_number', 'invoice_number', 'box_quantity']
        ].astype(str)

        pair_to_closest_part = {}
        ocr_results, ocr_boxes = [], []
        
        processed_pairs, processed_results = set(), set()

        # Step 1: Precompute closest part numbers
        for number_number, closest_text, invoice_number, box_quantity, _ in pairs:
            pair_key = (number_number, closest_text, invoice_number, box_quantity)
            if pair_key in processed_pairs:
                continue

            if closest_text != "No match found":
                closest_part = self.find_closest_match_sequence(str(closest_text), df['part_number'].tolist())
                if closest_part:
                    pair_to_closest_part[pair_key] = closest_part

            processed_pairs.add(pair_key)

        # Step 2: Process each unique pair and find matches in CSV
        processed_pairs.clear()  

        for number_number, closest_text, invoice_number, box_quantity, bbox in pairs:
            pair_key = (number_number, closest_text, invoice_number, box_quantity)
            if pair_key in processed_pairs:
                continue

            closest_part = pair_to_closest_part.get(pair_key, "No match found")
            result_key = (number_number, closest_part, invoice_number, box_quantity)

            if result_key in processed_results:
                continue

            # Matching logic based on invoice_number presence
            match_conditions = (
                (df['box_number'] == number_number) &
                (df['part_number'] == closest_part) &
                (df['box_quantity'] == box_quantity)
            )
            if invoice_number != "missing invoice":
                match_conditions &= (df['invoice_number'] == invoice_number)

            matched_rows = df[match_conditions]

            if not matched_rows.empty:
                row = matched_rows.iloc[-1]  # Take the last matched row
                ocr_results.append({
                    "box_number": row['box_number'],
                    "invoice_number": row['invoice_number'],
                    "box_quantity": row['box_quantity'],
                    "part_number": row['part_number']
                })

                # Compute bounding box center
                bbox_center_x = (bbox[0][0] + bbox[2][0]) // 2
                bbox_center_y = (bbox[0][1] + bbox[2][1]) // 2
                ocr_boxes.append((bbox_center_x, bbox_center_y))

                processed_results.add(result_key)

            processed_pairs.add(pair_key)

        return ocr_results, ocr_boxes


    def map_ocr_to_boxes(self, box_centers, ocr_centers, ocr_results):
        """
        Maps OCR results to the closest box centers using Euclidean distance.

        Args:
            box_centers (list): List of (x, y) coordinates for detected box centers.
            ocr_centers (list): List of (x, y) coordinates for OCR text positions.
            ocr_results (list): Corresponding OCR results for each OCR center.

        Returns:
            dict: Mapping of OCR results to the closest box centers.
        """
        print("OCR centers:", ocr_centers)
        print("Box Centers:", box_centers)

        if not ocr_centers or not box_centers:
            return {}

        # Convert lists to numpy arrays
        ocr_centers = np.asarray(ocr_centers).reshape(-1, 2)
        box_centers = np.asarray(box_centers).reshape(-1, 2)

        # Compute pairwise distances between OCR centers and box centers
        distances = cdist(ocr_centers, box_centers)

        # Map each OCR center to the closest box center
        mapped_ocr = {
            i: {
                'box_center': box_centers[np.argmin(distances[i])].tolist(),
                'ocr_data': ocr_results[i]
            }
            for i in range(len(ocr_centers))
        }

        print("Mapped OCR:", mapped_ocr)
        return mapped_ocr


    def map_qr_to_boxes(self, box_centers, qr_centers, qr_data):
        """
        Maps QR codes to the closest box centers using Euclidean distance.

        Args:
            box_centers (list): List of (x, y) coordinates for detected box centers.
            qr_centers (list): List of (x, y) coordinates for QR code positions.
            qr_data (list): Corresponding QR code data for each QR center.

        Returns:
            dict: Mapping of QR codes to the closest box centers.
        """
        print("QR Boxes:", qr_centers)
        print("Box Centers:", box_centers)

        if not qr_centers or not box_centers:
            return {}

        # Convert lists to numpy arrays
        qr_centers = np.asarray(qr_centers).reshape(-1, 2)
        box_centers = np.asarray(box_centers).reshape(-1, 2)

        # Compute pairwise distances between QR centers and box centers
        distances = cdist(qr_centers, box_centers)

        # Map each QR center to the closest box center
        mapped_qrs = {
            i: {
                'box_center': box_centers[np.argmin(distances[i])].tolist(),
                'qr_data': qr_data[i]
            }
            for i in range(len(qr_centers))
        }

        print("Mapped QRs:", mapped_qrs)
        return mapped_qrs


    def log(self, q1, q2, q3, q4, validator):
        """Enhanced log function using CSV validation for adjacent racks."""

        def get_rack_position(rack, position):
            """Helper function to retrieve rack positions (upper/lower) safely."""
            return validator.get_rack_positions().get(rack, {}).get(position)

        def get_adjacent_rack(rack, direction):
            """Helper function to retrieve left or right adjacent racks."""
            return getattr(validator, f"get_{direction}_adjacent_rack")(rack)

        def is_even(rack):
            """Helper function to check if rack number is even."""
            return int(rack[3:5]) % 2 == 0

        # Validate and infer lower racks
        if q1 != "Unable to decode" and q2 != "Unable to decode":
            lower_q1, lower_q2 = get_rack_position(q1, "lower"), get_rack_position(q2, "lower")

            if q3 == "Unable to decode" and q4 != "Unable to decode" and lower_q1 == q4:
                q3 = lower_q2
            elif q4 == "Unable to decode" and q3 != "Unable to decode" and lower_q2 == q3:
                q4 = lower_q1
            elif q3 == "Unable to decode" and q4 == "Unable to decode":
                q3, q4 = lower_q2, lower_q1

        # Validate and infer upper racks
        if q3 != "Unable to decode" and q4 != "Unable to decode":
            upper_q3, upper_q4 = get_rack_position(q3, "upper"), get_rack_position(q4, "upper")

            if q1 == "Unable to decode" and q2 != "Unable to decode" and upper_q3 == q2:
                q1 = upper_q4
            elif q2 == "Unable to decode" and q1 != "Unable to decode" and upper_q4 == q1:
                q2 = upper_q3
            elif q1 == "Unable to decode" and q2 == "Unable to decode":
                q1, q2 = upper_q4, upper_q3

        # Infer missing adjacent racks based on parity
        adjacent_cases = [
            (q1, q4, "right", "left", "even"),
            (q2, q3, "left", "right", "even"),
            (q1, q3, "lower", "upper", None),
            (q4, q2, "upper", "lower", None),
            (q1, None, "right", "left", "even"),
            (q2, None, "left", "right", "even"),
            (q3, None, "left", "right", "even"),
            (q4, None, "right", "left", "even"),
        ]

        for fixed, inferred, right_dir, left_dir, parity in adjacent_cases:
            if fixed != "Unable to decode" and q2 == "Unable to decode" and q3 == "Unable to decode" and q4 == "Unable to decode":
                if parity is None or is_even(fixed) == (parity == "even"):
                    q2 = get_adjacent_rack(fixed, right_dir)
                else:
                    q2 = get_adjacent_rack(fixed, left_dir)
                q3 = get_rack_position(q2, "lower")
                q4 = get_rack_position(q1, "lower")

        return q1, q2, q3, q4


    def generate_exclusion(self, left_box_count, right_box_count, box_centers, mapped_qrs, center_x, mapped_ocr):
        """Generates exclusion messages for missing or problematic box stickers."""

        def get_assigned_boxes(mapped_entries):
            """Extracts assigned box centers into left and right sets."""
            left, right = set(), set()
            if mapped_entries:
                for entry in mapped_entries.values():
                    box_center = tuple(entry['box_center'])
                    (left if box_center[0] < center_x else right).add(box_center)
            return left, right

        # Get assigned boxes from QR and OCR mappings
        assigned_left_qr, assigned_right_qr = get_assigned_boxes(mapped_qrs)
        assigned_left_ocr, assigned_right_ocr = get_assigned_boxes(mapped_ocr)

        # Combine assigned boxes
        assigned_boxes_left = assigned_left_qr | assigned_left_ocr
        assigned_boxes_right = assigned_right_qr | assigned_right_ocr

        # Calculate missing box counts
        left_missing = left_box_count - len(assigned_boxes_left)
        right_missing = right_box_count - len(assigned_boxes_right)

        def generate_message(box_count, missing):
            """Generates exclusion messages based on missing sticker count."""
            if box_count > 0 and missing > 0:
                return (
                    f"There are {box_count} boxes, but there is a problem with the sticker of one box."
                    if missing == 1 else
                    f"There are {box_count} boxes, but there is a problem with the stickers of {missing} boxes."
                )
            return ""

        return generate_message(left_box_count, left_missing), generate_message(right_box_count, right_missing)

