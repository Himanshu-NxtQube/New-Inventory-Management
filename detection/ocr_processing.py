from google.cloud import vision
from detection.utils import Utilities
import cv2
import re
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "models/GoogleVisionCredential.json"

class OCRProcessor:
    def __init__(self):
        self.util = Utilities()

    def extract_ocr_info(self, image_path, left_line_x, right_line_x, upper_line_y, lower_line_y):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        self.height, self.width, _ = image.shape
        self.center_x, self.center_y = self.util.find_image_center(image)

        # Convert image to bytes for Google Vision API
        _, image_encoded = cv2.imencode('.jpg', image)
        content = image_encoded.tobytes()

        # Initialize Google Vision client and perform text detection
        client = vision.ImageAnnotatorClient()
        response = client.text_detection(image=vision.Image(content=content))
        annotations = response.text_annotations

        rack_data = self.extract_rack_info(annotations, left_line_x, right_line_x)

        box_data = self.extract_box_info(annotations, left_line_x, right_line_x, upper_line_y, lower_line_y)

        return rack_data, box_data


    def extract_box_info(self, annotations, left_line_x, right_line_x, upper_line_y, lower_line_y):
        """
        Extracts unique ids from an image using Google Vision API with co-ordinates.

        Args:
            image_path (str): Path to the image file.
            left_line_x (int): X-coordinate of the left boundary of the ROI.
            right_line_x (int): X-coordinate of the right boundary of the ROI.
            upper_line_y (int): Y-coordinate of the top boundary of the ROI.
            lower_line_y (int): Y-coordinate of the bottom boundary of the ROI.

        Returns:
            List[Tuple[Str, Tuple(int, int)]]:
                - Extracted Unique ID.
                - The computed coordinates of extracted unique IDs.
        """
        # corrector = Corrections()

        # unique_id_pattern = re.compile(r'^@[A-Za-z0-9]{5}$')

        uids = []
        # Process detected text and extract bounding boxes
        # for annotation in annotations[1:]:  # Skip the first annotation (full text)
        #     text = annotation.description 
        #     vertices = annotation.bounding_poly.vertices
        #     bbox = [(v.x, v.y) for v in vertices]

        #     x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])
        #     bbox_center_x, bbox_center_y = (x1 + x2) // 2, (y1 + y2) // 2

        #     # Check if text is within the ROI
        #     if left_line_x <= bbox_center_x <= right_line_x and upper_line_y <= bbox_center_y <= lower_line_y:
        #         if unique_id_pattern.match(text):
        #             unique_id_data.append((text, (bbox_center_x, bbox_center_y)))
        # return unique_id_data

        # unique_ids = set()
        i = 1  # Skip the first annotation (full text)

        while i < len(annotations):
            text = annotations[i].description.strip()
            vertices = annotations[i].bounding_poly.vertices
            bbox = [(v.x, v.y) for v in vertices]

            x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])
            bbox_center_x, bbox_center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Check if text is within the ROI
            if left_line_x <= bbox_center_x <= right_line_x and upper_line_y <= bbox_center_y <= lower_line_y:
                
                # Case 1: ID is complete like "@A2451"
                if text.startswith("@") and len(text) == 6:
                    uids.append((text, (bbox_center_x, bbox_center_y)))

                # Case 2: '@' is alone and next item is 5 chars
                elif text == "@" and i + 1 < len(annotations):
                    next_token = annotations[i + 1].description.strip()
                    if len(next_token) == 5:
                        uids.append((f"@{next_token}", (bbox_center_x, bbox_center_y)))
                        i += 1  # Skip next because it's already consumed

            i += 1

        return uids

    def determine_quadrant(self,coordinate):

        quadrants = {
            'Q1': {'x_min': self.center_x, 'x_max': self.width, 'y_min': 0, 'y_max': self.center_y},
            'Q2': {'x_min': 0, 'x_max': self.center_x, 'y_min': 0, 'y_max': self.center_y},
            'Q3': {'x_min': 0, 'x_max': self.center_x, 'y_min': self.center_y, 'y_max': self.height},
            'Q4': {'x_min': self.center_x, 'x_max': self.width, 'y_min': self.center_y, 'y_max': self.height}
        }
        x, y = coordinate
        for quad, bounds in quadrants.items():
            if bounds['x_min'] <= x <= bounds['x_max'] and bounds['y_min'] <= y <= bounds['y_max']:
                return quad
        return None
    
    def extract_rack_info(self, annotations, left_line_x, right_line_x, area_threshold=1000):
        """
        Detects and consolidates HD entries from an image using Google Cloud Vision API.

        Args:
            image_path (str): Path to the image file.
            left_line_x (int): X-coordinate of the left boundary of the ROI.
            right_line_x (int): X-coordinate of the right boundary of the ROI.
            area_threshold (int, optional): Minimum area for an HD text entry to be considered. Defaults to 1000.

        Returns:
            List[Tuple[str, Tuple[Tuple[int]], float]]: A list of tuples containing:
                - Consolidated HD text.
                - Bounding box coordinates centers (x, y).
        """

        hd_entries = []

        def get_text_details(annotation):
            """Extracts bounding box, area, and center coordinates from a text annotation."""
            vertices = annotation.bounding_poly.vertices
            min_x, min_y = min(v.x for v in vertices), min(v.y for v in vertices)
            max_x, max_y = max(v.x for v in vertices), max(v.y for v in vertices)
            text_center_x = sum(v.x for v in vertices) / 4  # Average of all vertices
            text_area = (max_x - min_x) * (max_y - min_y)
            return (min_x, min_y, max_x, max_y, text_center_x, text_area)

        i = 0
        while i < len(annotations):
            annotation = annotations[i]
            text = annotation.description
            min_x, min_y, max_x, max_y, text_center_x, text_area = get_text_details(annotation)

            # Only process text if its center falls within the ROI
            if left_line_x <= text_center_x <= right_line_x:
                # Identify HD text
                is_hd_text = text.startswith("HD") or (len(text) == 4 and "HD" in text)

                if is_hd_text and (area_threshold is None or text_area >= area_threshold):
                    consolidated_text = "HD"

                    # Initialize bounding box and confidence score
                    overall_min_x, overall_min_y = min_x, min_y
                    overall_max_x, overall_max_y = max_x, max_y
                    confidence_score = getattr(annotation, 'confidence', 0.5)  # Default to 0.5 if confidence is unavailable

                    # Consolidate next 6 words
                    words_to_consolidate = min(6, len(annotations) - (i + 1))
                    consolidated_count = 1

                    for j in range(1, words_to_consolidate + 1):
                        next_annotation = annotations[i + j]
                        next_text = next_annotation.description
                        next_min_x, next_min_y, next_max_x, next_max_y, next_center_x, _ = get_text_details(next_annotation)

                        # Ensure the next text is within ROI
                        if left_line_x <= next_center_x <= right_line_x:
                            consolidated_text += next_text
                            consolidated_count += 1

                            # Expand bounding box
                            overall_min_x = min(overall_min_x, next_min_x)
                            overall_min_y = min(overall_min_y, next_min_y)
                            overall_max_x = max(overall_max_x, next_max_x)
                            overall_max_y = max(overall_max_y, next_max_y)

                            # Update confidence score
                            next_confidence = getattr(next_annotation, 'confidence', 0.5)
                            confidence_score = (confidence_score * (consolidated_count - 1) + next_confidence) / consolidated_count
                        else:
                            break  # Stop consolidation if next text is out of ROI

                    # Create bounding box in [top-left, top-right, bottom-right, bottom-left] order
                    # coordinates = [
                    #     [overall_min_x, overall_min_y],
                    #     [overall_max_x, overall_min_y],
                    #     [overall_max_x, overall_max_y],
                    #     [overall_min_x, overall_max_y]
                    # ]
                    coordinates = ((overall_min_x + overall_max_x)//2, (overall_min_y + overall_max_y)//2)

                    quad = self.determine_quadrant(coordinates)
                    # hd_entries.append((coordinates, consolidated_text, confidence_score))
                    hd_entries.append((consolidated_text, quad))
                    i += consolidated_count  # Skip consolidated words
                else:
                    i += 1  # Move to the next annotation
            else:
                i += 1  # Move to the next annotation

        return hd_entries