import re
import io
import cv2
import os
import numpy as np
from PIL import Image
from google.cloud import vision
from detection.correction import Corrector


class ImageProcessor:
    def __init__(self):
        pass


    def enhance_image(self, image):
        """
        Enhances the input image using various image processing techniques.
        
        :param image: Input image (grayscale or color)
        :return: List of tuples containing enhancement name and processed image
        """
        enhancements = []
        
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        enhancements.append(('grayscale', gray))
        
        # Histogram Equalization
        enhancements.append(('equalized', cv2.equalizeHist(gray)))
        
        # Gaussian Blurring with different kernel sizes
        for ksize in [3, 5, 7]:
            enhancements.append((f'blurred_{ksize}', cv2.GaussianBlur(image, (ksize, ksize), 0)))
        
        # Sharpening
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhancements.append(('sharpened', cv2.filter2D(image, -1, kernel_sharpen)))
        
        # Adaptive Thresholding with different block sizes
        for block_size in [11, 15, 19]:
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2
            )
            enhancements.append((f'adaptive_threshold_{block_size}', adaptive_thresh))
        
        # Binary Thresholding
        _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        enhancements.append(('binary', binary_image))
        
        # Morphological Operations
        kernel = np.ones((5, 5), np.uint8)
        enhancements.extend([
            ('dilated', cv2.dilate(binary_image, kernel, iterations=1)),
            ('eroded', cv2.erode(binary_image, kernel, iterations=1))
        ])
        
        return enhancements


    def resize_image(self, image_path, max_size=1024):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")

        height, width = image.shape[:2]

        if max(height, width) > max_size:
            scaling_factor = max_size / float(max(height, width))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            return resized_image
        return image


    def draw_bounding_boxes(self, results, left_line_x, right_line_x, original_size, resized_size, roi_top, roi_bottom, threshold=0.80):
        """
        Draws bounding boxes and counts objects based on their positions.

        Args:
            results (List): YOLO model detection results.
            left_line_x (float): X-coordinate of the left boundary.
            right_line_x (float): X-coordinate of the right boundary.
            original_size (Tuple[int, int]): (Height, Width) of the original image.
            resized_size (Tuple[int, int]): (Height, Width) of the resized image.
            roi_top (float): Top Y-coordinate of the region of interest.
            roi_bottom (float): Bottom Y-coordinate of the region of interest.
            threshold (float, optional): Confidence threshold for filtering detections. Defaults to 0.80.

        Returns:
            Tuple[int, int, List]: (Left count, Right count, List of bounding box centers)
        """
        # Compute scaling factors for coordinate transformation
        orig_height, orig_width = original_size
        resized_height, resized_width = resized_size
        scale_x, scale_y = orig_width / resized_width, orig_height / resized_height

        # Compute image center based on defined left and right boundaries
        image_center_x = (left_line_x + right_line_x) / 2

        left_count, right_count = 0, 0
        box_centers = []

        for i, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for j, (box, score) in enumerate(zip(boxes, scores)):
                if score < threshold:
                    continue  # Skip low-confidence detections

                xmin, ymin, xmax, ymax = box
                # Scale coordinates back to original image size
                xmin_orig, ymin_orig = xmin * scale_x, ymin * scale_y
                xmax_orig, ymax_orig = xmax * scale_x, ymax * scale_y

                center_x_box = (xmin_orig + xmax_orig) / 2
                center_y_box = (ymin_orig + ymax_orig) / 2

                # Check if bounding box is within the defined ROI
                if roi_top <= center_y_box <= roi_bottom and left_line_x <= center_x_box <= right_line_x:
                    box_centers.append((center_x_box, center_y_box))

                    #print(f"[{i}-{j}] Bounding Box Center: ({center_x_box:.2f}, {center_y_box:.2f})")

                    # Count objects based on their position relative to the image center
                    if center_x_box <= image_center_x:
                        left_count += 1
                        #print(f"    -> Left bbox detected: Center at {center_x_box:.2f}")
                    else:  # Right side
                        right_count += 1
                        #print(f"    -> Right bbox detected: Center at {center_x_box:.2f}")

        # Summary of detections
        # print(f"\nDetection Summary:")
        # print(f"  - Left Count: {left_count}")
        # print(f"  - Right Count: {right_count}")
        # print(f"  - Total Bounding Boxes: {left_count + right_count}")

        return left_count, right_count, box_centers


    def process_and_extract_pairs(self, image_path, left_roi_x, right_roi_x, top_roi_y, bottom_roi_y):
        """
        Extracts text pairs from an image using Google Vision API and organizes them into structured data.

        Args:
            image_path (str): Path to the image file.
            left_roi_x (int): X-coordinate of the left boundary of the ROI.
            right_roi_x (int): X-coordinate of the right boundary of the ROI.
            top_roi_y (int): Y-coordinate of the top boundary of the ROI.
            bottom_roi_y (int): Y-coordinate of the bottom boundary of the ROI.

        Returns:
            Tuple[List[Tuple[str, str, str, str, List[Tuple[int, int]]]], float]:
                - A list of tuples containing:
                    (box_number, fixed_part_number, invoice_number, quantity, bounding_box).
                - The computed image center X-coordinate.
        """
        corrector = Corrector()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        height, width, _ = image.shape

        # Convert image to bytes for Google Vision API
        _, image_encoded = cv2.imencode('.jpg', image)
        content = image_encoded.tobytes()

        # Initialize Google Vision client and perform text detection
        client = vision.ImageAnnotatorClient()
        response = client.text_detection(image=vision.Image(content=content))
        annotations = response.text_annotations

        # Define patterns for text classification
        patterns = {
            "box_number": re.compile(r'^\d+/\d+$'),
            "part_number": re.compile(r'^[A-Za-z0-9]{8,15}$'),
            "numeric": re.compile(r'^\d{1,6}$')
        }

        # Store detected texts categorized by type
        slash_texts, alphanumeric_texts, numeric_texts = [], [], []

        # Process detected text and extract bounding boxes
        for annotation in annotations[1:]:  # Skip the first annotation (full text)
            text = annotation.description
            vertices = annotation.bounding_poly.vertices
            bbox = [(v.x, v.y) for v in vertices]

            x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])
            bbox_center_x, bbox_center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Check if text is within the ROI
            if left_roi_x <= bbox_center_x <= right_roi_x and top_roi_y <= bbox_center_y <= bottom_roi_y:
                if patterns["box_number"].match(text):
                    slash_texts.append((text, bbox, bbox_center_x, bbox_center_y))
                elif patterns["part_number"].match(text) and any(char.isdigit() for char in text):
                    alphanumeric_texts.append((text, bbox))
                elif patterns["numeric"].match(text):
                    numeric_texts.append((text, bbox, bbox_center_x, bbox_center_y))

        pairs = []

        # Process each detected box number
        for slash_text, slash_bbox, slash_center_x, slash_center_y in slash_texts:
            # Find the closest part number
            closest_text = min(
                alphanumeric_texts,
                key=lambda a: ((slash_center_x - ((a[1][0][0] + a[1][2][0]) // 2)) ** 2 +
                            (slash_center_y - ((a[1][0][1] + a[1][2][1]) // 2)) ** 2) ** 0.5,
                default=(None, None)
            )[0]

            # Find the closest numeric values (invoice & quantity)
            numeric_distances = sorted(
                [(num_text, ((slash_center_x - num_x) ** 2 + (slash_center_y - num_y) ** 2) ** 0.5)
                for num_text, _, num_x, num_y in numeric_texts if num_y > slash_center_y],
                key=lambda x: x[1]
            )

            # Extract first and second closest numeric values
            closest_invoice = numeric_distances[0][0] if len(numeric_distances) >= 1 else "missing invoice"
            box_quantity = numeric_distances[1][0] if len(numeric_distances) >= 2 else "missing quantity"

            # Apply part number correction
            fixed_alphanumeric_text = corrector.fix_part_number(closest_text) if closest_text else "No match found"

            # Store extracted information
            pairs.append((slash_text, fixed_alphanumeric_text, closest_invoice, box_quantity, slash_bbox))

        # Compute the image's center X-coordinate
        image_center_x = (left_roi_x + right_roi_x) / 2

        return pairs, image_center_x


    def detect_and_consolidate_hd_entries(self, image_path, left_roi_x, right_roi_x, area_threshold=1000):
        """
        Detects and consolidates HD entries from an image using Google Cloud Vision API.

        Args:
            image_path (str): Path to the image file.
            left_roi_x (int): X-coordinate of the left boundary of the ROI.
            right_roi_x (int): X-coordinate of the right boundary of the ROI.
            area_threshold (int, optional): Minimum area for an HD text entry to be considered. Defaults to 1000.

        Returns:
            List[Tuple[List[List[int]], str, float]]: A list of tuples containing:
                - Bounding box coordinates [[x1, y1], [x2, y1], [x2, y2], [x1, y2]].
                - Consolidated HD text.
                - Average confidence score.
        """
        # Initialize Google Cloud Vision client
        client = vision.ImageAnnotatorClient()

        # Load image file
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
        except FileNotFoundError:
            raise ValueError(f"File not found: {image_path}")

        # Perform OCR using Google Cloud Vision API
        image = vision.Image(content=content)
        response = client.text_detection(image=image)

        if response.error.message:
            raise Exception(f"Google Cloud Vision API error: {response.error.message}")

        text_annotations = response.text_annotations[1:]  # Skip the first annotation (full text)

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
        while i < len(text_annotations):
            annotation = text_annotations[i]
            text = annotation.description
            min_x, min_y, max_x, max_y, text_center_x, text_area = get_text_details(annotation)

            # Only process text if its center falls within the ROI
            if left_roi_x <= text_center_x <= right_roi_x:
                # Identify HD text
                is_hd_text = text.startswith("HD") or (len(text) == 4 and "HD" in text)

                if is_hd_text and (area_threshold is None or text_area >= area_threshold):
                    consolidated_text = "HD"

                    # Initialize bounding box and confidence score
                    overall_min_x, overall_min_y = min_x, min_y
                    overall_max_x, overall_max_y = max_x, max_y
                    confidence_score = getattr(annotation, 'confidence', 0.5)  # Default to 0.5 if confidence is unavailable

                    # Consolidate next 6 words
                    words_to_consolidate = min(6, len(text_annotations) - (i + 1))
                    consolidated_count = 1

                    for j in range(1, words_to_consolidate + 1):
                        next_annotation = text_annotations[i + j]
                        next_text = next_annotation.description
                        next_min_x, next_min_y, next_max_x, next_max_y, next_center_x, _ = get_text_details(next_annotation)

                        # Ensure the next text is within ROI
                        if left_roi_x <= next_center_x <= right_roi_x:
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
                    coordinates = [
                        [overall_min_x, overall_min_y],
                        [overall_max_x, overall_min_y],
                        [overall_max_x, overall_max_y],
                        [overall_min_x, overall_max_y]
                    ]

                    hd_entries.append((coordinates, consolidated_text, confidence_score))
                    i += consolidated_count  # Skip consolidated words
                else:
                    i += 1  # Move to the next annotation
            else:
                i += 1  # Move to the next annotation

        return hd_entries