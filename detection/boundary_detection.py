import cv2
import numpy as np
from typing import Tuple, List
from ultralytics import YOLO
from config.settings import CONFIG
import os
from contextlib import redirect_stdout


class BoundaryDetector:
    def __init__(self):
        """Initializes models and thresholds from the configuration."""
        self.blue_bar_model = YOLO(CONFIG['models']['blue_bar_model'])
        self.orange_bar_model = YOLO(CONFIG['models']['orange_bar_model'])

        self.blue_threshold = CONFIG['thresholds']['blue_bar_model']['confidence_threshold']
        self.blue_merge_threshold = CONFIG['thresholds']['blue_bar_model']['merge_threshold']
        self.orange_threshold = CONFIG['thresholds']['orange_bar_model']['confidence_threshold']


    def get_orange_bar_boundaries(
        self, 
        image_path: str
    ) -> Tuple[int, int]:
        """
        Detects the top and bottom Y-coordinates of the orange bar using a YOLO model.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[int, int]: Upper and lower Y-coordinate boundaries of the detected orange bar.
        """
        image = self._load_image(image_path)
        image_height = image.shape[0]

        detections = self._predict_and_filter(self.orange_bar_model, image, self.orange_threshold)

        y_mid = image_height // 2
        upper = [box for box in detections if box[1] <= y_mid]
        lower = [box for box in detections if box[1] >= y_mid]

        upper_y = int(min(upper, key=lambda b: b[1])[1]) if upper else 0
        lower_y = int(max(lower, key=lambda b: b[1])[1]) if lower else image_height

        return upper_y, lower_y


    def get_blue_bar_boundaries(
        self, 
        image_path: str
    ) -> Tuple[int, int]:
        """
        Detects the left and right X-coordinates of the blue bar using a YOLO model.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[int, int]: Left and right X-coordinate boundaries of the detected blue bar.
        """
        image = self._load_image(image_path)
        img_width = image.shape[1]

        
        results = self.blue_bar_model(image, verbose = False)

        centers = self._extract_bbox_centers_x(results, self.blue_threshold)
        merged_centers = self._merge_close_centers(centers, self.blue_merge_threshold)

        if len(merged_centers) >= 2:
            return merged_centers[0], merged_centers[-1]
        else:
            return 0, img_width - 1  # Fallback: full image width


    def _load_image(
        self, 
        path: str
    ) -> np.ndarray:
        """
        Loads an image from the provided path.

        Args:
            path (str): Path to the image.

        Returns:
            np.ndarray: Loaded image.

        Raises:
            ValueError: If the image cannot be loaded.
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Unable to load image from path: {path}")
        return img


    def _predict_and_filter(
        self, 
        model: YOLO, 
        image: np.ndarray, 
        conf_threshold: float
    ) -> List[np.ndarray]:
        """
        Runs prediction on the image and filters detections by confidence.

        Args:
            model (YOLO): Trained YOLO model.
            image (np.ndarray): Input image.
            conf_threshold (float): Confidence threshold.

        Returns:
            List[np.ndarray]: List of filtered bounding boxes.
        """
        results = model.predict(image, verbose = False)

        boxes = results[0].boxes.data.cpu().numpy()
        return [box for box in boxes if box[4] > conf_threshold]  # box[4] = confidence


    def _extract_bbox_centers_x(
        self, 
        results, 
        threshold: float
    ) -> List[int]:
        """
        Extracts X-centers of bounding boxes above a confidence threshold.

        Args:
            results: YOLO model results.
            threshold (float): Confidence threshold.

        Returns:
            List[int]: List of center X-coordinates.
        """
        centers = []
        for box in results[0].boxes:
            if box.conf.item() >= threshold:
                x1, _, x2, _ = map(int, box.xyxy[0])
                centers.append((x1 + x2) // 2)
        return sorted(centers)


    def _merge_close_centers(
        self, 
        centers: List[int], 
        min_distance: int
    ) -> List[int]:
        """
        Merges bounding box centers that are closer than a threshold.

        Args:
            centers (List[int]): Sorted list of center X-coordinates.
            min_distance (int): Minimum distance to treat centers as distinct.

        Returns:
            List[int]: Merged list of center X-coordinates.
        """
        merged = []
        for c in centers:
            if not merged or c - merged[-1] > min_distance:
                merged.append(c)
        return merged
