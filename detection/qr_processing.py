import cv2
import numpy as np
from typing import List, Tuple, Dict
from pyzbar.pyzbar import decode, ZBarSymbol
from detection.utils import Utilities
from detection.image_processing import ImageProcessor
from config.settings import CONFIG
from ultralytics import YOLO


class QRProcessor:
    def __init__(self):
        self.util = Utilities()
        self.img_processor = ImageProcessor()

        self.qr_model = YOLO(CONFIG['models']['qr_model'])

    def determine_quadrant(self,img,coordinate):
        height, width, _ = img.shape

        center_x, center_y = self.util.find_image_center(img)
        quadrants = {
            'Q1': {'x_min': center_x, 'x_max': width, 'y_min': 0, 'y_max': center_y},
            'Q2': {'x_min': 0, 'x_max': center_x, 'y_min': 0, 'y_max': center_y},
            'Q3': {'x_min': 0, 'x_max': center_x, 'y_min': center_y, 'y_max': height},
            'Q4': {'x_min': center_x, 'x_max': width, 'y_min': center_y, 'y_max': height}
        }
        x, y = coordinate
        for quad, bounds in quadrants.items():
            if bounds['x_min'] <= x <= bounds['x_max'] and bounds['y_min'] <= y <= bounds['y_max']:
                return quad
        return None

    def extract_qr_code_data(
        self, 
        img_path: str,  
        left_roi_x: int, 
        right_roi_x: int, 
        upper_roi_y: int, 
        lower_roi_y: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Detects and decodes QR codes from an image and classifies them as rack or box codes.

        Args:
            img_path (str): Path to the input image.
            model: Trained YOLO model for QR detection.
            left_roi_x (int): Left boundary of the ROI.
            right_roi_x (int): Right boundary of the ROI.
            upper_roi_y (int): Upper boundary for box QR codes.
            lower_roi_y (int): Lower boundary for box QR codes.

        Returns:
            Tuple[List[Dict], List[Dict]]: Lists of decoded rack and box QR code info.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at: {img_path}")
        
        center_x, center_y = self.util.find_image_center(img)
        detections = self._get_qr_code_detections(img, center_x, center_y, left_roi_x, right_roi_x)

        selected_qr_codes = self._select_closest_per_quadrant(detections, center_x, center_y)

        rack_qrs, box_qrs = [], []

        for qr in selected_qr_codes:
            result = self._process_qr_code(qr, img, center_x, center_y, upper_roi_y, lower_roi_y)
            if result["barcode_data"].startswith("HD"):
                rack_id = result["barcode_data"]
                x1, y1, x2, y2 = result['coordinates']
                rack_center = ((x1+x2)//2, (y1+y2)//2)

                quad = self.determine_quadrant(img,rack_center)

                rack_qrs.append((rack_id,quad))

            elif result["barcode_data"].startswith("Box"):
                box_info = result['barcode_data']
                start_idx = box_info.find('@')
                unique_id = box_info[start_idx:start_idx+6]
                x1, y1, x2, y2 = result['coordinates']
                box_center = ((x1+x2)//2, (y1+y2)//2)
                box_qrs.append((unique_id,box_center))

        return rack_qrs, box_qrs


    def _get_qr_code_detections(
        self, 
        img, 
        center_x: int, 
        center_y: int, 
        left_roi_x: int, 
        right_roi_x: int
    ) -> List[Tuple[int, int, int, int, int, int, float]]:
        """
        Runs the YOLO model and filters QR code detections within horizontal ROI.

        Returns:
            List of QR code bounding box data including distance from center.
        """
        results = self.qr_model.predict(img, conf=0.4, verbose = False)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                xmin, ymin, xmax, ymax = map(int, box)
                center_x_qr = (xmin + xmax) // 2
                center_y_qr = (ymin + ymax) // 2

                if left_roi_x <= center_x_qr <= right_roi_x:
                    distance = self.util.calculate_distance(center_x, center_y, center_x_qr, center_y_qr)
                    detections.append((xmin, ymin, xmax, ymax, center_x_qr, center_y_qr, distance))

        return detections


    def _select_closest_per_quadrant(
        self, 
        qr_codes: List[Tuple], 
        center_x: int, 
        center_y: int
    ) -> List[Tuple]:
        """
        Organizes QR codes by quadrant and selects the closest one in each.

        Returns:
            List of selected QR code bounding boxes.
        """
        quadrants = {
            1: [qr for qr in qr_codes if qr[4] >= center_x and qr[5] < center_y],  # Top-right
            2: [qr for qr in qr_codes if qr[4] < center_x and qr[5] < center_y],   # Top-left
            3: [qr for qr in qr_codes if qr[4] < center_x and qr[5] >= center_y],  # Bottom-left
            4: [qr for qr in qr_codes if qr[4] >= center_x and qr[5] >= center_y]  # Bottom-right
        }

        return [min(qr_list, key=lambda x: x[6]) for qr_list in quadrants.values() if qr_list]
        #return quadrants.values()


    def _process_qr_code(
        self,
        qr_code: Tuple[int, int, int, int, int, int, float],
        img: np.ndarray,
        center_x: int,
        center_y: int,
        upper_roi_y: int,
        lower_roi_y: int
    ) -> Dict:
        """
        Extracts ROI for a QR code, enhances and decodes it.

        Returns:
            Dictionary with QR code metadata.
        """
        xmin, ymin, xmax, ymax, center_x_qr, center_y_qr, _ = qr_code

        # Slightly expand bounding box
        pad = 10
        roi = img[
            max(0, ymin - pad):min(img.shape[0], ymax + pad),
            max(0, xmin - pad):min(img.shape[1], xmax + pad)
        ]

        enhanced_images = self.img_processor.enhance_image(roi)
        barcode_data = self._decode_qr_code(enhanced_images)

        return {
            "barcode_data": barcode_data,
            "coordinates": (xmin, ymin, xmax, ymax),
            "decoded": barcode_data != "Unable to decode",
            "quadrant": self.util.determine_quadrant(center_x, center_y, center_x_qr, center_y_qr)
        }


    def _decode_qr_code(
        self, 
        enhancements: List[Tuple[str, np.ndarray]]
        ) -> str:
        """
        Attempts to decode a QR code from multiple enhanced versions.

        Returns:
            Decoded string or failure message.
        """
        for _, img in enhancements:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            min_dim = min(gray.shape[:2])
            block_size = max(3, (min_dim // 3) | 1)

            thresh_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2
            )

            decoded = decode(thresh_img, symbols=[ZBarSymbol.QRCODE])
            if decoded:
                return decoded[0].data.decode("utf-8")

        return "Unable to decode"


    def box_qr(
        self, 
        qr_results: List[Dict]
    ) -> Tuple[List[Tuple[float, float]], List[str]]:
        """
        Extracts center points and data from QR code results.

        Returns:
            Tuple of centers and decoded strings.
        """
        centers = [
            ((qr["coordinates"][0] + qr["coordinates"][2]) / 2,
             (qr["coordinates"][1] + qr["coordinates"][3]) / 2)
            for qr in qr_results
        ]
        data = [qr["barcode_data"] for qr in qr_results]

        return centers, data
