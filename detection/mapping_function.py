import os
import json
import cv2
import ast
from typing import Any, Tuple, List, Dict, Optional
from collections import defaultdict
from ultralytics import YOLO

class RecordMapper:
    def __init__(
        self,
        image_path: str,
        records: Any,  # Either List[Dict] or List[List[Dict]]
        rack_dict: Dict[str, str],
        lower_bar_y2: float,
        vertical_threshold: float = 200,
        conf_threshold: float = 0.5,
    ):
        self.image_path = image_path
        self.rack_dict = rack_dict
        self.image_id = os.path.splitext(os.path.basename(image_path))[0]
        self.lower_bar_y2 = lower_bar_y2
        self.vertical_threshold = vertical_threshold
        self.conf_threshold = conf_threshold
        self.model_path = '/Users/saieshagre/Downloads/New-Inventory-Management/models/pallet1.pt'

        self._load_image_center()

        if isinstance(records, list) and records and not isinstance(records[0], list):
            self.records = [records]
            self._single = True
        else:
            self.records = records  # type: ignore
            self._single = False


    def _load_image_center(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        _, width = img.shape[:2]
        self.center_x = width / 2.0


    def _normalize_coord(self, coord_raw: Any) -> Tuple[float, float]:
        if isinstance(coord_raw, str):
            coord = ast.literal_eval(coord_raw)
        else:
            coord = coord_raw
        if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
            raise ValueError(f"Invalid coordinates {coord_raw!r}: expected a pair (x,y).")
        return float(coord[0]), float(coord[1])


    def _rack_id_for_coord(self, coord_raw: Any) -> Optional[str]:
        x, _ = self._normalize_coord(coord_raw)
        quadrant = "Q3" if x < self.center_x else "Q4"
        rev = {v: k for k, v in self.rack_dict.items()}
        return rev.get(quadrant)


    def _count_pallets_by_side(self) -> Dict[str, int]:
        model = YOLO(self.model_path)
        results = model(self.image_path, conf=self.conf_threshold)
        detections = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        side_counts = {'left': 0, 'right': 0}

        for box, conf in zip(detections, confidences):
            if float(conf) < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            if abs(y2 - self.lower_bar_y2) > self.vertical_threshold:
                continue
            center_x = (x1 + x2) / 2
            side = 'left' if center_x < self.center_x else 'right'
            side_counts[side] += 1

        return side_counts


    def _map_single(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        unique_id = rec['uniqueId']
        boxnum = rec['box_number']
        boxqty = rec['box_quantity']
        partnum = rec['part_number']
        invnum = rec['invoice_number']
        coord_raw = rec['coordinates']
        rack_id = self._rack_id_for_coord(coord_raw)

        return {
            "IMG_ID":         self.image_id,
            "RACK_ID":        rack_id,
            "UNIQUE_ID":      unique_id,
            "BOXNUMBER":      boxnum,
            "BOXQUANTITY":    boxqty,
            "PARTNUMBER":     partnum,
            "INVOICE_NUMBER": invnum
        }


    # def _map_batch(self) -> List[Dict[str, Any]]:
    #     if not self.records:
    #         pallet_counts = self._count_pallets_by_side()
    #         outputs = []
    #         for rack_id, quadrant in self.rack_dict.items():
    #             side = 'left' if quadrant == 'Q3' else 'right'
    #             p = pallet_counts.get(side, 0)
    #             excl = None if p == 0 else f"There are {p} pallets, but there is only 0 unique ID"
    #             outputs.append({
    #                 "IMG_ID":         self.image_id,
    #                 "RACK_ID":        rack_id,
    #                 "UNIQUE_ID":      None,
    #                 "BOXNUMBER":      None,
    #                 "BOXQUANTITY":    None,
    #                 "PARTNUMBER":     None,
    #                 "INVOICE_NUMBER": None,
    #                 "EXCLUSION":      excl
    #             })
    #         return outputs

    #     grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    #     for rec in self.records:
    #         rack_id = self._rack_id_for_coord(rec['coordinates'])
    #         if rack_id:
    #             grouped[rack_id].append(rec)

    #     pallet_counts = self._count_pallets_by_side()
    #     outputs: List[Dict[str, Any]] = []
    #     for rack_id, items in grouped.items():
    #         quadrant = self.rack_dict[rack_id]
    #         side = 'left' if quadrant == 'Q3' else 'right'
    #         p = pallet_counts.get(side, 0)
    #         uids = [i["uniqueId"] for i in items]
    #         u = len(uids)

    #         excl = None if p == u else (
    #             f"There are {p} pallets, but there is only {u} unique ID" if p > u
    #             else f"There are {p} pallets, but there are extra {u - p} unique ID(s)"
    #         )

    #         outputs.append({
    #             "IMG_ID":         self.image_id,
    #             "RACK_ID":        rack_id,
    #             "UNIQUE_ID":      ",".join(uids),
    #             "BOXNUMBER":      ",".join(str(i["box_number"]) for i in items),
    #             "BOXQUANTITY":    ",".join(str(i["box_quantity"]) for i in items),
    #             "PARTNUMBER":     ",".join(i["part_number"] for i in items),
    #             "INVOICE_NUMBER": ",".join(i["invoice_number"] for i in items),
    #             "EXCLUSION":      excl
    #         })
    #     return outputs


    def process(self) -> Any:
        return self._map_single(self.records)
