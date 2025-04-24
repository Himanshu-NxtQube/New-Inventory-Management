from detection.qr_processing import QRProcessor
from detection.ocr_processing import OCRProcessor
from detection.rack_validator import RackValidator
from detection.correction import Corrector
from config.settings import CONFIG
from detection.utils import Utilities
import cv2

class DataExtractor:
    def __init__(self):
        self.qr_processor = QRProcessor()
        self.ocr_processor = OCRProcessor()
        self.validator = RackValidator(CONFIG['csv']['nomenclature'])
        self.corrector = Corrector()
        self.util = Utilities()

    def get_info(self, image_path, left_line_x, right_line_x, upper_line_y, lower_line_y):
        qr_rack_data, qr_box_data = self.qr_processor.extract_qr_code_data(image_path, left_line_x, right_line_x, upper_line_y, lower_line_y)

        rack_dict = {id:quad for id, quad in qr_rack_data}
        box_dict = {id:coordinate for id, coordinate in qr_box_data}

        # TODO: check if all racks and boxes are successfully extracted, if yes then remove ocr based extraction
        ocr_rack_data, ocr_box_data = self.ocr_processor.extract_ocr_info(image_path,left_line_x, right_line_x, upper_line_y, lower_line_y)

        # Initialize quadrants
        quadrant_mapping = {'Q1': "Unable to decode", 'Q2': "Unable to decode", 'Q3': "Unable to decode", 'Q4': "Unable to decode"}

        # image = cv2.imread(image_path)
        # height, width, _ = image.shape

        # center_x, center_y = width / 2, height / 2
        # quadrants = {
        #     'Q1': {'x_min': center_x, 'x_max': width, 'y_min': 0, 'y_max': center_y},
        #     'Q2': {'x_min': 0, 'x_max': center_x, 'y_min': 0, 'y_max': center_y},
        #     'Q3': {'x_min': 0, 'x_max': center_x, 'y_min': center_y, 'y_max': height},
        #     'Q4': {'x_min': center_x, 'x_max': width, 'y_min': center_y, 'y_max': height}
        # }

        # def determine_quadrant(coordinate):
        #     x, y = coordinate
        #     for quad, bounds in quadrants.items():
        #         if bounds['x_min'] <= x <= bounds['x_max'] and bounds['y_min'] <= y <= bounds['y_max']:
        #             return quad
        #     return None
        
        def validate_racks():
            try:
                return self.util.log(quadrant_mapping['Q1'], quadrant_mapping['Q2'], quadrant_mapping['Q3'], quadrant_mapping['Q4'], self.validator)
            except ValueError as e:
                #print(f"Error in position validation: {e}")
                return quadrant_mapping['Q1'], quadrant_mapping['Q2'], quadrant_mapping['Q3'], quadrant_mapping['Q4']

        for rack_id, quad in ocr_rack_data: 
            try:
                rack_id = self.corrector.fix_new_rack_numbers_character(rack_id)
            except:
                continue
            if self.validator.validate_rack_id(rack_id) and rack_id not in rack_dict.keys():
                rack_dict[rack_id] = quad
                # quad = determine_quadrant(coordinate)
                # quadrant_mapping[quad] = rack_id

        for rack_id, quad in rack_dict.items():
            quadrant_mapping[quad] = rack_id 

        quadrant_mapping['Q1'], quadrant_mapping['Q2'], quadrant_mapping['Q3'], quadrant_mapping['Q4'] = validate_racks()

        for quad, rack_id in quadrant_mapping.items():
            if rack_id == "Unable to decode":
                continue
            if rack_id not in rack_dict.keys():
                rack_dict[rack_id] = quad
    
        # TODO: integrate unique id validator 
        for uid, coordinate in ocr_box_data:
            if uid not in box_dict.keys():
                box_dict[uid] = coordinate

        return rack_dict, box_dict

        
        
        
        

        

