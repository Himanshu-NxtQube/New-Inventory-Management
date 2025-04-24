import os
import glob
import cv2
from itertools import chain
from config.settings import CONFIG
from detection.boundary_detection import BoundaryDetector
from detection.data_extraction import DataExtractor
from detection.mapping_function import RecordMapper
from detection.data_retriever import RDSDataFetcher
from collections import defaultdict
import json
import sys



boundary_detector = BoundaryDetector()
data_extractor = DataExtractor()
def process_single_image(image_path):

    # Extract bounding lines
    left_line_x, right_line_x = boundary_detector.get_blue_bar_boundaries(image_path)
    upper_line_y, lower_line_y = boundary_detector.get_orange_bar_boundaries(image_path)

    rack_dict, box_dict = data_extractor.get_info(image_path,left_line_x,right_line_x,upper_line_y,lower_line_y)

    # print("\nQR: \n")
    # print(qr_rack_data, qr_box_data)
    # print("\nOCR: \n")
    # print(ocr_rack_data, ocr_box_data)
    # print(image_path)
    # image_name = image_path.split('/')[-1]
    # image = cv2.imread(image_path)

    # print("\nBOX: \n")
    # print(box_dict)


    # print("\nRACK: \n")
    # print(rack_dict)


    racks = {"HD-123" : "Q3", "HD-124" : "Q4"}

    # 2) Run your mapper once per UNIQUE_ID and collect the outputs
    batch = []
    updated_record = []
    box_dict = {'@A1145': (1077, 1744), '@A1111': (2758, 1340)}
    #print("BOX_DICT", box_dict)

    mapper = RecordMapper(image_path, batch, racks, lower_line_y)

    for unique_id, coordinates in box_dict.items():
        if not unique_id:
            continue

        data_fetcher = RDSDataFetcher()
        record = data_fetcher.fetch_closest_match(unique_id, sys.argv[4])
        del data_fetcher


        record["coordinates"] = coordinates

        mapper = RecordMapper(image_path, record, racks, lower_line_y)

        #rec = mapper.process()
        print(json.dumps(mapper.process(), indent=2))

        del mapper

        #batch.append(updated_record)

    
    #USE THE FOLLOWING TO 
    #rec = mapper.process()
    #print(json.dumps(mapper.process(), indent=2))












def main():
    image_directory = CONFIG['input']['image_directory']
    #image_directory = './test images'
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # Loop over all supported extensions
    # for ext in image_extensions:
    #     pattern = os.path.join(image_directory, ext)
    #     print(pattern)
    #     for image_path in glob.glob(pattern):
    #         print(f"Processing image: {image_path}")
    #         print(BoundaryDetector.get_blue_bar_boundaries(image_path))

    for image_file in os.listdir(image_directory):
        if not image_file.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(image_directory,image_file)
        #print(f"Processing image: {image_path}")
        process_single_image(image_path)


if __name__ == "__main__" :
    # print("running")
    main()
    # print("completed")