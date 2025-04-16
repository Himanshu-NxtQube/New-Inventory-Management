import os
import glob
from itertools import chain
from config.settings import CONFIG
from detection.boundary_detection import BoundaryDetector
from detection.qr_processing import QRProcessor

def process_single_image(image_path):
    boundary_detector = BoundaryDetector()
    qr_processor = QRProcessor()

    # Extract bounding lines
    left_line_x, right_line_x = boundary_detector.get_blue_bar_boundaries(image_path)
    upper_line_y, lower_line_y = boundary_detector.get_orange_bar_boundaries(image_path)

    rack_data, box_data = qr_processor.extract_qr_code_data(image_path, left_line_x, right_line_x, upper_line_y, lower_line_y)

    print(rack_data, box_data)

def main():
    image_directory = CONFIG['input']['image_directory']
    image_extensions = ["png", "jpg", "jpeg", "bmp", "gif", "tiff", "JPG"]

    # Loop over all supported extensions
    # for ext in image_extensions:
    #     pattern = os.path.join(image_directory, ext)
    #     print(pattern)
    #     for image_path in glob.glob(pattern):
    #         print(f"Processing image: {image_path}")
    #         print(BoundaryDetector.get_blue_bar_boundaries(image_path))

    for image_file in os.listdir(image_directory):
        if image_file.split('.')[1].lower() not in image_extensions:
            continue

        image_path = os.path.join(image_directory,image_file)
        print(f"Processing image: {image_path}")
        process_single_image(image_path)


if __name__ == "__main__" :
    print("running")
    main()
    print("completed")