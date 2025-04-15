from google.cloud import vision
import os
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/saieshagre/Downloads/NxtQube/UpdatedInventoryManagement/fluted-box-439104-p7-5e010cdb323b.json"

def extract_text_from_image(image_path, client):
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    print(f"\n--- Detected text in {os.path.basename(image_path)} ---")
    if texts:
        print(texts[0].description)
    else:
        print("No text found.")

    if response.error.message:
        raise Exception(f"API Error in {image_path}: {response.error.message}")

def process_folder(folder_path):
    client = vision.ImageAnnotatorClient()
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff")

    start = time.time()
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(folder_path, filename)
            extract_text_from_image(image_path, client)
    end = time.time()

    print(f"\nTotal time taken: {end - start:.2f} seconds")

# Example usage
process_folder("/Users/saieshagre/Downloads/OCR")
