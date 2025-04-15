import cv2
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt

# Load image
img_path = '/Users/saieshagre/Desktop/Screenshot 2025-04-15 at 11.49.48.png'
image = cv2.imread(img_path)

# Decode all QR/barcodes
decoded_objects = decode(image)

# Loop through detected objects
for obj in decoded_objects:
    # Extract bounding box
    (x, y, w, h) = obj.rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Put decoded data as text
    data = obj.data.decode("utf-8")
    barcode_type = obj.type
    text = f'{barcode_type}: {data}'
    cv2.putText(image, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    print(f"Found {barcode_type}: {data}")
    print(data)

# Display result


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.title('Detected QR/Barcodes')
plt.show()
