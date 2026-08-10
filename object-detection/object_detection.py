import os
import cv2
from ultralytics import YOLO

DATASET_FOLDER = "dataset"
OUTPUT_FOLDER = "outputs"
# C
reate output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully!\n")

image_files = [
    file for file in os.listdir(DATASET_FOLDER)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_files)} images.\n")

for image_name in image_files:

    image_path = os.path.join(DATASET_FOLDER, image_name)

    print(f"Processing: {image_name}")

    results = model(image_path)

    annotated_image = results[0].plot()

    output_path = os.path.join(OUTPUT_FOLDER, image_name)

    cv2.imwrite(output_path, annotated_image)

print("\n===================================")
print("Object Detection Completed!")
print(f"Processed {len(image_files)} images.")
print(f"Results saved in '{OUTPUT_FOLDER}' folder.")
print("===================================")