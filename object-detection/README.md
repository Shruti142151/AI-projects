AI Object Detection

Overview

This project is an AI-powered Object Detection system built using Python, YOLOv8, OpenCV, and Ultralytics. It detects common objects in images, draws bounding boxes around them, and labels each detected object with its class name.

Features

- Detects multiple objects in a single image
- Uses the YOLOv8 Nano model for fast and accurate detection
- Automatically saves annotated images with bounding boxes
- Easy to use and beginner-friendly

Technologies Used

- Python
- YOLOv8
- Ultralytics
- OpenCV

Project Structure

- "object_detection.py" – Main program
- "requirements.txt" – Required Python libraries
- "dataset/" – Sample input images
- "outputs/" – Detection results

How to Run

1. Install the required libraries:
   pip install -r requirements.txt
2. Place sample images in the "dataset" folder.
3. Run:
   python object_detection.py
4. The detected images will be saved in the "outputs" folder.

Future Improvements

- Real-time webcam object detection
- Video object detection
- Custom-trained object detection models
- Support for additional object classes

