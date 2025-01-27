import cv2
import torch
import numpy as np
import os

# Loads yolov model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='outputs/train/license_detection2/weights/best.pt')

# Some preprocessing functions
def adaptive_threshold_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def equalize_histogram(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return equalized

def preprocess_plate(plate):
    equalized = equalize_histogram(plate)
    thresh = adaptive_threshold_plate(plate)
    return thresh

# Uses yolov to detect plate boundaries
def detect_license_plate_boundaries(image_path, conf_threshold=0.4):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    results = model(img)
    detections = results.xyxy[0].cpu().numpy()
    detected_boxes = []

    for detection in detections:
        x_min, y_min, x_max, y_max, conf, cls = detection
        if conf >= conf_threshold:
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            detected_boxes.append((x_min, y_min, x_max, y_max, conf))

    return detected_boxes

# Pretrained license plate recognition model initialisation
from fast_plate_ocr import ONNXPlateRecognizer
ocr_model = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model')

# Test image directory
test_images_path = "preprocessed_dataset/images/test"
output_dir = "validation_results"
os.makedirs(output_dir, exist_ok=True)

# This programs works for multiple images,
for image_file in os.listdir(test_images_path):
    image_path = os.path.join(test_images_path, image_file)

    # Detects license plates using yolov
    detected_boxes = detect_license_plate_boundaries(image_path)
    img = cv2.imread(image_path)

    for idx, box in enumerate(detected_boxes):
        x_min, y_min, x_max, y_max, conf = box
        cropped_plate = img[y_min:y_max, x_min:x_max]

        if cropped_plate is not None:
            # Converts to grayscale / preprocesses for OCR
            gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
            
            # Runs OCR
            ocr_result = ocr_model.run(gray_plate)
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                ocr_result = ocr_result[0]
            else:
                ocr_result = "[Unrecognized]"

            print(f"OCR Result: {ocr_result}")

            # Draws bounding box and writes the OCR result onto the image
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, ocr_result, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)


    # Save the image in a local output directory
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, img)
    print(f"Saved annotated image to {output_path}")
