# License Plate Recognition

A complete license plate recognition (LPR) pipeline that combines a custom-trained YOLOv5 detector with post-processing and optical character recognition (OCR) to extract plate text from real-world images. The repository contains both the inference script used to localize and read license plates as well as the training utilities and datasets required to retrain the detector.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Inference Pipeline](#inference-pipeline)
  - [Pre-processing](#pre-processing)
  - [Running Detection and OCR](#running-detection-and-ocr)
  - [Outputs](#outputs)
- [Training the YOLOv5 Detector](#training-the-yolov5-detector)
- [Customization Tips](#customization-tips)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

## Project Overview
This project tackles automatic license plate recognition by chaining together three stages:
1. **Detection** – A YOLOv5 model trained on the [Large License Plate Detection Dataset](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset/data) to localize plates in an image.
2. **Pre-processing** – Histogram equalization and adaptive thresholding to normalize plates before OCR.
3. **Recognition** – [Fast Plate OCR](https://ankandrew.github.io/fast-plate-ocr/latest/) (an ONNX-based model) to transcribe the characters.

The pipeline was originally developed and validated in a WSL Ubuntu environment, but it can be run on any Linux, macOS, or Windows setup that meets the dependency requirements.

## Key Features
- 🧠 **Custom YOLOv5 weights** stored in `outputs/train/license_detection2/weights/best.pt` for license plate detection.
- 🧼 **Image pre-processing** (CLAHE histogram equalization + adaptive thresholding) to boost OCR robustness in challenging lighting conditions.
- 🔤 **ONNX OCR backend** leveraging Fast Plate OCR for multilingual plate transcription.
- 📦 **Self-contained dataset** under `preprocessed_dataset/` in YOLO format for training and quick experimentation.
- 🛠️ **Training script** (`train_yolov.py`) wrapping the official Ultralytics training entry point for convenient fine-tuning.

## Repository Structure
```
.
├── LPR6.py                     # Main inference script (detection + OCR + annotation)
├── train_yolov.py              # Helper script to retrain the YOLOv5 detector
├── license_plate.yaml          # Dataset definition used by YOLOv5 during training
├── preprocessed_dataset/
│   ├── images/                 # YOLOv5 image splits (train/val/test)
│   └── labels/                 # Matching YOLO-format label files
├── outputs/
│   └── train/                  # Training runs and exported YOLO checkpoints
├── requirements.txt            # Python dependencies for inference and training
└── yolov5/                     # Ultralytics YOLOv5 source code (vendor directory)
```

## Getting Started
### Prerequisites
- **Python** 3.12.7 (any 3.12.x release should work).
- **Pip** and, optionally, a virtual environment tool such as `venv` or `conda`.
- **GPU (optional but recommended):** CUDA 12.4-compatible drivers are listed in `requirements.txt`. The project also runs on CPU, but inference and training will be slower.
- **System packages:** Ensure OpenCV and ONNXRuntime can find their shared library dependencies. On Ubuntu/Debian you may need packages such as `libgl1`, `libglib2.0-0`, and `ffmpeg`.

### Installation
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd license_plate_recognition
   ```
2. **(Optional) create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
3. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Download YOLOv5 weights (if needed)** – the repository ships with a trained detector at `outputs/train/license_detection2/weights/best.pt`. Replace this file with your own weights if desired.

## Data Preparation
The repository includes a preprocessed copy of the Large License Plate Detection Dataset already split into `train`, `val`, and `test` folders inside `preprocessed_dataset/images`. Bounding boxes follow the standard YOLO format in the corresponding `preprocessed_dataset/labels` directory.

To use your own dataset:
1. Convert images and annotations to the YOLOv5 format.
2. Update `license_plate.yaml` with absolute paths to your `train` and `val` image folders.
3. Place any evaluation images under `preprocessed_dataset/images/test` (or edit `LPR6.py` to point elsewhere).

## Inference Pipeline
### Pre-processing
`LPR6.py` applies two steps before OCR:
- **CLAHE histogram equalization** (`equalize_histogram`) to boost contrast.
- **Adaptive thresholding** (`adaptive_threshold_plate`) to highlight characters for Fast Plate OCR.

### Running Detection and OCR
Run the inference script on all images located in `preprocessed_dataset/images/test`:
```bash
python LPR6.py
```
Key configuration points inside the script:
- `model = torch.hub.load(...)` expects the YOLOv5 checkpoint at `outputs/train/license_detection2/weights/best.pt`. Adjust the path to use different weights.
- `test_images_path = "preprocessed_dataset/images/test"` controls the batch of images to process.

### Outputs
- Annotated images are written to the `validation_results/` directory (created automatically).
- OCR predictions are printed to stdout and drawn above the detected bounding boxes.

## Training the YOLOv5 Detector
To retrain or fine-tune the detector on your dataset:
1. Confirm the paths in `license_plate.yaml` point to your training/validation images.
2. (Optional) adjust hyperparameters inside `train_yolov.py` such as image size, batch size, epochs, or experiment name.
3. Launch training:
   ```bash
   python train_yolov.py
   ```
4. YOLOv5 results, logs, and weights will be saved under `outputs/train/<run-name>/`. Update `LPR6.py` to load the newly generated `best.pt` or `last.pt` file.

## Customization Tips
- **Confidence threshold:** Modify `conf_threshold` in `detect_license_plate_boundaries` to balance recall vs. precision for plate detection.
- **OCR handling:** `LPR6.py` currently keeps the first prediction returned by Fast Plate OCR. Extend this logic to handle multiple hypotheses or confidence scores if needed.
- **Batch inference:** Wrap the detection/OCR logic into a function and call it from other applications or APIs to integrate LPR into a larger system.
- **Video streams:** Adapt the script to read frames from `cv2.VideoCapture` for real-time recognition.

## Troubleshooting
- **`FileNotFoundError` for weights or images:** Verify the paths in `LPR6.py`, `license_plate.yaml`, and your dataset structure.
- **ONNXRuntime or OpenCV missing libraries:** Install `libgl1`, `libglib2.0-0`, and other multimedia dependencies appropriate for your OS.
- **Slow inference:** Ensure you are using a GPU build of PyTorch and ONNXRuntime; otherwise, decrease image resolution or use the smaller `yolov5n` architecture.
- **Incorrect OCR output:** Try tweaking the pre-processing functions or replacing them with alternative filters such as morphological opening/closing.

## Acknowledgements
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the detection backbone and training framework.
- [Fast Plate OCR](https://ankandrew.github.io/fast-plate-ocr/latest/) by Andrés for the OCR component.
- [Large License Plate Detection Dataset](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset/data) for providing high-quality labeled data.
