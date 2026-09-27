# 🚗 Automatic Number Plate Recognition(ANPR) - Vision AI By Muhammad Anas

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-green?logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white)

An advanced Computer Vision project designed to detect vehicles, track them across frames, and extract clear text from their license plates using a custom object detection pipeline and Optical Character Recognition (OCR).

## ✨ Features
1. **Vehicle Detection**: Utilizes a pre-trained **YOLOv8** model to accurately detect vehicles (cars, motorcycles, buses, and trucks) in high-resolution video streams.
2. **Multi-Object Tracking**: Integrates the **SORT (Simple Online and Realtime Tracking)** algorithm to assign persistent IDs to detected vehicles across multiple frames, enabling trajectory analysis.
3. **License Plate Detection**: Implements a custom-trained YOLOv8 model (`license_plate_detector.pt`) specifically tuned to accurately isolate license plates from moving vehicles.
4. **Optical Character Recognition (OCR)**: Leverages **EasyOCR** with a custom data validation mapping step to intelligently correct and extract alphanumeric license plate characters from skewed sub-images.

---

## 🛠️ Project Architecture Workflow
1. **`main.py`** processes the input `sample.mp4` video frame by frame.
2. The global **YOLO model** locates the bounds of all vehicles.
3. The **SORT Tracker** links these bounds contextually.
4. The region corresponding to the vehicle is cropped and passed to the **License Plate YOLO model**.
5. Discovered license plates undergo grayscale thresholding using OpenCV.
6. **EasyOCR** parses the characters, which are then cleaned by custom verification functions inside **`util.py`** (ensuring O vs 0, I vs 1 format corrections).
7. Frame and tracking data are logged line-by-line into `test.csv`.
8. Finally, running **`visualize.py`** parses the `test.csv` outputs to generate the final rendered bounding box video results.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. You can set up the required background engine locally using exactly what this project requires:
```bash
# Set up a new virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install the correct packages
pip install -r requirements.txt
```

### Running the Project

**1. Process the Video**
Run the core pipeline to identify the cars and their license plates.
```bash
python main.py
```
*Depending on hardware, processing the video sequentially frame by frame might take some time. Once complete, it produces `test.csv` containing detection mappings.*

**2. Render the Output**
Visualize your track history! Run the visualizer to render boxes mapping the license texts onto the tracked cars over a video file:
```bash
python visualize.py
```

---

## 📂 Repository Structure
- **`main.py`** : The primary orchestrator script for the computer vision pipeline.
- **`visualize.py`** : The rendering script for testing and viewing tracked components graphically.
- **`util.py`** : Contains the OCR validation dictionary mappings and bounding box coordinate manipulations.
- **`sort/`** : Contains the SORT tracking algorithm implementation.
- **`models/`** : Place your `.pt` YOLO models here (e.g., `license_plate_detector.pt` custom weights).

---

> Note: For deploying as a web application or inference endpoint, it is highly recommended to adapt the pipeline using **Hugging Face Spaces** or **Docker** to bypass Serverless function limitations.
