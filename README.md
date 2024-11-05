# Vehicle Tracking System

## Overview
This is a Vehicle Tracking System freelance project that uses computer vision to detect and track vehicles in real time. It leverages YOLOv8 model for detection and PaddleOCR for license plate recognition, along with data storage and visualization tools for tracking vehicle movement. These YOLO models will be deployed on the cloud and connected to the camera of the client using its url.

## Features
- **Vehicle Detection:** Detects and identifies vehicles in each frame.
- **License Plate Recognition:** Recognizes and extracts license plate numbers from detected vehicles.
- **Data Storage:** Stores vehicle details, including images and detection timestamps.
- **Traffic Analysis:** Generates and plots traffic distribution based on time data.
- **Reporting:** Produces daily traffic reports in Excel format with hourly vehicle counts.

## Project Structure
- `models/`: Pre-trained model for vehicle detection, custom model for license plate detection and OCR.
- `License-Plate-Recognition-2/`: Dataset to train the license plate detection model. Download the data first from the code in file.ipynb
- `output/`: Saved results, such as CSV files, Excel reports, and graphs.
- `file.ipynb`: Download the dataset and train the license plate detection model. 
- `main.py`: Detecting vehicles and store data
- `visualize.py`: Visualize the results

## Installation

### Prerequisites
- **Python 3.10**
- **ultralytics**
- **pytorch torchvision pytorch-cuda=11.8** (optional to use gpu instead of cpu)
- **paddlepaddle** or **paddlepaddle-gpu**
- **paddleocr**

### Steps
1. **Clone the Repository:**
2. **install Prerequisites**
3. **run main.py** file