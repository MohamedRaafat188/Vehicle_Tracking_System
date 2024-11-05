from ultralytics import YOLO
import cv2
from time import time
from paddleocr import PaddleOCR


# Initializing the models
model_vehicles = YOLO("models/yolov8s")
model_lp = YOLO("models/best.pt")

# Initialize ocr model
ocr = PaddleOCR(lang='en', use_gpu=True)

# Input and output videos
video_path = input("Enter video path: ")
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4
out = cv2.VideoWriter('output_video2.mp4', fourcc, 30.0, (1920, 1080))

vehicles_ids = []
tracked_vehicles_ids = []

start_time = time()
frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frames += 1
    # if frames % 2:
    #     continue
    frame = cv2.resize(frame, (1920, 1080))

    # Detecting vehicles and license plates
    results_vehicles = model_vehicles.track(source=frame, conf=0.7, classes=[2, 3, 5, 7], persist=True)[0]
    results_lps = model_lp(source=frame, device="gpu")[0]

    vehicles_boxes = results_vehicles.boxes.data.int().tolist()
    lps_boxes = results_lps.boxes.data.int().tolist()

    # Looping over vehicles and license plates and plotting results on each frame
    for vehicle_box in vehicles_boxes:
        x1, y1, x2, y2 = vehicle_box[:4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    for lp_box in lps_boxes:
        x1, y1, x2, y2 = lp_box[:4]
        lp = frame[y1: y2, x1: x2]

        try:
            result, score = ocr.ocr(lp, rec=True)[0][0][1]
        except:
            pass
        else:
            if score >= 0.9:
                cv2.putText(frame, result, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    out.write(frame)

end_time = time()
print(end_time - start_time)
print(f"Number of frames {frames}")
print(f"FPS = {frames / (end_time - start_time)}")

cap.release()
out.release()