import os
from ultralytics import YOLO
import cv2
from time import time
from paddleocr import PaddleOCR
from utils import read_valid_license_plate


# Initializing the models
model_vehicles = YOLO("models/yolov8s.pt")
model_lp = YOLO("models/best.pt")

# Initialize ocr model
ocr = PaddleOCR(
    lang="en",
    ocr_version="PP-OCRv4",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Input and output videos
video_path = input("Enter video path: ")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit(f"Error: could not open video: {video_path}")

source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

os.makedirs("output/demo", exist_ok=True)
output_path = os.path.join("output/demo", "annotated_" + os.path.basename(video_path))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4
out = cv2.VideoWriter(output_path, fourcc, source_fps, (1920, 1080))

vehicles_ids = []
tracked_vehicles_ids = []
last_plate_by_id = {}  # vehicle track id -> last known plate text, so it keeps showing between reads

start_time = time()
frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frames += 1
    frame = cv2.resize(frame, (1920, 1080))

    # Detect vehicles first, then run plate detection on each vehicle's crop -
    # the plate model was trained on close-up car crops (see main.py), not on
    # tiny plates within a full 1920x1080 frame, so detecting on the whole
    # frame directly misses most plates.
    results_vehicles = model_vehicles.track(source=frame, conf=0.5, classes=[2, 3, 5, 7], persist=True)[0]
    vehicles_boxes = results_vehicles.boxes.data.int().tolist()

    for vehicle_box in vehicles_boxes:
        x1, y1, x2, y2, vehicle_id = vehicle_box[:5]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        car = frame[y1:y2, x1:x2]
        if car.size == 0:
            continue
        lp_result = model_lp(source=car)[0]
        lp_boxes = lp_result.boxes.data.int().tolist()
        if not lp_boxes:
            continue

        lp_x1, lp_y1, lp_x2, lp_y2 = lp_boxes[0][:4]
        lp = car[lp_y1:lp_y2, lp_x1:lp_x2]
        cv2.rectangle(frame, (x1 + lp_x1, y1 + lp_y1), (x1 + lp_x2, y1 + lp_y2), (0, 255, 0), 3)

        lp_num = read_valid_license_plate(ocr, lp)
        if lp_num:
            last_plate_by_id[vehicle_id] = lp_num

        display_num = last_plate_by_id.get(vehicle_id)
        if display_num:
            cv2.putText(frame, display_num, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    out.write(frame)

end_time = time()
print(f"Elapsed: {end_time - start_time:.1f}s")
print(f"Number of frames {frames}")
print(f"Processing FPS = {frames / (end_time - start_time):.2f}")
print(f"Saved annotated video to {output_path}")

cap.release()
out.release()