from ultralytics import YOLO
import cv2
from threading import Thread
from paddleocr import PaddleOCR
from utils import save_cars


# Initializing the models
model_vehicles = YOLO("models/yolov8s")
model_lp = YOLO("models/best.pt")

# Initialize ocr model
ocr = PaddleOCR(lang='en', use_gpu=True)

tracked_vehicles_ids = []
results = []
frames = 0

video_path = input("Enter video path: ")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open camera.")

while True:
    ret, frame = cap.read()

    if not ret:
        save_cars(results)
        break
    
    frames += 1
    if frames % 2:
        continue

    frame = cv2.resize(frame, (1920, 1080))

    results_vehicles = model_vehicles.track(source=frame, conf=0.7, classes=[2, 3, 5, 7], persist=True)[0]
    boxes_data = results_vehicles.boxes.data.int().tolist()

    # loop over new tracked vehicles
    if boxes_data is not None:
        for vehicle_box in boxes_data:
            x1, y1, x2, y2, vehicle_id = vehicle_box[:5]

            if y2 < 1050 and vehicle_id not in tracked_vehicles_ids:
                car = frame[y1: y2, x1: x2]
                # Detect license plates
                lp_result = model_lp(source=car)[0]
                lp_box = lp_result.boxes.data.int().tolist()
                if len(lp_box):
                    lp_x1, lp_y1, lp_x2, lp_y2 = lp_box[0][:4]
                    lp = car[lp_y1: lp_y2, lp_x1: lp_x2]

                    try:
                        result, score = ocr.ocr(lp, rec=True)[0][0][1]
                    except:
                        pass
                    else:
                        if score >= 0.9:
                            tracked_vehicles_ids.append(vehicle_id)
                            results.append([car, result])

    # Save every 10 detected vehicles together using different thread to optimize performance
    if len(tracked_vehicles_ids) % 10 == 0:
        save_cars(results)
        temp_results = results.copy()
        thread = Thread(target=save_cars, args=(temp_results,))
        thread.start()
        results.clear()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if thread.is_alive():
    thread.join()

cap.release()
cv2.destroyAllWindows()
