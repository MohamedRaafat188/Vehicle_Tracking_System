from ultralytics import YOLO
import cv2
from threading import Thread
from paddleocr import PaddleOCR
from utils import save_cars, read_valid_license_plate


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

tracked_vehicles_ids = []
results = []
frames = 0
thread = None

video_path = input("Enter video path: ")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open camera.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frames += 1
    frame = cv2.resize(frame, (1920, 1080))

    results_vehicles = model_vehicles.track(source=frame, conf=0.5, classes=[2, 3, 5, 7], persist=True)[0]
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

                    full_lp_num = read_valid_license_plate(ocr, lp)
                    if full_lp_num:
                        tracked_vehicles_ids.append(vehicle_id)
                        results.append([car, full_lp_num])

    # Save every 10 detected vehicles together using different thread to optimize performance
    if results and len(tracked_vehicles_ids) % 10 == 0:
        temp_results = results.copy()
        thread = Thread(target=save_cars, args=(temp_results,))
        thread.start()
        results.clear()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

save_cars(results)
if thread is not None and thread.is_alive():
    thread.join()

cap.release()
cv2.destroyAllWindows()
