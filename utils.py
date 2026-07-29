from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import re


def plot_last_day_graph(csv_file_path):
    """
    Plots and saves a bar chart of the traffic distribution for the previous day.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing vehicle traffic data.

    - Each bar represents the number of vehicles that passed in a specific hour.
    - The function labels each bar with the count for better readability.

    Returns:
    - None
    """
    weekdays = ['Monday', 'Tuesday', 'Wednesday',
                'Thursday', 'Friday', 'Saturday', 'Sunday']
    day = weekdays[datetime.now().weekday() - 1]
    df = pd.read_csv(csv_file_path)
    data = df.Hour.value_counts().to_dict()

    keys = list(data.keys())
    values = list(data.values())

    # Plotting the vertical bar plot
    plt.figure(figsize=(12, 8))  # Adjust the figure size if needed
    plt.bar(keys, values, color='skyblue')

    for i, value in enumerate(values):
        plt.text(keys[i], value + 0.1, str(value), ha='center', va='bottom')

    # Adding labels and title
    plt.xticks(keys)
    plt.xlabel('Hours')
    plt.ylabel('Number of passed Vehicles')
    plt.title(f'Traffic Distribution {day}')

    plt.savefig(os.path.join(os.path.dirname(csv_file_path), "statistics.png"))


def save_cars(cars_data):
    """
    Saves car images and license plate data to a CSV file and stores images.

    Parameters:
    - cars_data (list of tuples): A list where each tuple contains:
        - car_image (numpy array): The image of the car to be saved.
        - lp_num (str): The license plate number of the car.

    Description:
    This function creates a new directory for storing data based on the current date.
    It saves each car image and records metadata for each vehicle (image path, license plate number,
    hour, minute, and second) in a CSV file.

    Returns:
    - None
    """
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    d, t = date[:-9], date[-8:]
    dir_path = os.path.join("output", d)
    csv_file_path = os.path.join(dir_path, "data.csv")
    data_to_append = []
    try:
        os.mkdir(dir_path)
        columns = ["Car Image Path", "License Plate Number",
                   "Hour", "Minute", "Second"]
        data_to_append.append(columns)
        plot_last_day_graph(os.path.join(
            "output", os.listdir("output")[-2], "data.csv"))
    except:
        pass

    for car_image, lp_num in cars_data:
        cv2.imwrite(os.path.join(
            dir_path, f"{t + '-' + lp_num}.png"), car_image)
        data_to_append.append([f'{t}-{lp_num}.png', lp_num, *t.split('-')])

    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_to_append)


def check_license_plate_pattern(text):
    """
    Checks if the given text matches one of the two license plate patterns.

    Parameters:
    - text (str): The input text to check.

    Returns:
    - bool: True if the text matches either pattern, False otherwise.
    """
    pattern1 = r"^\d{4}[A-Z]{3}$"
    pattern2 = r"^[A-Z]{2}\d{4}[A-Z]{2}$"

    return bool(re.match(pattern1, text) or re.match(pattern2, text))


def assemble_plate_text(texts, scores, boxes, min_score=0.9):
    """
    Assembles the OCR text tokens of a single plate into one plate number.

    Parameters:
    - texts (list of str): The recognized text of each detected line.
    - scores (list of float): The recognition confidence of each line.
    - boxes (list): Bounding box [x1, y1, x2, y2] of each line, or None if unavailable.
    - min_score (float): Minimum recognition confidence for a line to be kept.

    Description:
    Car plates are printed on a single line, but motorcycle plates are stacked on two
    lines (e.g. "5545" above "GZN"), and PaddleOCR does not necessarily return the
    lines in reading order. The bounding boxes are therefore used to group the lines
    into rows top to bottom, and to order the lines inside each row left to right,
    so both plate shapes assemble into the same "5545GZN" form.

    Single character lines are skipped because they are the country code printed on
    the blue EU band at the side of the plate ("E", "D", "F", ...), never part of the
    plate number itself. If no boxes are available the tokens are simply kept in the
    order PaddleOCR returned them.

    Returns:
    - str: The assembled plate number, not yet validated against the plate patterns.
    """
    lines = []
    boxes = list(boxes) if boxes is not None else []

    for i, (txt, score) in enumerate(zip(texts, scores)):
        txt = str(txt).strip().upper()

        # Drop unreadable lines, separators, and the EU country code band
        if float(score) < min_score or not txt.isalnum() or len(txt) < 2:
            continue

        box = None
        if i < len(boxes):
            coords = np.asarray(boxes[i]).ravel().tolist()
            if len(coords) >= 4:
                box = [float(c) for c in coords[:4]]

        if box is None:
            # No geometry: fall back to the order PaddleOCR gave us
            lines.append({"text": txt, "center_y": float(len(lines)), "x": 0.0, "height": 1.0})
        else:
            x1, y1, x2, y2 = box
            lines.append({"text": txt, "center_y": (y1 + y2) / 2,
                          "x": x1, "height": max(1.0, y2 - y1)})

    if not lines:
        return ""

    # Group the lines into rows: a new row starts once the vertical gap to the
    # current row is larger than roughly half a line height.
    lines.sort(key=lambda line: line["center_y"])
    rows = [[lines[0]]]
    for line in lines[1:]:
        row_center = sum(l["center_y"] for l in rows[-1]) / len(rows[-1])
        if abs(line["center_y"] - row_center) <= 0.6 * line["height"]:
            rows[-1].append(line)
        else:
            rows.append([line])

    lp_num = ""
    for row in rows:
        row.sort(key=lambda line: line["x"])
        lp_num += "".join(line["text"] for line in row)

    return lp_num


def read_valid_license_plate(ocr, lp_image, min_score=0.9):
    """
    Reads a license plate number from a cropped image and validates it against
    the known plate patterns via check_license_plate_pattern.

    Parameters:
    - ocr (PaddleOCR): An already initialized PaddleOCR instance.
    - lp_image (numpy array): The cropped image of the license plate.
    - min_score (float): Minimum recognition confidence for a text line to be kept.

    Description:
    The detected text lines are reassembled by assemble_plate_text, which handles both
    single line car plates and two line motorcycle plates. The result is only returned
    if it matches one of the known plate patterns, which filters out partial reads and
    text picked up from around the plate.

    Returns:
    - str: The recognized plate number, or an empty string if nothing valid was read.
    """
    try:
        ocr_results = ocr.predict(lp_image)
    except Exception:
        return ""

    for res in ocr_results:
        data = res.json['res']
        lp_num = assemble_plate_text(
            data['rec_texts'],
            data['rec_scores'],
            data.get('rec_boxes'),
            min_score)

        if lp_num and check_license_plate_pattern(lp_num):
            return lp_num

    return ""
