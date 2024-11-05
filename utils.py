from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import cv2


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

    plt.savefig(os.path.join("output", csv_file_path, "statistics.png"))


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
