import socket
import cv2
import pickle
import struct
import numpy as np
import os
import sys
import pygame.mixer
import csv
from datetime import datetime
import argparse

# python socket_connection_camera.py --cam 1 --sample u --ip 169.254.120.100 --port 1035 --rotation --sensitivity 5.5
# python socket_connection_camera.py --cam 1 --sample mert_asphalt --ip 169.254.120.100 --port 1035 --rotation --sensitivity 5
# python socket_connection_camera.py --cam 2 --sample Volume_testing --ip 169.254.120.200 --port 1036 --rotation --sensitivity 5.5
# python socket_connection_camera.py --cam 2 --sample mert_asphalt --ip 169.254.120.200 --port 1036 --rotation --sensitivity 5

# Create an argument parser
parser = argparse.ArgumentParser(description="Select camera number and sample name")

# Add a command-line argument flag with a default value
parser.add_argument('--cam', default='1', help='Select camera number')
parser.add_argument('--sample', default='material', help='Sample name')
parser.add_argument('--ip', default='169.254.120.100', help='Host IP')
parser.add_argument('--port', default='1035', help='Host port number')
parser.add_argument('--rotation', action='store_true', help='Rotate an image')
parser.add_argument('--sensitivity', default='8.0', help='Sensitivity to change in %')

# Parse the command-line arguments
args = parser.parse_args()

# Access the value of the input flag
camera_number = int(args.cam)
sample_name = args.sample
HOST = args.ip
PORT = int(args.port)
rotation = args.rotation
sensitivity = float(args.sensitivity)

# Your code goes here, and you can use the 'input_value' variable
print(f"Selected camera: {camera_number}")
print(f"Sample name: {sample_name}")
print(f"IP address: {HOST}")
print(f"Port number: {PORT}")
print(f"Rotation: {rotation}")


def play_mp3(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def get_mean_channels(image):
    return image.mean(axis=(0, 1))


def crop_image(image, relative_left, relative_width):
    if relative_left + relative_width > 1:
        sys.exit("The sum of relative start and relative width must be less than or equal to 1")
    width = image.shape[1]
    return image[:, int(width * relative_left):int((relative_left + relative_width) * width)]


# Line placement from a left side of an image as a decimal number
def is_contact(mean_channels_reference, image, sensitivity_to_change=8.0, relative_left=0.3, relative_width=0.02):
    cropped_image = crop_image(image, relative_left, relative_width)
    blurred_cropped_image = cv2.GaussianBlur(cropped_image, (7, 7), 0)
    mean_channels_current = get_mean_channels(blurred_cropped_image)
    channels_difference = abs(mean_channels_current - mean_channels_reference) / 2.56
    if np.max(channels_difference) > sensitivity_to_change:
        print(channels_difference, "Image taken")
        return True
    else:
        print(channels_difference)
        return False


relative_virtual_line_width = 0.05
relative_virtual_line_start = 0.19
#relative_virtual_line_start = 0.18

counter = 0

'''
Slečna Terezka - tahle část inicializuje spojení přes socket - vstupuje tam IP adresa síťového kabelu, kterou
jsem nastavil ručně pro daný kabel na trvalo (není to bezpečné) a číslo portu. 
'''

open_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('New socket connection formed.')
open_socket.bind((HOST, PORT))
open_socket.listen(50)
conn, addr = open_socket.accept()
print("Data transfer ready for Raspberry.\n")

data = b""
payload_size = struct.calcsize(">L")
print("Payload_size: {}".format(payload_size))

cv2.namedWindow("ImageWindow", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("ImageWindow", 3200, 3200)
font = cv2.FONT_HERSHEY_SIMPLEX  # Choose the font
font_scale = 1  # Font scale factor
font_color = (0, 255, 0)  # BGR color (in this case, red)
font_thickness = 2  # Thickness of the text
text_position = (50, 50)  # Position to place the text (x, y)
space_width = 20
first_image = True
first_iterations = 0
contact_detected = False
weight_list = [0, 0, 0, 0, 0]
weight_index = 0
iteration_index = 0
contact_index = 10e6
csv_dir = "csv_files"
os.makedirs(csv_dir, exist_ok=True)
today = datetime.now()
csv_filename = f"csv_files/weight_measurements/camera_{camera_number}_{today.strftime('%Y_%m_%d_%H_%M_%S')}"
if not os.path.isfile(csv_filename):
    # If the file doesn't exist, create it and write the header
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["Counter", "Timestamp", "Image Name", "Weight List"]
        writer.writerow(header)

while True:
    iteration_index += 1
    if first_iterations < 8:
        first_iterations += 1
    elif first_iterations == 8:
        cropped_image = crop_image(decoded_image, relative_left=relative_virtual_line_start,
                                   relative_width=relative_virtual_line_width)
        blurred_cropped_image = cv2.GaussianBlur(cropped_image, (7, 7), 0)
        mean_channels = get_mean_channels(blurred_cropped_image)

        cropped_image = crop_image(decoded_image, relative_virtual_line_start, relative_virtual_line_width)
        blurred_cropped_image = cv2.GaussianBlur(cropped_image, (7, 7), 0)
        mean_channels_current = get_mean_channels(blurred_cropped_image)
        channels_difference = abs(mean_channels_current - mean_channels) / 2.56
        print("channels difference - first iteration for testing: ", channels_difference)

        first_image = False
        first_iterations += 1
    else:
        pass

# Slečna Terezka - Tenhle while cyklus přijímá data - list, který je zakódovaný pomocí pickle

    while len(data) < payload_size:
        # print("Recv: {}".format(len(data)))
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)

    received_data = data[:msg_size]
    data = data[msg_size:]
    data_list = pickle.loads(received_data, fix_imports=True, encoding="bytes")

# Slečna Terezka - tady jsou data už načtená

    frame = data_list[0]
    img_numpy = np.frombuffer(frame, dtype=np.uint8)
    decoded_image = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)
    decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
    if rotation:
        decoded_image = cv2.rotate(decoded_image, cv2.ROTATE_90_CLOCKWISE)

    if not first_image:
        if is_contact(mean_channels, decoded_image, sensitivity_to_change=sensitivity,
                      relative_left=relative_virtual_line_start, relative_width=relative_virtual_line_width):
            if not contact_detected:
                counter += 1
                os.makedirs(f"images/contact_cam{camera_number}/{today.strftime('%Y_%m_%d_%H_%M_%S')}", exist_ok=True)
                img_path = f"images/contact_cam{camera_number}/{today.strftime('%Y_%m_%d_%H_%M_%S')}"
                img_name = "/img-%s-cam-%d_%04d.jpg" % (sample_name, camera_number, counter)
                cv2.imwrite(img_path + img_name, decoded_image)
                play_mp3("sounds/beep.mp3")
                contact_detected = True
                contact_index = iteration_index * 1
        else:
            contact_detected = False


    weight = data_list[1]
    weight_list[weight_index % len(weight_list)] = weight
    weight_index += 1
    if iteration_index == contact_index + int(len(weight_list) / 2):
        time_now = datetime.now()
        time_stamp = f"{time_now.strftime('%H_%M_%S')}"
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            list_to_append = [counter, time_stamp, img_name, weight_list]
            writer.writerow(list_to_append)
        print(f"Data: {list_to_append}")
        print(f"Data appended to {csv_filename}")

    weight_str = '%.0f' % weight
    image_for_plotting = decoded_image.copy()
    image_for_plotting = cv2.putText(image_for_plotting, weight_str, text_position, font, font_scale, font_color,
                                     font_thickness)
    if contact_detected:
        cv2.line(image_for_plotting, (int(decoded_image.shape[1] * relative_virtual_line_start), 0),
                 (int(decoded_image.shape[1] * relative_virtual_line_start), decoded_image.shape[0]), (0, 255, 0), 2)
        cv2.line(image_for_plotting, (int(decoded_image.shape[1] * (relative_virtual_line_width +
                                                                    relative_virtual_line_start)), 0),
                 (int(decoded_image.shape[1] * (relative_virtual_line_width +
                                                relative_virtual_line_start)),
                  decoded_image.shape[0]), (0, 255, 0), 2)
    else:
        cv2.line(image_for_plotting, (int(decoded_image.shape[1] * relative_virtual_line_start), 0),
                 (int(decoded_image.shape[1] * relative_virtual_line_start), decoded_image.shape[0]), (0, 0, 255), 2)
        cv2.line(image_for_plotting,
                 (int(decoded_image.shape[1] * (relative_virtual_line_width + relative_virtual_line_start)), 0),
                 (int(decoded_image.shape[1] * (relative_virtual_line_width + relative_virtual_line_start)),
                  decoded_image.shape[0]),
                 (0, 0, 255), 2)

    cv2.imshow('ImageWindow', image_for_plotting)
    cv2.waitKey(1)
