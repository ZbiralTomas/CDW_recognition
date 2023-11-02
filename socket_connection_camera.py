import socket
import cv2
import pickle
import struct
# from rembg import remove
import numpy as np
# import os
import sys
import pygame.mixer


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
    return image[:, int(width*relative_left):int((relative_left+relative_width)*width)]


def is_contact(mean_channels_reference, image, sensitivity_to_change=7, relative_left=0.3, relative_width=0.05):  # Line placement from a left side of an image as a decimal number
    mean_channels_current = get_mean_channels(crop_image(image, relative_left, relative_width))
    channels_difference = abs(mean_channels_current-mean_channels_reference)/2.56
    if np.max(channels_difference) > sensitivity_to_change:
        print(channels_difference, "Image taken")
        return True
    else:
        print(channels_difference)
        return False


HOST = "169.254.120.100"
PORT = 1035
relative_virtual_line_width = 0.05
relative_virtual_line_start = 0.45


camera_number = 1
sample_name = 'asphalt'
counter = 1

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
cv2.resizeWindow("ImageWindow", 2000, 3200)
font = cv2.FONT_HERSHEY_SIMPLEX  # Choose the font
font_scale = 1  # Font scale factor
font_color = (0, 255, 0)  # BGR color (in this case, red)
font_thickness = 2  # Thickness of the text
text_position = (50, 50)  # Position to place the text (x, y)
space_width = 20
first_image = True
first_iterations = 0
contact_detected = False

while True:
    if first_iterations < 8:
        first_iterations += 1
    elif first_iterations == 8:
        mean_channels = get_mean_channels(crop_image(decoded_image, relative_left=relative_virtual_line_start,
                                                     relative_width=relative_virtual_line_width))
        first_image = False
        first_iterations += 1
    else:
        pass

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
    frame = data_list[0]
    img_numpy = np.frombuffer(frame, dtype=np.uint8)
    decoded_image = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)
    decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
    if not first_image:
        if is_contact(mean_channels, decoded_image, relative_left=relative_virtual_line_start,
                      relative_width=relative_virtual_line_width):
            if not contact_detected:
                counter += 1
                print("Image saved.")
                cv2.imwrite("images/img-%s-cam-%d_%04d.jpg" % (sample_name, camera_number, counter), decoded_image)
                play_mp3("sounds/beep.mp3")
                contact_detected = True
        else:
            contact_detected = False

    weight = data_list[1]
    weight_str = '%.0f' % weight
    image_for_plotting = cv2.putText(decoded_image, weight_str, text_position, font, font_scale, font_color, font_thickness)
    if contact_detected:
        cv2.line(image_for_plotting, (int(decoded_image.shape[1]*relative_virtual_line_start), 0),
                 (int(decoded_image.shape[1]*relative_virtual_line_start), decoded_image.shape[0]), (0, 255, 0), 2)
        cv2.line(image_for_plotting, (int(decoded_image.shape[1]*(relative_virtual_line_width+relative_virtual_line_start)), 0),
                                      (int(decoded_image.shape[1]*(relative_virtual_line_width+relative_virtual_line_start)), decoded_image.shape[0]),
                                       (0, 255, 0), 2)
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

