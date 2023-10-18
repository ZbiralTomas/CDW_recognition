import socket
import cv2
import pickle
import struct
from rembg import remove
import numpy as np


# Define the position of the imaginary vertical line
vertical_line_x = 300  # Adjust this value according to your needs


def is_contact(mask, line_x):
    line_height = mask.shape[0]
    background_pixels_count = np.sum(mask[:, line_x] != 0)  # Count background pixels in the line
    return background_pixels_count >= 0.1 * line_height


font = cv2.FONT_HERSHEY_SIMPLEX
org = (500, 500)
fontScale = 8
color = (255, 0, 0)
thickness = 2
HOST = "169.254.120.100"
PORT = 1027

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('New socket connection formed.')

s.bind((HOST, PORT))
print('Socket connection bind complete.')
s.listen(10)
print('Socket is ready now and listening...')
conn, addr = s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("Payload_size: {}".format(payload_size))
while True:
    while len(data) < payload_size:
        # print("Recv: {}".format(len(data)))
        data += conn.recv(4096)

    # print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    # print("Message size: {}".format(msg_size))
    h = 0
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    data_list = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = data_list[0]
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    removed = remove(frame)
    removed = np.array(removed)
    mask = np.moveaxis(removed, -1, 0)
    mask = mask[3]
    mask.setflags(write=1)
    mask[mask < 100] = 0
    mask[mask >= 100] = 1
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    weight = data_list[1]
    weight_str = '%.0f' % weight

    if is_contact(mask, vertical_line_x):
        print("Contact")

    original_frame = cv2.line(frame, (vertical_line_x, 0), (vertical_line_x, frame.shape[0]), (0, 255, 0), 2)
    original_frame = cv2.putText(original_frame, 'weight: '+weight_str+'g', org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    print('next')
    im_v = cv2.vconcat([original_frame, masked])
    cv2.imshow('ImageWindow', im_v)
    cv2.waitKey(1)
