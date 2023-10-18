import socket
import cv2
import pickle
import struct
# from rembg import remove
import numpy as np
import sys


HOST1 = "169.254.120.200"
PORT1 = 1035
pi1_connected = True

HOST2 = "169.254.120.100"
PORT2 = 1034
pi2_connected = True

if not pi1_connected and not pi2_connected:
    sys.exit("There must be at least one pi connected.")

# create first socket connection
if pi1_connected:
    s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('New socket connection formed.')
    s1.bind((HOST1, PORT1))
    print('Socket connection bind complete.')
    s1.listen(50)
    print('Socket is ready now and listening...')
    conn1, addr1 = s1.accept()
    print("Data transfer ready for Pi 1\n")

# create second socket connection
if pi2_connected:
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('New socket connection formed.')
    s2.bind((HOST2, PORT2))
    print('Socket connection bind complete.')
    s2.listen(50)
    print('Socket is ready now and listening...')
    conn2, addr2 = s2.accept()
    print("Data transfer ready for Pi 2")

if pi1_connected:
    data1 = b""
    payload_size1 = struct.calcsize(">L")
    print("Payload_size (pi 1): {}".format(payload_size1))

if pi2_connected:
    data2 = b""
    payload_size2 = struct.calcsize(">L")
    print("Payload_size (pi 2): {}".format(payload_size2))

cv2.namedWindow("ImageWindow", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ImageWindow", 2000, 3200)
font = cv2.FONT_HERSHEY_SIMPLEX  # Choose the font
font_scale = 1  # Font scale factor
font_color = (0, 0, 255)  # BGR color (in this case, red)
font_thickness = 2  # Thickness of the text
text_position = (50, 50)  # Position to place the text (x, y)
space_width = 20

while True:
    if pi1_connected:
        while len(data1) < payload_size1:
            # print("Recv: {}".format(len(data)))
            data1 += conn1.recv(4096)

        # print("Done Recv: {}".format(len(data)))
        packed_msg_size1 = data1[:payload_size1]
        data1 = data1[payload_size1:]
        msg_size1 = struct.unpack(">L", packed_msg_size1)[0]

        while len(data1) < msg_size1:
            data1 += conn1.recv(4096)
        frame_data1 = data1[:msg_size1]
        data1 = data1[msg_size1:]
        data_list1 = pickle.loads(frame_data1, fix_imports=True, encoding="bytes")
        frame1 = data_list1[0]
        img_np1 = np.frombuffer(frame1, dtype=np.uint8)
        decoded_image1 = cv2.imdecode(img_np1, cv2.IMREAD_COLOR)
        decoded_image1 = cv2.cvtColor(decoded_image1, cv2.COLOR_BGR2RGB)

    if pi2_connected:
        while len(data2) < payload_size2:
            # print("Recv: {}".format(len(data)))
            data2 += conn2.recv(4096)
        packed_msg_size2 = data2[:payload_size2]
        data2 = data2[payload_size2:]
        msg_size2 = struct.unpack(">L", packed_msg_size2)[0]

        while len(data2) < msg_size2:
            data2 += conn2.recv(4096)
        frame_data2 = data2[:msg_size2]
        data2 = data2[msg_size2:]
        data_list2 = pickle.loads(frame_data2, fix_imports=True, encoding="bytes")
        frame2 = data_list2[0]
        img_np2 = np.frombuffer(frame2, dtype=np.uint8)
        decoded_image2 = cv2.imdecode(img_np2, cv2.IMREAD_COLOR)
        decoded_image2 = cv2.cvtColor(decoded_image2, cv2.COLOR_BGR2RGB)
        weight = data_list2[1]
        weight_str = '%.0f' % weight
        cv2.putText(decoded_image2, weight_str, text_position, font, font_scale, font_color, font_thickness)

        # print(weight)

    if not pi2_connected:
        cv2.imshow('ImageWindow', decoded_image1)
    elif not pi1_connected:
        cv2.imshow('ImageWindow', decoded_image2)
    else:
        empty_space = np.zeros((decoded_image1.shape[0], space_width, 3), dtype=np.uint8)
        im_h = cv2.hconcat([decoded_image1, decoded_image2])
        cv2.imshow('ImageWindow', im_h)
    cv2.waitKey(1)
