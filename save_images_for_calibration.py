import socket
import cv2
import pickle
import struct
import argparse
import numpy as np

# python save_images_for_calibration.py --camera right  --ip 169.254.120.100 --port 1035
# python save_images_for_calibration.py --camera left  --ip 169.254.120.200 --port 1036
parser = argparse.ArgumentParser(description="Select information")
parser.add_argument('--camera', default='left', help='Camera side')
parser.add_argument('--ip', default='169.254.120.100', help='Host IP')
parser.add_argument('--port', default='1035', help='Host port number')
args = parser.parse_args()
HOST = args.ip
PORT = int(args.port)
camera = args.camera

print(f"IP address: {HOST}")
print(f"Port number: {PORT}")

open_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('New socket connection formed.')
open_socket.bind((HOST, PORT))
open_socket.listen(50)
conn, addr = open_socket.accept()
print("Data transfer ready for Raspberry.\n")

data = b""
payload_size = struct.calcsize(">L")
print("Payload_size: {}".format(payload_size))

chessboard_list = []
i = 0
while i < 15:
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

    cv2.imshow(f'Calibration_{camera}', decoded_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.imwrite(f"calibration_images/{camera}_images/image_{i}.jpg", decoded_image)
        print('Image saved.')
        i += 1
