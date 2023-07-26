import socket
import cv2
import pickle
import struct

# HOST = "169.254.73.204"
HOST = "192.168.0.101"
PORT = 1024

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
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cv2.imshow('ImageWindow', frame)
    cv2.waitKey(1)
