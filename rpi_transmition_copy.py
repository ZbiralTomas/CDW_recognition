import cv2
import socket
import numpy as np

# Create a socket connection
receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receiver_ip = "169.254.73.204"  # Replace with the IP address of your computer
receiver_port = 1024  # Choose the same port number used on the Raspberry Pi
receiver_address = (receiver_ip, receiver_port)
receiver_socket.bind(receiver_address)
receiver_socket.listen(1)
print("Waiting for connections...")

# Accept a connection from the Raspberry Pi
connection, address = receiver_socket.accept()
print("Connected to:", address)

# Determine the expected frame size based on your captured resolution and format
# Adjust the values accordingly for your specific setup

while True:
    # Receive frame data from the Raspberry Pi
    frame_data = b''
    while True:
        data = connection.recv(4096)
        frame_data += data
        if len(data) < 4096:
            break

    # Convert the received frame data into a numpy array
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

    # Decode the numpy array into an OpenCV frame
    frame = cv2.imdecode(frame_np, 1)

    # Display the frame
    cv2.imshow("Received Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        



