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

frame_bytes = b''  # Initialize an empty byte string

# Determine the expected frame size based on your captured resolution and format
# Adjust the values accordingly for your specific setup
expected_frame_size = 640 * 640 * 3  # Assuming RGB format and 8-bit per channel

while len(frame_bytes) < expected_frame_size:
    data = connection.recv(4096)
    if not data:
        break
    frame_bytes += data

# Check if the received frame size exceeds the expected size
    if len(frame_bytes) > expected_frame_size:
        print("Received frame size exceeded the expected size.")
        break

    # Check if the received frame size matches the expected size
    if len(frame_bytes) == expected_frame_size:
        # Convert the frame bytes to a NumPy array
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)

        # Reshape the frame array to the original shape
        frame = frame.reshape((480, 640, 3))  # Adjust the shape as per the captured resolution

        # Display the image frame
        cv2.imshow('Received Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

connection.close()
receiver_socket.close()
# Create a loop to continuously receive and display frames
'''
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
        
        '''





