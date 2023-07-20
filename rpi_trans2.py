import socket
import numpy as np
import cv2

HOST = '169.254.73.204'
PORT = 1024

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

conn, addr = s.accept()

try:
    frame_bytes = b''  # Initialize an empty byte string
    expected_frame_size = 480 * 640 * 3  # Assuming RGB format and 8-bit per channel

    while True:
        data = conn.recv(4096)
        if not data:
            break
        frame_bytes += data

        # Process the received frame if it matches the expected size
        while len(frame_bytes) >= expected_frame_size:
            # Extract a single frame from the received bytes
            frame_data = frame_bytes[:expected_frame_size]

            # Convert the frame bytes to a NumPy array
            frame = np.frombuffer(frame_data, dtype=np.uint8)

            # Reshape the frame array to the original shape
            frame = frame.reshape((480, 640, 3))  # Adjust the shape as per the captured resolution

            # Display the image frame
            cv2.imshow('Received Image', frame)
            cv2.imwrite('a.jpg', frame)
            if cv2.waitKey(1) == 27:  # Exit when Esc key is pressed
                break

            # Remove the processed frame from the received bytes
            frame_bytes = frame_bytes[expected_frame_size:]

finally:
    conn.close()
    s.close()
    cv2.destroyAllWindows()
