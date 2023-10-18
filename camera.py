import cv2
from matplotlib import pyplot as plt
from rembg import remove
import numpy as np
from skimage.measure import label, regionprops
import math

# Calibration
scale = 1
speed = 1
# Connect to a webcam
cap = cv2.VideoCapture(0)
# This access frame from video - ret false/true identify if we are getting anything back
while cap.isOpened():
    # Access frames
    ret, frame = cap.read()

    img_out = remove(frame)
    img_out = np.array(img_out)
    mask = np.moveaxis(img_out, -1, 0)
    # Mask separation
    mask = mask[3]
    mask.setflags(write=1)
    mask[mask < 100] = 0
    mask[mask >= 100] = 1
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    # Mask original image
    masked = cv2.bitwise_and(frame, frame, mask=mask)


    # Show video
    cv2.imshow('Webcam', masked)
    # If Q is pressed, the loop is gonna break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releases the webcam
cap.release()
# Close the video
cv2.destroyAllWindows()