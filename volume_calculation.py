import cv2
import numpy as np


left_calibration_data = np.load('calibration_params/left_camera_calibration.npz')
left_camera_matrix = left_calibration_data['camera_matrix']
left_distortion_coefficients = left_calibration_data['distortion_coefficients']

right_calibration_data = np.load('calibration_params/right_camera_calibration.npz')
right_camera_matrix = right_calibration_data['camera_matrix']
right_distortion_coefficients = right_calibration_data['distortion_coefficients']

# Load the stereo calibration parameters (rotation matrix, translation vector, etc.)
stereo_calibration_data = np.load('calibration_params/stereo_calibration_parameters.npz')
R = stereo_calibration_data['R']  # Rotation matrix
T = stereo_calibration_data['T']  # Translation vector

left_image = cv2.imread("calibration_images/left_images/image_6.jpg")
left_image = cv2.rotate(left_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
right_image = cv2.imread("calibration_images/right_images/image_6.jpg")
right_image = cv2.rotate(right_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Rectify the stereo images
width, height = left_image.shape[1], left_image.shape[0]
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    left_camera_matrix, left_distortion_coefficients,
    right_camera_matrix, right_distortion_coefficients,
    (width, height), R, T
)

map1_left, map2_left = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion_coefficients, R1, P1, (width, height), cv2.CV_16SC2
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion_coefficients, R2, P2, (width, height), cv2.CV_16SC2
)

left_rectified = cv2.remap(left_image, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_image, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

cv2.imshow('left_rectified', left_rectified)
cv2.imshow('right_rectified', right_rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()
