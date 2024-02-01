import numpy as np
import cv2
from matplotlib import pyplot as plt


imgL = cv2.imread('images/contact_cam1/2023_12_11_13_10_38/img-Volume_testing-cam-1_0004.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('images/contact_cam2/2023_12_11_13_10_34/img-Volume_testing-cam-2_0004.jpg', cv2.IMREAD_GRAYSCALE)
left_calibration_data = np.load('calibration_params/left_camera_calibration.npz')
left_camera_matrix = left_calibration_data['camera_matrix']
left_distortion_coefficients = left_calibration_data['distortion_coefficients']

right_calibration_data = np.load('calibration_params/right_camera_calibration.npz')
right_camera_matrix = right_calibration_data['camera_matrix']
right_distortion_coefficients = right_calibration_data['distortion_coefficients']
imgL = cv2.rotate(imgL, cv2.ROTATE_90_COUNTERCLOCKWISE)
imgR = cv2.rotate(imgR, cv2.ROTATE_90_COUNTERCLOCKWISE)
imgL = cv2.undistort(imgL, left_camera_matrix, left_distortion_coefficients)
imgR = cv2.undistort(imgR, right_camera_matrix, right_distortion_coefficients)
cv2.imshow("right image", imgR)
cv2.imshow("left image", imgL)
cv2.waitKey(0)
#stereo = cv2.StereoBM_create(numDisparities=160, blockSize=5)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,          # Minimum disparity (usually 0)
    numDisparities=16*16,     # Number of disparities to consider
    blockSize=27,             # Size of the window used for matching (odd number)
    P1=8 * 3 * 7 ** 2,       # Penalty parameter for the disparity smoothness term
    P2=32 * 3 * 7 ** 2,      # Penalty parameter for the disparity smoothness term
    disp12MaxDiff=20,         # Maximum allowed difference in the left and right disparity values
    preFilterCap=63,         # Truncation value for the prefiltered image pixels
    uniquenessRatio=4,      # Margin in percentage by which the minimum computed cost function value should "win"
    speckleWindowSize=100,   # Maximum size of smooth disparity regions to consider for speckle filtering
    speckleRange=64,         # Maximum disparity variation within a connected component
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 3-way semi-global block matching mode
)

disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()