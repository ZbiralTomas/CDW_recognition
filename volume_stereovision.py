import cv2
import numpy as np
import os


# Function to perform camera calibration
def calibrate_camera(image_dir):
    # Load calibration images from the directory
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]

    # Define the dimensions of the checkerboard (number of internal corners)
    checkerboard_size = (8, 6)  # Change this based on your checkerboard

    # Create arrays to store object points and image points from all images
    object_points = []  # 3D points in real-world coordinates
    image_points = []  # 2D points in image coordinates

    # Generate object points based on the checkerboard size
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    i = 2
    # Loop through your calibration images
    for image_path in image_paths:
        i += 1
        if image_path == "calibration_images/left_images/.DS_Store":
            continue
        elif image_path == "calibration_images/right_images/.DS_Store":
            continue
        else:

            # Load the image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            shape = gray.shape[::-1]

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            image_with_corners = cv2.drawChessboardCorners(image.copy(), checkerboard_size, corners, ret)
            if image_dir == "calibration_images/left_images":
                cv2.imwrite(f"calibration_images/left_corners/image_{i}.jpg", image_with_corners)
            elif image_dir == "calibration_images/right_images":
                cv2.imwrite(f"calibration_images/right_corners/image_{i}.jpg", image_with_corners)


            if ret:
                object_points.append(objp)
                image_points.append(corners)

    # Perform camera calibration
    ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )

    j = 2
    for image_path in image_paths:
        if image_path == "calibration_images/left_images/.DS_Store":
            continue
        elif image_path == "calibration_images/right_images/.DS_Store":
            continue
        else:
            image = cv2.imread(image_path)
            j += 1
            undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients)
            if image_dir == "calibration_images/left_images":
                cv2.imwrite(f"calibration_images/left_undistorted/image_{j}.jpg", undistorted_image)
            elif image_dir == "calibration_images/right_images":
                cv2.imwrite(f"calibration_images/right_undistorted/image_{j}.jpg", undistorted_image)


    return camera_matrix, distortion_coefficients, object_points, image_points, shape


# Paths to the "left_images" and "right_images" directories
left_images_dir = 'calibration_images/left_images'
right_images_dir = 'calibration_images/right_images'

# Perform camera calibration for the left and right cameras
left_camera_matrix, left_distortion_coefficients, left_object_points, left_image_points, shape = calibrate_camera(
    left_images_dir)
right_camera_matrix, right_distortion_coefficients, right_object_points, right_image_points, shape = calibrate_camera(
    right_images_dir)


# Stereo calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retval, camera_matrix1, distortion_coefficients1, camera_matrix2, distortion_coefficients2, R, T, E, F = cv2.stereoCalibrate(
    left_object_points, left_image_points, right_image_points,
    left_camera_matrix, left_distortion_coefficients,
    right_camera_matrix, right_distortion_coefficients,
    shape, criteria=criteria
)

# Save the stereo calibration parameters
stereo_calibration_data = {
    'R': R,
    'T': T,
    'E': E,
    'F': F,
}
np.savez('calibration_params/stereo_calibration_parameters.npz', **stereo_calibration_data)

# Save individual camera calibration parameters
np.savez('calibration_params/left_camera_calibration.npz', camera_matrix=left_camera_matrix,
         distortion_coefficients=left_distortion_coefficients)
np.savez('calibration_params/right_camera_calibration.npz', camera_matrix=right_camera_matrix,
         distortion_coefficients=right_distortion_coefficients)

print("Camera calibration and stereo calibration completed and saved.")
