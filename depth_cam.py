import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime


def count_subdirectories(directory):
    """Returns the number of subdirectories within the given directory."""
    # List all entries in the directory
    entries = os.listdir(directory)
    # Count and return the number of entries that are directories
    return sum(os.path.isdir(os.path.join(directory, entry)) for entry in entries)


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)


try:
    now = datetime.now()
    dir_name = f"Measurements/{now.year}_year/{now.strftime('%B')}/day_{now.day:02d}"
    measurement_idx = count_subdirectories(dir_name)
    depth_dir = os.path.join(dir_name, f"measurement_{measurement_idx}/depth_data")
    color_dir = os.path.join(dir_name, f"measurement_{measurement_idx}/color_image")
    colormap_dir = os.path.join(dir_name, f"measurement_{measurement_idx}/depth_colormap")

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Show images
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Colormap', depth_colormap)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            # Create directories based on current date and time
            now = datetime.now()

            for d in [depth_dir, color_dir, colormap_dir]:
                os.makedirs(d, exist_ok=True)

            # File paths with timestamp
            # timestamp = now.strftime("%Y%m%d_%H%M%S")
            timestamp = now.strftime("%H%M%S")
            depth_file_path = os.path.join(depth_dir, f"image_{timestamp}.npy")
            color_file_path = os.path.join(color_dir, f"{timestamp}.jpg")
            colormap_file_path = os.path.join(colormap_dir, f"{timestamp}.jpg")

            # Save files
            np.save(depth_file_path, depth_image)
            cv2.imwrite(color_file_path, color_image)
            cv2.imwrite(colormap_file_path, depth_colormap)

            print(f"Saved: {depth_file_path}, {color_file_path}, {colormap_file_path}")

        elif key & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
