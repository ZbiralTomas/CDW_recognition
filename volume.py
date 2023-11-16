import numpy as np
import cv2
from rembg import remove
from skimage.measure import label, regionprops
import math
import os

true_volume = [281.250000, 156.250000, 593.750000, 203.125000, 171.875000, 156.250000, 390.625000, 328.125000,
               312.500000, 165.625000, 218.750000, 234.375000, 687.500000, 781.250000, 468.750000, 828.125000,
               359.375000, 203.125000, 140.625000, 343.750000, 261.363636, 909.090909, 863.636364, 647.727273,
               659.090909, 204.545455, 215.909091, 261.363636, 181.818182, 204.545455, 159.090909, 272.727273,
               181.818182, 1090.909091, 181.818182, 204.545455, 56.818182, 90.909091]


def load_images_from_directory(directory):
    image_list = []
    file_paths = []

    # Iterate through the contents of the directory
    for root, _, files in os.walk(directory):
        i = 0
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))

    file_paths.sort()
    for file_path in file_paths:
        # Check if the file is an image (you can add more extensions as needed)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                # Attempt to read the image using cv2
                img = cv2.imread(file_path)

                if img is not None:
                    image_list.append(img)
                else:
                    print(f"Error loading image {file_path}")
            except Exception as e:
                print(f"Error loading image {file_path}: {str(e)}")

    return image_list


def draw_bbox(image, bbox):
    minr, minc, maxr, maxc = bbox
    image_with_bbox = cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 0, 255), 2)

    return image_with_bbox


def calculate_volume(cam1, cam2, scale1, scale2, counter):
    no_background_cam1 = remove(cam1)
    no_background_cam2 = remove(cam2)
    mask_cam1 = np.moveaxis(no_background_cam1, -1, 0)
    mask_cam11 = mask_cam1[3]
    mask_cam1 = mask_cam11.copy()
    mask_cam1.setflags(write=1)
    mask_cam1[mask_cam1 < 100] = 0
    mask_cam1[mask_cam1 >= 100] = 1
    labeled_cam1 = label(mask_cam1)
    mask_cam2 = np.moveaxis(no_background_cam2, -1, 0)
    mask_cam22 = mask_cam2[3]
    mask_cam2 = mask_cam22.copy()
    mask_cam2.setflags(write=1)
    mask_cam2[mask_cam2 < 100] = 0
    mask_cam2[mask_cam2 >= 100] = 1
    labeled_cam2 = label(mask_cam2)
    masked_cam1 = cv2.bitwise_and(cam1, cam1, mask=mask_cam1)
    regions_cam1 = regionprops(labeled_cam1)
    masked_cam2 = cv2.bitwise_and(cam2, cam2, mask=mask_cam2)
    regions_cam2 = regionprops(labeled_cam2)
#   cv2.imshow("no background images", cv2.vconcat([masked_cam1, masked_cam2]))

    area_bbox1 = 0
    for region in regions_cam1:
        if region.area_bbox > area_bbox1:
            min_h_cam1, min_w_cam1, max_h_cam1, max_w_cam1 = region.bbox
            bbox1 = region.bbox
            # Filled area
            area_cam1 = region.area
            # bbox area
            area_bbox1 = region.area_bbox
            cam1_coefficient = math.sqrt(area_cam1 / area_bbox1)
            length = abs(min_w_cam1 - max_w_cam1)
            width = abs(min_h_cam1 - max_h_cam1)

    area_bbox2 = 0
    for region in regions_cam2:
        if region.area_bbox > area_bbox2:
            min_h_cam2, min_w_cam2, max_h_cam2, max_w_cam2 = region.bbox
            bbox2 = region.bbox
            # Filled area
            area_cam2 = region.area
            # bbox area
            area_bbox2 = region.area_bbox
            cam2_coefficient = math.sqrt(area_cam2 / area_bbox2)
            height = abs(min_h_cam2 - max_h_cam2)

    cam1_bbox = draw_bbox(cam1, bbox1)
    cam2_bbox = draw_bbox(cam2, bbox2)
    cv2.imwrite(f"images/bounding_boxes/img_{counter}.jpg",
                cv2.vconcat([cam1_bbox, cam2_bbox]))
    volume = height * length * width * cam1_coefficient * cam2_coefficient * scale1 * scale1 * scale2

    return volume


camera1 = load_images_from_directory("images/contact_cam1/2023_11_06_16_06_14")
camera1 = camera1[:-1]
camera2 = load_images_from_directory("images/contact_cam2/2023_11_06_16_06_09")
camera2 = camera2[:-1]

calculated_volume = []
i = 1
for image1, image2 in zip(camera1, camera2):
    print(i)
    print(true_volume[i-1])
    calculated_volume.append(calculate_volume(image1, image2, 30.85 / image1.shape[1], 27.5 / image2.shape[1], i))
    print(calculated_volume[i-1])
    i += 1

