import pandas as pd
import numpy as np
import re
import os
import cv2
import math
from rembg import remove
from skimage.measure import label, regionprops


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


# Load the CSV file into a DataFrame
df = pd.read_csv('csv_files/weight_measurements/camera_1_2023_11_06_16_06_14')


# Define a function to parse the weights from the string format
def parse_weights(weight_str):
    weights = re.findall(r'\d+\.\d+', weight_str)  # Extract numeric values
    return [float(weight) for weight in weights]


# Replace 'weights_column_name' with the actual column name containing the string representation of weights
df['weights_list'] = df['Weight List'].apply(parse_weights)


# Create a function to filter out weights above 3000 grams
def filter_weights(weights_list):
    return [weight for weight in weights_list if weight <= 3000]


# Apply the filter function to each row in the DataFrame
df['filtered_weights'] = df['weights_list'].apply(filter_weights)

# Calculate the average of the filtered weights
df['average_weight'] = df['filtered_weights'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)


def average_of_closest_three(values, target):
    sorted_values = sorted(values, key=lambda x: abs(x - target))
    closest_values = sorted_values[:3]
    return np.mean(closest_values)


# Calculate the average of all the averages
# Calculate the final average by applying the function to each row
df['final_average'] = df.apply(lambda row: average_of_closest_three(row['filtered_weights'], row['average_weight']),
                               axis=1)

measured_weights = df['final_average']
measured_weights = measured_weights[:-1].round(1)

gt_weights = np.array([462.6, 385.0, 630.9, 107.3, 241.4, 327.3, 752.0, 657.7, 704.2, 360.7, 500.2, 360.0, 995.2, 402.2,
                       868.5, 1615.7, 557.8, 247.6, 123.1, 549.6, 505.8, 2143.8, 2166.8, 968.9, 1379.6, 251.4, 435.7,
                       585.1, 245.2, 606.1, 300.5, 582.6, 406.3, 2227.2, 439.6, 472.6, 135.8, 225.8])
gt_weights = pd.Series(gt_weights)
weight_difference = measured_weights-gt_weights

# print("Final Averages:")
# print(df['final_average'])
# print(weight_df)

measured_volume = [9.0, 5.0, 19.0, 6.5, 5.5, 5.0, 12.5, 10.5, 10.0, 5.3, 7.0, 7.5, 22.0, 25.0, 15.0, 26.5, 11.5, 6.5,
                   4.5, 11.0, 11.5, 40.0, 38.0, 28.5, 29.0, 9.0, 9.5, 11.5, 8.0, 9.0, 7.0, 12.0, 8.0, 48.0, 8.0, 9.0,
                   2.5, 4.0]
volume_coefficient = [500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 16,
                      500 / 16, 500 / 16, 500 / 16,
                      500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 16, 500 / 22,
                      500 / 22, 500 / 22, 500 / 22,
                      500 / 22, 500 / 22, 500 / 22, 500 / 22, 500 / 22, 500 / 22, 500 / 22, 500 / 22, 500 / 22,
                      500 / 22, 500 / 22, 500 / 22,
                      500 / 22, 500 / 22]

volume = pd.DataFrame({"measured_value": measured_volume, "volume_ratio": volume_coefficient})
gt_volume = volume["measured_value"] * volume["volume_ratio"]
# print(volume["gt_volume"])

camera1 = load_images_from_directory("images/contact_cam1/2023_11_06_16_06_14")
camera1 = camera1[:-1]
camera2 = load_images_from_directory("images/contact_cam2/2023_11_06_16_06_09")
camera2 = camera2[:-1]

calculated_volume = []
i = 1
for image1, image2 in zip(camera1, camera2):
    calculated_volume.append(calculate_volume(image1, image2, 30.85 / image1.shape[1], 27.5 / image2.shape[1], i))
    i += 1

calculated_volume = pd.Series(calculated_volume)
volume_difference = calculated_volume-gt_volume
df_to_csv = pd.concat([measured_weights, gt_weights, weight_difference, calculated_volume, gt_volume, volume_difference]
                      , axis=1)
df_to_csv = pd.DataFrame({"calculated weight[g]": measured_weights, "gt weight[g]": gt_weights,
                          "weight difference[g]": weight_difference, "calculated volume[ml]": calculated_volume,
                          "gt volume[ml]": gt_volume, "volume difference[ml]": volume_difference})

df_to_csv.to_csv("csv_files/plot_data/measurement-06_11_23.csv", index=False)

