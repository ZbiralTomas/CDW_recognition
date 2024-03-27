from rembg import remove
import numpy as np
import os
from PIL import Image
import cv2
import csv
import matplotlib.pyplot as plt
import pickle


def count_underscores_after_second_slash(input_string: str) -> int:
    # Split the string by "/"
    parts = input_string.split('/')

    # Ensure there's a part after the second "/"
    if len(parts) > 2:
        # Count underscores in the part immediately after the second "/"
        return parts[2].count('_')
    else:
        return 0


def load_images_from_directory(directory, underscore_count):
    image_list = []

    for root, _, files in os.walk(directory):
        files.sort(key=lambda file: int(file.split('_')[underscore_count].split('.')[0]))
        for file_name in files:

            file_path = os.path.join(root, file_name)

            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                try:
                    img = cv2.imread(file_path)

                    if img is not None:
                        image_list.append(img)
                    else:
                        print(f"Error loading image {file_path}")
                except Exception as e:
                    print(f"Error loading image {file_path}: {str(e)}")

    return image_list, files

def load_depths_from_directory(directory, underscore_count):
    depth_list = []

    for root, _, files in os.walk(directory):
        files.sort(key=lambda file: int(file.split('_')[underscore_count].split('.')[0]))
        for file_name in files:

            file_path = os.path.join(root, file_name)

            if file_name.lower().endswith(('.npy')):
                try:
                    depth = np.load(file_path)

                    if depth is not None:
                        depth_list.append(depth)
                    else:
                        print(f"Error loading image {file_path}")
                except Exception as e:
                    print(f"Error loading image {file_path}: {str(e)}")

    return depth_list, files

def create_masks(path, depth_path, underscore_count):
    images, file_names = load_images_from_directory(path, underscore_count)
    i = 0
    mask_list = []
    depth_mat, depth_files = load_depths_from_directory(depth_path, underscore_count)

    for image, file_name in zip(images, file_names):
        # print(i)
        image_removed = remove(image)
        # print("removed")
        result_array = np.array(image_removed)
        alpha_channel = result_array[:, :, 3]
        mask = (alpha_channel > 0).astype(np.uint8) * 255
        mask_filename = "/"+"mask_"+file_name
        # mask_filename = f"/mask_{i}.jpg"
        i += 1
        mask_image = Image.fromarray(mask)
        image_np = np.array(mask_image)
        mask_list.append(image_np)

    return images, mask_list, depth_mat


def crop_image(image, crop_percentage):
    """Crops the given percentage from all sides of the image."""
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * crop_percentage / 100), int(w * crop_percentage / 100)
    return image[crop_h:h-crop_h, crop_w:w-crop_w]


RGB_images = "measurements/2024-03-12/asphalt_cylinders_17cm_above_belt_00/RGB images"
depth_matrices = "measurements/2024-03-12/asphalt_cylinders_17cm_above_belt_00/Depth matrices"
underscores = count_underscores_after_second_slash(RGB_images)
img, masks, depths = create_masks(RGB_images, depth_matrices, underscores)

volume_list, mean_heights_list, gt_list = [], [], []
# px_size_function implementation
coefficients = [1.11027573e-03, -1.29131737e-01,  5.71983949e+00, -1.20102507e+02, 1.19473500e+03]
polynomial_function = np.poly1d(coefficients)

''' This section is used for another data
gt_csv_file = 'csv_files/plot_data/measurement-06_20_23.csv'

with open(gt_csv_file, mode='r', newline='') as infile:
    # Create csv reader and writer objects
    reader = csv.DictReader(infile)
    # Write the header to the output file

    # Iterate through each row of the input file
    for row in reader:
        # Extract and process the gt volume[ml] value
        gt_volume_l = float(row['gt volume[ml]']) / 1000
        # Write the processed value to the output file
        gt_list.append(gt_volume_l)
'''

# calculate volumes, mean_heights and create surfplots for ROI
for i in range(len(img)):

    # depth at the conveyor belt
    mean_depth_zero_mask = np.mean(depths[i][(masks[i] == 0) & (depths[i] > 0)])
    # masked array of depths in ROI
    masked_depths = np.ma.masked_where((masks[i] != 255) | (depths[i] <= 0) | (depths[i] >= mean_depth_zero_mask),
                                       depths[i])
    heights_mm = mean_depth_zero_mask - masked_depths
    mean_height = np.mean(heights_mm)
    mean_heights_list.append(mean_height)
    # implementation of px size function fit
    px_size_mm = 70/polynomial_function((mean_depth_zero_mask-heights_mm)/10)
    px_size_diameter = 70/polynomial_function((mean_depth_zero_mask-mean_height)/10)
    # Calculation of diameter
    non_masked_indices = np.ma.nonzero(heights_mm)
    min_x = np.min(non_masked_indices[1])
    max_x = np.max(non_masked_indices[1])
    diameter_px = max_x-min_x
    diameter_mm = diameter_px*px_size_diameter
    # calculation of volume
    volumes = heights_mm*px_size_mm*px_size_mm
    volume_l = np.sum(volumes)/1000000
    volume_list.append(volume_l)

    # Creating figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("RGB Image")
    axs[0, 0].axis('off')

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(heights_mm.shape[1]), np.arange(heights_mm.shape[0]))

    # Since we have masked values, we can use them to mask the X, Y coordinates as well
    X = np.ma.masked_array(X, mask=heights_mm.mask)
    Y = np.ma.masked_array(Y, mask=heights_mm.mask)

    # Plotting the surface plot
    surf = ax.plot_surface(X, Y, heights_mm, cmap='jet')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height (mm)')
    ax.set_zlim(0, round(mean_depth_zero_mask))
    ax.set_ylim(0, 1256)
    ax.set_xlim(0, 1256)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Height (mm)')

    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    # Plot mask at position (1, 0)
    axs[1, 0].imshow(masks[i], cmap='gray')
    axs[1, 0].set_title("Mask")
    axs[1, 0].axis('off')

    # Calculate the 20% range for the mean height
    margin = 0.2 * mean_height

    # Lower and upper bounds for filtering out outliers
    lower_bound = mean_height - margin
    upper_bound = mean_height + margin

    # Filter the heights_mm array to exclude outliers (in order to make the plot more clear)
    filtered_heights = heights_mm[(heights_mm > lower_bound) & (heights_mm < upper_bound)]
    # Set the number of bins here!
    counts, bin_edges = np.histogram(filtered_heights.compressed(), bins=70)

    # Calculate the histogram of the filtered heights_mm array
    # Filter by number of elements in a bin
    valid_bins_mask = counts >= 300
    valid_counts = counts[valid_bins_mask]
    valid_bin_edges = bin_edges[:-1][valid_bins_mask]

    # Plot each valid bin as a bar
    for j in range(len(valid_counts)):
        axs[1, 1].bar(valid_bin_edges[j], valid_counts[j], width=np.diff(bin_edges)[0], color='skyblue', alpha=0.7, edgecolor='black', linewidth=1)

    # Plot the mean height as a vertical line
    axs[1, 1].axvline(mean_height, color='r', linestyle='solid', linewidth=3)

    # Adjust x-axis limits if necessary
    axs[1, 1].set_xlim(valid_bin_edges[0]-0.1, valid_bin_edges[-1]+0.1)
    axs[1, 1].set_xlabel('Height (mm)')
    axs[1, 1].set_ylabel('Frequency')
    fig.suptitle(r'$\bar{h}$ = ' + f'{mean_height:.2f}mm; '+r'$d$ = ' + f'{diameter_mm:.2f}mm')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    plt.show()
    # Saving fig as a pickle so we can load it as an interactive graphics
    with open(f'figures/surfplots/asphalt_cylinders_17/asphalt_cylinders_17cm_above_belt_{i+1}.pickle', 'wb') as f:
        pickle.dump(fig, f)
    # saving figure as pdf
    fig.savefig(f'figures/surfplots/asphalt_cylinders_17/asphalt_cylinders_17cm_above_belt_{i+1}.pdf')

# save volume data into csv
csv_file_path = 'csv_files/asphalt_cylinders_17cm.csv'

with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Check if the file is empty to decide whether to write the header
    file.seek(0)  # Go to the beginning of the file
    if file.tell() == 0:  # If file is empty (pointer is at 0), write header
        writer.writerow(['Volume [l]'])

    # Write the volume data
    for volume in volume_list:
        writer.writerow([volume])  


