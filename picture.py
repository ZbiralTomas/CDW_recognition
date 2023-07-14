import numpy as np
import cv2
from rembg import remove
from skimage.measure import label, regionprops
import math
from get_features import get_mig, get_rgb, get_entropy
import csv

# determine constants
top_scale = 1
side_scale = 1
subset_size = 100
subset_spacing = 105
weight = 10
min_subset_fraction_in_sample = 0.9
# read the image
top_view = cv2.imread('volume/vrchni.jpg')
side_view = cv2.imread('volume/bocni.jpg')
# remove background and creates mask
removed_top = remove(top_view)
removed_top = np.array(removed_top)
mask_top = np.moveaxis(removed_top, -1, 0)
mask_top = mask_top[3]
mask_top.setflags(write=1)
mask_top[mask_top < 100] = 0
mask_top[mask_top >= 100] = 1
labeled_top = label(mask_top)
# put mask over the original image
masked_top = cv2.bitwise_and(top_view, top_view, mask=mask_top)
regions_top = regionprops(labeled_top)

removed_side = remove(side_view)
removed_side = np.array(removed_side)
mask_side = np.moveaxis(removed_side, -1, 0)
mask_side = mask_side[3]
mask_side.setflags(write=1)
mask_side[mask_side < 100] = 0
mask_side[mask_side >= 100] = 1
labeled_side = label(mask_side)
# put mask over the original image
masked_side = cv2.bitwise_and(side_view, side_view, mask=mask_side)
regions_side = regionprops(labeled_side)

for props in regions_top:
    min_w_top, min_l_top, max_w_top, max_l_top = props.bbox
    # Filled area
    area = props.area
    # bbox area
    area_bbox = props.area_bbox
    top_coefficient = math.sqrt(area / area_bbox)
    length = top_coefficient * abs(min_l_top - max_l_top)
    width = top_coefficient * abs(min_w_top - max_w_top)

for props in regions_top:
    min_w_side, min_l_side, max_w_side, max_l_side = props.bbox
    # Filled area
    area = props.area
    # bbox area
    area_bbox = props.area_bbox
    side_coefficient = math.sqrt(area / area_bbox)
    height = side_coefficient * abs(min_w_side - max_w_side)

width *= top_scale
length *= top_scale
height *= side_scale

n_col_top = int((abs(max_l_top - min_l_top) - subset_size) / subset_spacing)
n_row_top = int((abs(max_w_top - min_w_top) - subset_size) / subset_spacing)

file = open('test.csv', 'w', newline='')
headerlist = ['H', 'MIG', 'r_rel', 'brightness', 'density', 'strana']
with file:
    writer = csv.DictWriter(file, fieldnames=headerlist)
    writer.writeheader()
file.close()

for i in range(n_row_top+1):
    for j in range(n_col_top+1):
        min_col_subset = int(min_l_top + j * subset_spacing)
        max_col_subset = min_col_subset + subset_size
        min_row_subset = int(min_w_top + i * subset_spacing)
        max_row_subset = min_row_subset + subset_size
        subset_mask = mask_top[min_row_subset:max_row_subset, min_col_subset:max_col_subset]
        if subset_mask.sum() > min_subset_fraction_in_sample * subset_size ** 2:
            # classify the image (subset)
            subset_image = top_view[min_row_subset:max_row_subset, min_col_subset:max_col_subset, :]
            entropy = get_entropy(subset_image)
            mig = get_mig(subset_image)
            r, g, b = get_rgb(subset_image)
            brightness = 0.299*r+0.587*g+0.114*b
            r_rel = r/(0.299*r+0.587*g+0.114*b)
            density = weight/(width*length*height)
            file = open('test.csv', 'a', newline='')
            strana = 0
            with file:
                writer = csv.DictWriter(file, fieldnames=headerlist)
                writer.writerow({'H': entropy, 'MIG': mig, 'brightness': brightness, 'r_rel': r_rel,
                                 'density': density, 'strana': strana})

n_col_side = int((abs(max_l_side - min_l_side) - subset_size) / subset_spacing)
n_row_side = int((abs(max_w_side - min_w_side) - subset_size) / subset_spacing)

for i in range(n_row_side+1):
    for j in range(n_col_side+1):
        min_col_subset = int(min_l_side + j * subset_spacing)
        max_col_subset = min_col_subset + subset_size
        min_row_subset = int(min_w_side + i * subset_spacing)
        max_row_subset = min_row_subset + subset_size
        subset_mask = mask_side[min_row_subset:max_row_subset, min_col_subset:max_col_subset]
        if subset_mask.sum() > min_subset_fraction_in_sample * subset_size ** 2:
            # classify the image (subset)
            subset_image = side_view[min_row_subset:max_row_subset, min_col_subset:max_col_subset, :]
            entropy = get_entropy(subset_image)
            mig = get_mig(subset_image)
            r, g, b = get_rgb(subset_image)
            brightness = 0.299*r+0.587*g+0.114*b
            r_rel = r/(0.299*r+0.587*g+0.114*b)
            density = weight/(width*length*height)
            strana = 1
            file = open('test.csv', 'a', newline='')
            with file:
                writer = csv.DictWriter(file, fieldnames=headerlist)
                writer.writerow({'H': entropy, 'MIG': mig, 'brightness': brightness, 'r_rel': r_rel,
                                 'density': density, 'strana': strana})

file.close()






