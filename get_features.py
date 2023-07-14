import numpy as np
import skimage.measure
import pandas as pd
import cv2
import os
import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename
import csv
import pathlib


def get_directory():
    temp_tk_dirname = tk.Tk()
    temp_tk_dirname.withdraw()
    dir_name = askdirectory(initialdir=os.getcwd(), title='Please select a directory')
    print('Chosen directory: %s' % dir_name)
    return dir_name


def get_images(save_base_path=False):
    all_paths, filenames = [], []
    base_path = get_directory()
    all_paths.append(base_path)
    valid_images = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    for paths in all_paths:
        path_list = os.listdir(paths)
        path_list.sort()
        for f in path_list:
            filename = os.path.join(paths, f)
            if os.path.isdir(filename):
                all_paths.append(filename)
            else:
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_images:
                    continue
                filenames.append(filename)

    print('Images collected.')
    if save_base_path:
        return filenames, base_path
    else:
        return filenames


def get_subset_position(im_width, im_height, subset_size, crop_subset=False):
    if crop_subset:
        x_pos = np.random.rand() * im_width
        y_pos = np.random.rand() * im_height
        min_x = int(max([0, x_pos - np.round(subset_size / 2)]))
        max_x = int(min([x_pos + np.round(subset_size / 2), im_width - 1]))
        min_y = int(max([0, y_pos - np.round(subset_size / 2)]))
        max_y = int(min([y_pos + np.round(subset_size / 2), im_height - 1]))
    else:
        x_pos = min([max([np.round(subset_size / 2), np.random.rand() * im_width]),
                     im_width - np.round(subset_size / 2)])
        y_pos = min([max([np.round(subset_size / 2), np.random.rand() * im_height]),
                     im_height - np.round(subset_size / 2)])

        min_x = x_pos - np.round(subset_size / 2)
        max_x = x_pos + np.round(subset_size / 2)
        min_y = y_pos - np.round(subset_size / 2)
        max_y = y_pos + np.round(subset_size / 2)

    return int(min_x), int(max_x), int(min_y), int(max_y)


def get_entropy(im):  # calculate shannon entropy
    e = skimage.measure.shannon_entropy(im)
    return e


def get_mig(im):
    g_x = cv2.Sobel(im, cv2.CV_64F, 1, 0)
    g_y = cv2.Sobel(im, cv2.CV_64F, 0, 1)
    g_combined = np.sqrt((g_x ** 2) + (g_y ** 2))
    z = len(im) * len(im[0])
    mig = np.sum(g_combined) / z
    return mig


def get_rgb(im):
    r_mean = np.mean(im[:, :, 0])
    g_mean = np.mean(im[:, :, 1])
    b_mean = np.mean(im[:, :, 2])
    return r_mean, g_mean, b_mean


# Chose csv_file
def get_csv_file():
    root = tk.Tk()
    root.withdraw()
    file_path = askopenfilename(title='Select csv file', filetypes=(('CSV files', '*.csv'),))
    opened_csv = pd.read_csv(file_path)
    return opened_csv


def get_features(size=200):

    entropy_values, mig_values, r_values, g_values, b_values = [], [], [], [], []
    brightness_values, r_rel_values, materials, picture_names = [], [], [], []
    file_names, basepath = get_images(save_base_path=True)
    csv_name = os.path.basename(basepath)

    for idx, im_name in enumerate(file_names):
        material = pathlib.PurePath(im_name)
        material = material.parent.name
        picture_name = os.path.basename(im_name)
        materials.append(material)
        picture_names.append(picture_name)
        im = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        im_width, im_height = len(im[0]), len(im)
        min_x, max_x, min_y, max_y = get_subset_position(im_width, im_height, size, crop_subset=False)
        im_cropped = im[min_y:max_y, min_x: max_x]
        r, g, b = get_rgb(im_cropped)
        r_rel_values.append(r/(0.299*r+0.587*g+0.114*b))
        brightness_values.append(0.299*r+0.587*g+0.114*b)
        im_cropped = cv2.cvtColor(im_cropped, cv2.COLOR_RGB2GRAY)
        entropy = get_entropy(im_cropped)
        entropy_values.append(entropy)
        mig = get_mig(im_cropped)
        mig_values.append(mig)

    Training_list = [picture_names, r_rel_values, brightness_values, entropy_values, mig_values, materials]
    csv_variables = [csv_name, file_names]

    return Training_list, csv_variables


def create_csv():
    feature_values, csv_variables = get_features()
    csv_exist = os.path.exists('csv_files')
    if not csv_exist:
        os.makedirs('csv_files')

    ID = []
    current_id = 0
    for i in range(len(csv_variables[1])):
        current_id += 1
        ID.append(current_id)
    headerlist = ['ID', 'picture_name', 'r_rel', 'brightness', 'H', 'MIG', 'label']
    file = open('csv_files/' + csv_variables[0] + '.csv', 'w', newline='')

    with file:
        writer = csv.DictWriter(file, fieldnames=headerlist)
        writer.writeheader()

        for i in range(len(csv_variables[1])):
            writer.writerow({'ID': ID[i], 'picture_name': feature_values[0][i], 'r_rel': feature_values[1][i],
                             'brightness': feature_values[2][i], 'H': feature_values[3][i],
                             'MIG': feature_values[4][i], 'label': feature_values[5][i]})

    file.close()


if __name__ == "__main__":
    create_csv()
