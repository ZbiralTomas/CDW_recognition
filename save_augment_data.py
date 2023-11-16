import cv2
import elasticdeform
import numpy as np
import os


def load_images_from_directory(directory):
    image_list = []

    for root, _, files in os.walk(directory):
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
def rotate_image(image, angle):
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image


def apply_elastic_deformation(image):
    sigma = 4  # Control the degree of deformation
    alpha = 50  # Control the spatial resolution of deformation
    image_deformed = elasticdeform.deform_random_grid(image, sigma=sigma, points=alpha)
    return image_deformed

def augment_images(path, rotation_range=(-15, 15)):

    images, file_names = load_images_from_directory(path)

    i = 0
    for image, file_name in zip(images, file_names):
        i += 1
        print(f"augmenting image number {i}")
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        image_rotated = rotate_image(image, angle)

        # Apply elastic deformation if enabled

        image_deformed = apply_elastic_deformation(image)
        image_rotated_deformed = apply_elastic_deformation(image_rotated)
        rotated_filename = "/rotated/r_"+file_name
        rotated_deformed_filename = "/rotated_deformed/rd_" + file_name
        deformed_filename = "/deformed/d_" + file_name

        cv2.imwrite(os.path.join(path + rotated_filename), image_rotated)
        cv2.imwrite(os.path.join(path + rotated_deformed_filename), image_rotated_deformed)
        cv2.imwrite(os.path.join(path + deformed_filename), image_deformed)


augment_images("images/unet_data/CDW_masks/AAC", rotation_range=(-15, 15))



