from rembg import remove
import numpy as np
import os
from PIL import Image
import cv2


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

def create_masks(path, save_path):
    images, file_names = load_images_from_directory(path)
    i = 0

    for image, file_name in zip(images, file_names):
        print(i)
        image_removed = remove(image)
        print("removed")
        result_array = np.array(image_removed)
        alpha_channel = result_array[:, :, 3]
        mask = (alpha_channel > 0).astype(np.uint8) * 255
        mask_filename = "/"+"mask_"+file_name
        # mask_filename = f"/mask_{i}.jpg"
        i += 1
        mask_image = Image.fromarray(mask)
        image_np = np.array(mask_image)
        cv2.imwrite(os.path.join(save_path+mask_filename), image_np)

create_masks("images/CDW_whole_fragments/Asphalt", "images/CDW_masks/Asphalt")




