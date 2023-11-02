import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import shuffle
import matplotlib.pyplot as plt
import elasticdeform
import tensorflow.keras.backend as K


# Define Intersection over Union (IoU) metric
def iou_metric(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = K.mean((intersection + K.epsilon()) / (union + K.epsilon()), axis=0)
    return iou


# Define Pixel Accuracy metric
def pixel_accuracy_metric(y_true, y_pred):
    y_pred = K.round(y_pred)
    correct_pixels = K.sum(K.cast(K.equal(y_true, y_pred), dtype=tf.float32))
    total_pixels = K.sum(K.cast(K.equal(y_true, y_true), dtype=tf.float32))  # Total pixels in ground truth
    pixel_accuracy = correct_pixels / (total_pixels + K.epsilon())
    return pixel_accuracy


# Define a function to perform data augmentation with three categories
def augment_data(images, masks, rotation_range=(-15, 15)):
    augmented_images_rotated = []
    augmented_masks_rotated = []
    augmented_images_rotated_deformed = []
    augmented_masks_rotated_deformed = []
    augmented_images_deformed = []
    augmented_masks_deformed = []

    for image, mask in zip(images, masks):
        # Randomly rotate the image and mask within the specified range
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        image_rotated = rotate_image(image, angle)
        mask_rotated = rotate_image(mask, angle)

        # Apply elastic deformation if enabled

        image_deformed, mask_deformed = apply_elastic_deformation(image, mask)
        image_rotated_deformed, mask_rotated_deformed = apply_elastic_deformation(image_rotated, mask_rotated)


        # Append rotated data
        augmented_images_rotated.append(image_rotated)
        augmented_masks_rotated.append(mask_rotated)

        # Append rotated and deformed data
        augmented_images_rotated_deformed.append(image_rotated_deformed)
        augmented_masks_rotated_deformed.append(mask_rotated_deformed)

        # Append deformed data
        augmented_images_deformed.append(image_deformed)
        augmented_masks_deformed.append(mask_deformed)

    return (
        augmented_images_rotated,
        augmented_masks_rotated,
        augmented_images_rotated_deformed,
        augmented_masks_rotated_deformed,
        augmented_images_deformed,
        augmented_masks_deformed,
    )


def rotate_image(image, angle):
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image


def apply_elastic_deformation(image, mask):
    sigma = 4  # Control the degree of deformation
    alpha = 50  # Control the spatial resolution of deformation
    image_deformed, mask_deformed = elasticdeform.deform_random_grid([image, mask], sigma=sigma, points=alpha)
    return image_deformed, mask_deformed


def load_images_from_directory(directory):
    image_list = []

    # Iterate through the contents of the directory
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Check if the file is an image (you can add more extensions as needed)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
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


# Define the U-Net model architecture
def unet_model(input_size):
    inputs = keras.Input(shape=input_size)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = layers.concatenate([conv3, up5], axis=3)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = layers.concatenate([conv2, up6], axis=3)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = layers.concatenate([conv1, up7], axis=3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)

    model = keras.Model(inputs, outputs)

    return model


# Define the input size based on your images
input_size = (256, 256, 3)

# Create the U-Net model
model = unet_model(input_size)

# Compile the model with an appropriate loss function and optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming you have a list of input images and corresponding masks as NumPy arrays
# X_train and y_train should contain your training data
X_data = load_images_from_directory("images/CDW_whole_fragments")
Y_data = load_images_from_directory("images/CDW_masks")
(
    augmented_images_rotated,
    augmented_masks_rotated,
    augmented_images_rotated_deformed,
    augmented_masks_rotated_deformed,
    augmented_images_deformed,
    augmented_masks_deformed,
) = augment_data(
    X_data,
    Y_data,
    rotation_range=(-15, 15)
)

X_data.append(augmented_images_deformed)
X_data.append(augmented_images_rotated_deformed)
X_data.append(augmented_images_rotated)
Y_data.append(augmented_masks_deformed)
Y_data.append(augmented_masks_rotated_deformed)
Y_data.append(augmented_masks_rotated)

X_train, X_temp, Y_train, Y_temp = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Train the model with 100 epochs
history = model.fit(X_train, Y_train, epochs=100, batch_size=8, validation_data=[X_val, Y_val])

iou_values = history.history['iou_metric']
pixel_accuracy_values = history.history['pixel_accuracy_metric']

for epoch, iou, pixel_accuracy in zip(range(1, len(iou_values) + 1), iou_values, pixel_accuracy_values):
    print(f"Epoch {epoch}: IoU = {iou:.4f}, Pixel Accuracy = {pixel_accuracy:.4f}")

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
