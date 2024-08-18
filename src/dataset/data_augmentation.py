from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, IAAPiecewiseAffine
)
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
import pandas as pd
from src.config import Config

class DataAugmentation:
    def __init__(self):
        """
        Initializing DataAugmentation with augmentation techniques.
        """
        self.augmentations = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Transpose(p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            Blur(blur_limit=3, p=0.5),
            OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
            GridDistortion(p=0.5),
            IAAPiecewiseAffine(p=0.5),
            ToTensorV2()
        ])

    def augment(self, image):
        """
        Applying augmentations to image.
        :param image: Image to augment
        :return: Augmented image
        """
        augmented = self.augmentations(image=image)
        return augmented['image']

def apply_augmentation(image_path):
    """
    Applying augmentation to image given its path.
    :param image_path: Path to the image file
    :return: Augmented image tensor
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmentation = DataAugmentation()
    augmented_image = augmentation.augment(image)
    return augmented_image

def augment_data(data):
    """
    Applying augmentation to all images in the dataset.
    :param data: DataFrame containing image paths and labels
    :return: DataFrame with augmented images and labels
    """
    augmented_images = []
    labels = []

    for index, row in data.iterrows():
        image_path = os.path.join(Config.RAW_DATA_DIR, 'images', row['filename'])
        augmented_image = apply_augmentation(image_path)
        augmented_images.append(augmented_image)
        labels.append(row['label'])

    augmented_data = pd.DataFrame({'image': augmented_images, 'label': labels})
    return augmented_data

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # usage
    image_path = os.path.join(Config.RAW_DATA_DIR, 'images', 'sample_image.jpg')
    augmented_image = apply_augmentation(image_path)

    # To display original and augmented images
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(augmented_image.permute(1, 2, 0).numpy())  
    ax[1].set_title("Augmented Image")
    ax[1].axis('off')

    plt.show()
