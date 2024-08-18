import cv2
import numpy as np
import logging
from skimage.feature import hog

def load_image(file_path):
    """
    Loading image file.
    :param file_path: Path to the image file
    :return: Loaded image
    """
    try:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Error loading image: {file_path}")
        logging.info(f"Image file loaded: {file_path}")
        return image
    except Exception as e:
        logging.error(f"Error loading image file {file_path}: {e}")
        raise

def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocessing input image by resizing and converting to grayscale.
    :param image: Loaded image
    :param target_size: Target size for resizing
    :return: Preprocessed image
    """
    try:
        resized_image = cv2.resize(image, target_size)
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        logging.info("Image preprocessed")
        return gray_image
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

def extract_hog_features(image):
    """
    Extracting Histogram of Oriented Gradients features from an image.
    :param image: Preprocessed image
    :return: HOG features
    """
    try:
        hog_features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        logging.info("HOG features extracted")
        return hog_features
    except Exception as e:
        logging.error(f"Error extracting HOG features: {e}")
        raise

def process_image(file_path):
    """
    Processing an image file and extracting features.
    :param file_path: Path to the image file
    :return: Extracted image features
    """
    try:
        image = load_image(file_path)
        preprocessed_image = preprocess_image(image)
        hog_features = extract_hog_features(preprocessed_image)
        
        logging.info(f"Extracted features from image file: {file_path}")
        
        return hog_features
    except Exception as e:
        logging.error(f"Error processing image file {file_path}: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python image_processing.py <path_to_image_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    features = process_image(file_path)
    print("Extracted Features:\n", features)
