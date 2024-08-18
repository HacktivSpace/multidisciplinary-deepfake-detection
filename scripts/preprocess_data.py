import os
import pandas as pd
import numpy as np
import cv2
from src.config import config
from src.utils.logger import setup_logger
from src.dataset.data_splitter import split_data
from src.dataset.data_augmentation import augment_data
from src.utils.file_utils import save_to_file

logger = setup_logger('preprocess_data_logger', os.path.join(config.LOG_DIR, 'data_preprocessing.log'))

def load_raw_data(data_dir):
    """
    Loading raw data from the data directory.
    :param data_dir: Directory where raw data is stored
    :return: DataFrame containing the raw data
    """
    logger.info("Loading raw data...")
    
    raw_data = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            raw_data.append(data)
            logger.info(f"Loaded {file_name} with shape {data.shape}")
    
    raw_data_df = pd.concat(raw_data, ignore_index=True)
    logger.info(f"Total raw data shape: {raw_data_df.shape}")
    
    return raw_data_df

def preprocess_data(data):
    """
    Preprocessing the data.
    :param data: DataFrame containing the data to preprocess
    :return: Preprocessed DataFrame
    """
    logger.info("Starting data preprocessing...")
    
    # Example preprocessing steps
    logger.info("Handling missing values...")
    data.fillna(method='ffill', inplace=True)
    
    logger.info("Normalizing data...")
    data = (data - data.min()) / (data.max() - data.min())
    
    logger.info("Data preprocessing complete.")
    return data

def preprocess_images(data, image_dir):
    """
    Preprocessing images from the data.
    :param data: DataFrame containing the image data
    :param image_dir: Directory where images are stored
    :return: Preprocessed images and labels
    """
    logger.info("Starting image preprocessing...")
    
    images = []
    labels = []
    for idx, row in data.iterrows():
        img_path = os.path.join(image_dir, row['image_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        images.append(image)
        labels.append(row['label'])
    
    images = np.array(images)
    labels = np.array(labels)
    
    logger.info(f"Preprocessed {len(images)} images.")
    
    return images, labels

def preprocess_audios(data, audio_dir):
    """
    Preprocessing audios from the data.
    :param data: DataFrame containing the audio data
    :param audio_dir: Directory where audios are stored
    :return: Preprocessed audios and labels
    """
    logger.info("Starting audio preprocessing...")
    
    audios = []
    labels = []
    for idx, row in data.iterrows():
        audio_path = os.path.join(audio_dir, row['audio_path'])
        audio = preprocess_audio(audio_path)
        audios.append(audio)
        labels.append(row['label'])
    
    audios = np.array(audios)
    labels = np.array(labels)
    
    logger.info(f"Preprocessed {len(audios)} audios.")
    
    return audios, labels

def preprocess_videos(data, video_dir):
    """
    Preprocessing videos from the data.
    :param data: DataFrame containing the video data
    :param video_dir: Directory where videos are stored
    :return: Preprocessed videos and labels
    """
    logger.info("Starting video preprocessing...")
    
    videos = []
    labels = []
    for idx, row in data.iterrows():
        video_path = os.path.join(video_dir, row['video_path'])
        video = preprocess_video(video_path)
        videos.append(video)
        labels.append(row['label'])
    
    videos = np.array(videos)
    labels = np.array(labels)
    
    logger.info(f"Preprocessed {len(videos)} videos.")
    
    return videos, labels

def save_processed_data(data, file_path):
    """
    Saving processed data to file.
    :param data: DataFrame containing the processed data
    :param file_path: Path to save the processed data
    """
    logger.info(f"Saving processed data to {file_path}...")
    save_to_file(data, file_path)
    logger.info("Processed data saved successfully.")

if __name__ == "__main__":
    logger.info("Starting data preprocessing pipeline...")
    
    raw_data = load_raw_data(config.RAW_DATA_DIR)
    processed_data = preprocess_data(raw_data)
    
    if config.PROCESS_IMAGES:
        images, labels = preprocess_images(processed_data, config.IMAGE_DIR)
        processed_data['image'] = images.tolist()
        processed_data['label'] = labels.tolist()

    if config.PROCESS_AUDIOS:
        audios, labels = preprocess_audios(processed_data, config.AUDIO_DIR)
        processed_data['audio'] = audios.tolist()
        processed_data['label'] = labels.tolist()

    if config.PROCESS_VIDEOS:
        videos, labels = preprocess_videos(processed_data, config.VIDEO_DIR)
        processed_data['video'] = videos.tolist()
        processed_data['label'] = labels.tolist()
    
    train_data, test_data = split_data(processed_data, test_size=0.2)
    augmented_train_data = augment_data(train_data)
    
    save_processed_data(augmented_train_data, config.PROCESSED_TRAIN_DATA_FILE)
    save_processed_data(test_data, config.PROCESSED_TEST_DATA_FILE)
    
    logger.info("Data preprocessing pipeline completed successfully.")
