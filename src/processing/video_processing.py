import cv2
import numpy as np
import logging

def load_video(file_path):
    """
    Loading video file.
    :param file_path: Path to the video file
    :return: Video capture object
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {file_path}")
        logging.info(f"Video file loaded: {file_path}")
        return cap
    except Exception as e:
        logging.error(f"Error loading video file {file_path}: {e}")
        raise

def extract_frames(cap, frame_rate=1):
    """
    Extracting frames from video file at a specified frame rate.
    :param cap: Video capture object
    :param frame_rate: Frame rate to extract frames (frames per second)
    :return: List of extracted frames
    """
    try:
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_rate == 0:
                frames.append(frame)
            frame_count += 1
        logging.info(f"Extracted {len(frames)} frames from video")
        return frames
    except Exception as e:
        logging.error(f"Error extracting frames: {e}")
        raise

def preprocess_frame(frame, target_size=(64, 64)):
    """
    Preprocessing single video frame by resizing and converting to grayscale.
    :param frame: Video frame
    :param target_size: Target size for resizing
    :return: Preprocessed frame
    """
    try:
        resized_frame = cv2.resize(frame, target_size)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        logging.info("Frame preprocessed")
        return gray_frame
    except Exception as e:
        logging.error(f"Error preprocessing frame: {e}")
        raise

def extract_video_features(frames):
    """
    Extracting features from video frames.
    :param frames: List of preprocessed video frames
    :return: Extracted video features
    """
    try:
        features = [frame.flatten() for frame in frames]
        video_features = np.mean(features, axis=0)
        logging.info("Video features extracted")
        return video_features
    except Exception as e:
        logging.error(f"Error extracting video features: {e}")
        raise

def process_video(file_path):
    """
    Processing video file and extract features.
    :param file_path: Path to the video file
    :return: Extracted video features
    """
    try:
        cap = load_video(file_path)
        frames = extract_frames(cap, frame_rate=10)  
        cap.release()
        
        preprocessed_frames = [preprocess_frame(frame) for frame in frames]
        video_features = extract_video_features(preprocessed_frames)
        
        logging.info(f"Extracted features from video file: {file_path}")
        
        return video_features
    except Exception as e:
        logging.error(f"Error processing video file {file_path}: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python video_processing.py <path_to_video_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    features = process_video(file_path)
    print("Extracted Features:\n", features)
