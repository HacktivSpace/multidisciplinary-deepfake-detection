import librosa
import numpy as np
import logging

def load_audio(file_path):
    """
    Loading audio file.
    :param file_path: Path to the audio file
    :return: Audio time series and sampling rate
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        logging.info(f"Audio file loaded: {file_path}")
        return y, sr
    except Exception as e:
        logging.error(f"Error loading audio file {file_path}: {e}")
        raise

def extract_mfcc(y, sr, n_mfcc=13):
    """
    Extracting MFCC features from audio time series.
    :param y: Audio time series
    :param sr: Sampling rate of the audio
    :param n_mfcc: Number of MFCCs to return
    :return: Mean MFCC features
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        logging.info("MFCC features extracted")
        return mfcc_mean
    except Exception as e:
        logging.error(f"Error extracting MFCC features: {e}")
        raise

def extract_chroma(y, sr):
    """
    Extracting chroma features from audio time series.
    :param y: Audio time series
    :param sr: Sampling rate of the audio
    :return: Mean chroma features
    """
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        logging.info("Chroma features extracted")
        return chroma_mean
    except Exception as e:
        logging.error(f"Error extracting chroma features: {e}")
        raise

def extract_spectral_contrast(y, sr):
    """
    Extracting spectral contrast features from audio time series.
    :param y: Audio time series
    :param sr: Sampling rate of the audio
    :return: Mean spectral contrast features
    """
    try:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        logging.info("Spectral contrast features extracted")
        return spectral_contrast_mean
    except Exception as e:
        logging.error(f"Error extracting spectral contrast features: {e}")
        raise

def process_audio(file_path):
    """
    Processing an audio file and extracting features.
    :param file_path: Path to the audio file
    :return: Extracted audio features
    """
    try:
        y, sr = load_audio(file_path)
        mfcc_features = extract_mfcc(y, sr)
        chroma_features = extract_chroma(y, sr)
        spectral_contrast_features = extract_spectral_contrast(y, sr)
        
        audio_features = np.hstack([mfcc_features, chroma_features, spectral_contrast_features])
        logging.info(f"Extracted features from audio file: {file_path}")
        
        return audio_features
    except Exception as e:
        logging.error(f"Error processing audio file {file_path}: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python audio_processing.py <path_to_audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    features = process_audio(file_path)
    print("Extracted Features:\n", features)
