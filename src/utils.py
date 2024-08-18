import os
import json
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from src.config import Config

def save_to_file(data, filename):
    """
    Saving data to file.
    :param data: Data to save
    :param filename: Name of the file
    """
    with open(filename, 'w') as file:
        if isinstance(data, (dict, list)):
            json.dump(data, file, indent=4)
        else:
            file.write(str(data))
    logging.info(f"Data saved to {filename}")

def read_from_file(filename):
    """
    Reading data from file.
    :param filename: Name of the file
    :return: Data read from the file
    """
    with open(filename, 'r') as file:
        if filename.endswith('.json'):
            return json.load(file)
        else:
            return file.read()

def calculate_metrics(true_labels, predictions):
    """
    Calculating accuracy, precision, recall, and F1 score.
    :param true_labels: True labels
    :param predictions: Model predictions
    :return: Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
        'f1_score': f1_score(true_labels, predictions)
    }
    logging.info(f"Metrics calculated: {metrics}")
    return metrics

def plot_metrics(history, metric='accuracy'):
    """
    Plotting training and validation metrics.
    :param history: Training history
    :param metric: Metric to plot
    """
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(Config.LOG_DIR, f'{metric}_plot.png'))
    plt.close()
    logging.info(f"{metric} plot saved.")

def create_directory(path):
    """
    Creating directory if it does not exist.
    :param path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info(f"Directory created at {path}")

def load_data(file_path, file_type='csv'):
    """
    Loading data from file.
    :param file_path: Path to the file
    :param file_type: Type of the file ('csv', 'json', etc.)
    :return: Loaded data
    """
    if file_type == 'csv':
        data = pd.read_csv(file_path)
    elif file_type == 'json':
        data = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    logging.info(f"Data loaded from {file_path}")
    return data

def preprocess_data(data):
    """
    Preprocessing data.
    :param data: Data to preprocess
    :return: Preprocessed data
    """
    data = data.fillna(0)
    logging.info("Data preprocessing complete.")
    return data

def split_data(data, labels, test_size=0.2):
    """
    Splitting data into training and test sets.
    :param data: Data features
    :param labels: Data labels
    :param test_size: Proportion of test set
    :return: Split data
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=Config.RANDOM_SEED)
    logging.info(f"Data split into training and test sets with test size = {test_size}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    create_directory(Config.LOG_DIR)

    sample_data = {'name': 'Deepfake Detection', 'version': '1.0'}
    save_to_file(sample_data, os.path.join(Config.LOG_DIR, 'sample_data.json'))
    loaded_data = read_from_file(os.path.join(Config.LOG_DIR, 'sample_data.json'))
    print(loaded_data)

    true_labels = [0, 1, 1, 0, 1]
    predictions = [0, 1, 0, 0, 1]
    metrics = calculate_metrics(true_labels, predictions)
    print(metrics)
