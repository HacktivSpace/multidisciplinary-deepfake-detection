import os
import json
import logging
from datetime import datetime

def create_directory(path: str):
    """
    Creating directory if it does not exist.
    :param path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info(f"Directory created at {path}")

def save_to_file(data, filename: str):
    """
    Saving data to file (JSON/plain text).
    :param data: Data to save
    :param filename: Name of the file
    """
    with open(filename, 'w') as file:
        if isinstance(data, (dict, list)):
            json.dump(data, file, indent=4)
        else:
            file.write(str(data))
    logging.info(f"Data saved to {filename}")

def read_from_file(filename: str):
    """
    Reading data from file (JSON/plain text).
    :param filename: Name of the file
    :return: Data read from the file
    """
    with open(filename, 'r') as file:
        if filename.endswith('.json'):
            return json.load(file)
        else:
            return file.read()

def get_timestamp() -> str:
    """
    To get the current timestamp in specific format.
    :return: Timestamp string
    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    To setup logger.
    :param name: Name of the logger
    :param log_file: File to log messages to
    :param level: Logging level
    :return: Configured logger
    """
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def calculate_metrics(true_labels, predictions):
    """
    Calculating accuracy, precision, recall, and F1 score.
    :param true_labels: True labels
    :param predictions: Model predictions
    :return: Dictionary with metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
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
    Ploting training and validation metrics.
    :param history: Training history
    :param metric: Metric to plot
    """
    import matplotlib.pyplot as plt
    
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join('logs', f'{metric}_plot.png'))
    plt.close()
    logging.info(f"{metric} plot saved.")

if __name__ == "__main__":

    create_directory('example_dir')

    sample_dict = {'name': 'Deepfake Detection', 'version': '1.0'}
    save_to_file(sample_dict, 'example_dir/sample_data.json')
    loaded_dict = read_from_file('example_dir/sample_data.json')
    print("Loaded JSON:\n", loaded_dict)

    # To get current timestamp
    timestamp = get_timestamp()
    print("Current Timestamp:", timestamp)

    logger = setup_logger('example_logger', 'example_dir/example.log')
    logger.info("This is a test log message.")

    true_labels = [0, 1, 1, 0, 1]
    predictions = [0, 1, 0, 0, 1]
    metrics = calculate_metrics(true_labels, predictions)
    print("Metrics:\n", metrics)
    
    class DummyHistory:
        def __init__(self):
            self.history = {
                'accuracy': [0.1, 0.2, 0.3],
                'val_accuracy': [0.15, 0.25, 0.35]
            }
    plot_metrics(DummyHistory())
