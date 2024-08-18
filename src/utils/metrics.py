from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_metrics(true_labels, predictions):
    """
    Calculating performance metrics.
    :param true_labels: True labels
    :param predictions: Model predictions
    :return: Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
        'f1_score': f1_score(true_labels, predictions),
        'roc_auc': roc_auc_score(true_labels, predictions)
    }
    logging.info(f"Metrics calculated: {metrics}")
    return metrics

def plot_confusion_matrix(true_labels, predictions, labels, output_dir, filename='confusion_matrix.png'):
    """
    Plotting and saving the confusion matrix.
    :param true_labels: True labels
    :param predictions: Model predictions
    :param labels: List of labels
    :param output_dir: Directory to save the plot
    :param filename: Name of the output file
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    logging.info(f"Confusion matrix plot saved to {os.path.join(output_dir, filename)}")

def log_metrics(metrics, logger_name='metrics_logger', log_file='metrics.log'):
    """
    To log the calculated metrics to file.
    :param metrics: Dictionary with calculated metrics
    :param logger_name: Name of the logger
    :param log_file: File to log metrics
    """
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info(f"Metrics: {metrics}")
    logger.removeHandler(handler)
    handler.close()

if __name__ == "__main__":
    true_labels = [0, 1, 1, 0, 1]
    predictions = [0, 1, 0, 0, 1]
    labels = [0, 1]

    metrics = calculate_metrics(true_labels, predictions)
    print("Calculated Metrics:\n", metrics)

    plot_confusion_matrix(true_labels, predictions, labels, 'logs', 'example_confusion_matrix.png')

    log_metrics(metrics, log_file='logs/example_metrics.log')
