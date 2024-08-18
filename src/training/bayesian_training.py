import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import config
from src.dataset.data_loader import load_csv_data
from src.dataset.data_splitter import split_data
from src.models.bayesian import BayesianModel
from src.utils.helpers import create_directory
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics, log_metrics, plot_confusion_matrix

def preprocess_data(data: pd.DataFrame):
    """
    Preprocessing the input data by standardizing numerical features.
    :param data: DataFrame containing the input data
    :return: Preprocessed DataFrame
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

def train_bayesian():
    logger = setup_logger(__name__, os.path.join(config.LOG_DIR, 'bayesian_training.log'))
    logger.info("Starting Bayesian model training...")
    
    create_directory(config.MODEL_DIR)

    data = load_csv_data(config.PROCESSED_DATA_FILE)
    X = data.drop('label', axis=1)
    y = data['label']
    X = preprocess_data(X)

    # To split data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(X, y)

    model = BayesianModel(prior_mean=config.BAYESIAN_PARAMS['prior_mean'], prior_std=config.BAYESIAN_PARAMS['prior_std'])
    model.fit(X_train.values, y_train.values)

    model_path = os.path.join(config.MODEL_DIR, 'bayesian_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Bayesian model saved at {model_path}")

    y_pred = model.predict(X_val.values)
    metrics = calculate_metrics(y_val, y_pred)
    log_metrics(metrics, log_file=os.path.join(config.LOG_DIR, 'bayesian_metrics.log'))
    plot_confusion_matrix(y_val, y_pred, labels=[0, 1], output_dir=config.LOG_DIR, filename='bayesian_confusion_matrix.png')

    logger.info("Bayesian model training and evaluation completed.")

if __name__ == "__main__":
    train_bayesian()
