import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.config import config
from src.dataset.data_loader import load_csv_data
from src.utils.helpers import create_directory
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics, log_metrics, plot_confusion_matrix

def preprocess_data(data: pd.DataFrame):
    """
    Preprocessing the input data by standardizing numerical features.
    :param data: DataFrame containing the input data
    :return: Preprocessed features and labels
    """
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

def train_svm():
    logger = setup_logger(__name__, os.path.join(config.LOG_DIR, 'svm_training.log'))
    logger.info("Starting SVM model training...")
    
    create_directory(config.MODEL_DIR)

    data = load_csv_data(config.PROCESSED_DATA_FILE)
    X, y = preprocess_data(data)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_SEED)

    # To create pipeline with scaler and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=config.SVM_PARAMS['kernel'], C=config.SVM_PARAMS['C'], probability=True))
    ])

    pipeline.fit(X_train, y_train)

    model_path = os.path.join(config.MODEL_DIR, 'svm_model.pkl')
    joblib.dump(pipeline, model_path)
    logger.info(f"SVM model saved at {model_path}")

    y_pred = pipeline.predict(X_val)
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    metrics = calculate_metrics(y_val, y_pred)
    log_metrics(metrics, log_file=os.path.join(config.LOG_DIR, 'svm_metrics.log'))
    plot_confusion_matrix(y_val, y_pred, labels=[0, 1], output_dir=config.LOG_DIR, filename='svm_confusion_matrix.png')

    logger.info("SVM model training and evaluation completed.")

if __name__ == "__main__":
    train_svm()
