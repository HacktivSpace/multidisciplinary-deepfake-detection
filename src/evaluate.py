import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import torch
import json

from src.config import config
from src.dataset.data_loader import load_csv_data
from src.dataset.data_preprocessor import preprocess_data
from src.utils.file_utils import save_to_file, read_from_file
from src.utils.logger import setup_logger
from src.models.svm import evaluate_svm
from src.models.bayesian import BayesianModel
from src.models.vision_transformer import VisionTransformer
from src.utils.metrics import calculate_metrics

logger = setup_logger('evaluation_logger', os.path.join(config.LOG_DIR, 'evaluation.log'))

def evaluate_cnn():
    logger.info("Evaluating CNN model...")
    try:
        model_path = os.path.join(config.MODEL_DIR, 'cnn_model.h5')
        model = load_model(model_path)

        test_data = load_csv_data(config.PROCESSED_DATA_FILE)
        X_test = preprocess_data(test_data.drop('label', axis=1))
        y_test = test_data['label']

        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        save_evaluation_results('cnn', report, cm)
        logger.info("CNN model evaluation complete.")
    except Exception as e:
        logger.error(f"Error evaluating CNN model: {e}", exc_info=True)

def evaluate_transformer():
    logger.info("Evaluating Transformer model...")
    try:
        model_path = os.path.join(config.MODEL_DIR, 'transformer_model.pth')
        model = torch.load(model_path)
        model.eval()

        test_data = load_csv_data(config.PROCESSED_DATA_FILE)
        X_test = torch.tensor(preprocess_data(test_data.drop('label', axis=1)).values, dtype=torch.float32)
        y_test = torch.tensor(test_data['label'].values, dtype=torch.float32)

        y_pred = model(X_test).detach().numpy()
        y_pred = (y_pred > 0.5).astype("int32")

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        save_evaluation_results('transformer', report, cm)
        logger.info("Transformer model evaluation complete.")
    except Exception as e:
        logger.error(f"Error evaluating Transformer model: {e}", exc_info=True)

def evaluate_svm():
    logger.info("Evaluating SVM model...")
    try:
        model_path = os.path.join(config.MODEL_DIR, 'svm_model.pkl')
        model = read_from_file(model_path)

        test_data = load_csv_data(config.PROCESSED_DATA_FILE)
        X_test = preprocess_data(test_data.drop('label', axis=1))
        y_test = test_data['label']

        accuracy = model.score(X_test, y_test)
        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        cm = confusion_matrix(y_test, model.predict(X_test))
        save_evaluation_results('svm', report, cm, accuracy)
        logger.info("SVM model evaluation complete.")
    except Exception as e:
        logger.error(f"Error evaluating SVM model: {e}", exc_info=True)

def evaluate_bayesian():
    logger.info("Evaluating Bayesian model...")
    try:
        model_path = os.path.join(config.MODEL_DIR, 'bayesian_model.pkl')
        model = read_from_file(model_path)

        test_data = load_csv_data(config.PROCESSED_DATA_FILE)
        X_test = preprocess_data(test_data.drop('label', axis=1))
        y_test = test_data['label']

        y_pred = model.predict(X_test.values)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        save_evaluation_results('bayesian', report, cm)
        logger.info("Bayesian model evaluation complete.")
    except Exception as e:
        logger.error(f"Error evaluating Bayesian model: {e}", exc_info=True)

def evaluate_vision_transformer():
    logger.info("Evaluating Vision Transformer model...")
    try:
        model_path = os.path.join(config.MODEL_DIR, 'vision_transformer_model.pth')
        model = VisionTransformer(
            img_size=config.VISION_TRANSFORMER_PARAMS['img_size'],
            patch_size=config.VISION_TRANSFORMER_PARAMS['patch_size'],
            num_classes=config.VISION_TRANSFORMER_PARAMS['num_classes'],
            dim=config.VISION_TRANSFORMER_PARAMS['dim'],
            depth=config.VISION_TRANSFORMER_PARAMS['depth'],
            heads=config.VISION_TRANSFORMER_PARAMS['heads'],
            mlp_dim=config.VISION_TRANSFORMER_PARAMS['mlp_dim']
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_data = load_csv_data(config.PROCESSED_DATA_FILE)
        X_test = torch.tensor(preprocess_data(test_data.drop('label', axis=1)).values, dtype=torch.float32)
        y_test = torch.tensor(test_data['label'].values, dtype=torch.float32)

        y_pred = model(X_test).detach().numpy()
        y_pred = (y_pred > 0.5).astype("int32")

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        save_evaluation_results('vision_transformer', report, cm)
        logger.info("Vision Transformer model evaluation complete.")
    except Exception as e:
        logger.error(f"Error evaluating Vision Transformer model: {e}", exc_info=True)

def save_evaluation_results(model_name, report, cm, accuracy=None):
    try:
        report_path = os.path.join(config.REPORT_DIR, f'{model_name}_classification_report.json')
        cm_path = os.path.join(config.REPORT_DIR, f'{model_name}_confusion_matrix.csv')

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
        cm_df.to_csv(cm_path)

        if accuracy:
            accuracy_path = os.path.join(config.REPORT_DIR, f'{model_name}_accuracy.txt')
            with open(accuracy_path, 'w') as f:
                f.write(f'Accuracy: {accuracy}')

        logger.info(f"Saved evaluation results for {model_name}.")
    except Exception as e:
        logger.error(f"Error saving evaluation results for {model_name}: {e}", exc_info=True)

if __name__ == "__main__":
    evaluate_cnn()
    evaluate_transformer()
    evaluate_svm()
    evaluate_bayesian()
    evaluate_vision_transformer()
