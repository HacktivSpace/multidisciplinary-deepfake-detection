import os
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
import torch
import joblib

from src.config import Config
from src.dataset.data_loader import load_csv_data
from src.dataset.data_splitter import split_data
from src.models.cnn import CNNModel
from src.models.transformer import TransformerModel
from src.models.svm import SVMModel
from src.models.bayesian import BayesianModel
from src.models.vision_transformer import VisionTransformer
from src.utils.logger import setup_logger

logger = setup_logger(__name__, os.path.join(Config.LOG_DIR, 'model_training.log'))

def train_cnn():
    logger.info("Training CNN model...")
    try:
        data = load_csv_data(Config.PROCESSED_DATA_FILE)
        X = data.drop('label', axis=1)
        y = data['label']

        X_train, X_val, y_train, y_val = split_data(X, y)

        model = CNNModel.build(input_shape=(64, 64, 3), num_classes=len(y.unique()))
        optimizer = Adam(learning_rate=Config.CNN_PARAMS['learning_rate'])

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=Config.CNN_PARAMS['epochs'], batch_size=Config.CNN_PARAMS['batch_size'], validation_data=(X_val, y_val))

        model_path = os.path.join(Config.MODEL_DIR, 'cnn_model.h5')
        model.save(model_path)
        logger.info(f"CNN model saved at {model_path}")
    except Exception as e:
        logger.error(f"Error during CNN model training: {e}", exc_info=True)
        raise

def train_transformer():
    logger.info("Training Transformer model...")
    try:
        data = load_csv_data(Config.PROCESSED_DATA_FILE)
        X = torch.tensor(data.drop('label', axis=1).values, dtype=torch.float32)
        y = torch.tensor(data['label'].values, dtype=torch.float32)

        model = TransformerModel(
            input_dim=Config.TRANSFORMER_PARAMS['input_dim'],
            model_dim=Config.TRANSFORMER_PARAMS['model_dim'],
            num_heads=Config.TRANSFORMER_PARAMS['num_heads'],
            num_layers=Config.TRANSFORMER_PARAMS['num_layers'],
            output_dim=Config.TRANSFORMER_PARAMS['output_dim']
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=Config.TRANSFORMER_PARAMS['learning_rate'])
        criterion = torch.nn.BCELoss()

        for epoch in range(Config.TRANSFORMER_PARAMS['epochs']):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch [{epoch+1}/{Config.TRANSFORMER_PARAMS['epochs']}], Loss: {loss.item()}")

        model_path = os.path.join(Config.MODEL_DIR, 'transformer_model.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Transformer model saved at {model_path}")
    except Exception as e:
        logger.error(f"Error during Transformer model training: {e}", exc_info=True)
        raise

def train_svm_model():
    logger.info("Training SVM model...")
    try:
        data = load_csv_data(Config.PROCESSED_DATA_FILE)
        X = data.drop('label', axis=1)
        y = data['label']

        model = SVMModel.build(kernel='linear', C=1.0)
        model.fit(X, y)
        
        model_path = os.path.join(Config.MODEL_DIR, 'svm_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"SVM model saved at {model_path}")
    except Exception as e:
        logger.error(f"Error during SVM model training: {e}", exc_info=True)
        raise

def train_bayesian():
    logger.info("Training Bayesian model...")
    try:
        data = load_csv_data(Config.PROCESSED_DATA_FILE)
        X = data.drop('label', axis=1)
        y = data['label']

        model = BayesianModel(prior_mean=Config.BAYESIAN_PARAMS['prior_mean'], prior_std=Config.BAYESIAN_PARAMS['prior_std'])
        model.fit(X.values, y.values)
        
        model_path = os.path.join(Config.MODEL_DIR, 'bayesian_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Bayesian model saved at {model_path}")
    except Exception as e:
        logger.error(f"Error during Bayesian model training: {e}", exc_info=True)
        raise

def train_vision_transformer():
    logger.info("Training Vision Transformer model...")
    try:
        data = load_csv_data(Config.PROCESSED_DATA_FILE)
        X = torch.tensor(data.drop('label', axis=1).values, dtype=torch.float32)
        y = torch.tensor(data['label'].values, dtype=torch.float32)

        model = VisionTransformer(
            img_size=Config.VISION_TRANSFORMER_PARAMS['img_size'],
            patch_size=Config.VISION_TRANSFORMER_PARAMS['patch_size'],
            num_classes=Config.VISION_TRANSFORMER_PARAMS['num_classes'],
            dim=Config.VISION_TRANSFORMER_PARAMS['dim'],
            depth=Config.VISION_TRANSFORMER_PARAMS['depth'],
            heads=Config.VISION_TRANSFORMER_PARAMS['heads'],
            mlp_dim=Config.VISION_TRANSFORMER_PARAMS['mlp_dim']
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=Config.VISION_TRANSFORMER_PARAMS['learning_rate'])
        criterion = torch.nn.BCELoss()

        for epoch in range(Config.VISION_TRANSFORMER_PARAMS['epochs']):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch [{epoch+1}/{Config.VISION_TRANSFORMER_PARAMS['epochs']}], Loss: {loss.item()}")

        model_path = os.path.join(Config.MODEL_DIR, 'vision_transformer_model.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Vision Transformer model saved at {model_path}")
    except Exception as e:
        logger.error(f"Error during Vision Transformer model training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    train_cnn()
    train_transformer()
    train_svm_model()
    train_bayesian()
    train_vision_transformer()
