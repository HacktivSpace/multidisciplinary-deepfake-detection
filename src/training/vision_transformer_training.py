import os
import torch
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
from src.config import config
from src.dataset.data_loader import load_csv_data
from src.models.vision_transformer import VisionTransformer
from src.utils.helpers import create_directory
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics, log_metrics, plot_confusion_matrix, plot_metrics

def preprocess_data(data: pd.DataFrame):
    """
    Preprocessing input data by converting it to tensors and normalizing.
    :param data: DataFrame containing the input data
    :return: Preprocessed tensors for features and labels
    """
    X = data.drop('label', axis=1).values
    y = data['label'].values
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def train_vision_transformer():
    logger = setup_logger(__name__, os.path.join(config.LOG_DIR, 'vision_transformer_training.log'))
    logger.info("Starting Vision Transformer model training...")
    
    create_directory(config.MODEL_DIR)

    data = load_csv_data(config.PROCESSED_DATA_FILE)
    X, y = preprocess_data(data)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_SEED)

    model = VisionTransformer(
        img_size=config.VISION_TRANSFORMER_PARAMS['img_size'],
        patch_size=config.VISION_TRANSFORMER_PARAMS['patch_size'],
        num_classes=2,  
        dim=config.VISION_TRANSFORMER_PARAMS['dim'],
        depth=config.VISION_TRANSFORMER_PARAMS['depth'],
        heads=config.VISION_TRANSFORMER_PARAMS['heads'],
        mlp_dim=config.VISION_TRANSFORMER_PARAMS['mlp_dim']
    )

    # To define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=config.VISION_TRANSFORMER_PARAMS['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.VISION_TRANSFORMER_PARAMS['epochs']):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch [{epoch+1}/{config.VISION_TRANSFORMER_PARAMS['epochs']}], Loss: {loss.item()}")

    model_path = os.path.join(config.MODEL_DIR, 'vision_transformer_model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Vision Transformer model saved at {model_path}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        _, y_pred = torch.max(outputs, 1)
    metrics = calculate_metrics(y_val.numpy(), y_pred.numpy())
    log_metrics(metrics, log_file=os.path.join(config.LOG_DIR, 'vision_transformer_metrics.log'))
    plot_confusion_matrix(y_val.numpy(), y_pred.numpy(), labels=[0, 1], output_dir=config.LOG_DIR, filename='vision_transformer_confusion_matrix.png')

    logger.info("Vision Transformer model training and evaluation completed.")

if __name__ == "__main__":
    train_vision_transformer()
