import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import config
from src.dataset.data_loader import load_csv_data
from src.utils.helpers import create_directory
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics, log_metrics, plot_confusion_matrix, plot_metrics

def preprocess_data(data: pd.DataFrame):
    """
    Preprocessing input data by standardizing numerical features and reshaping.
    :param data: DataFrame containing the input data
    :return: Preprocessed DataFrame
    """
    scaler = StandardScaler()
    X = data.drop('label', axis=1)
    X_scaled = scaler.fit_transform(X)
    y = data['label']
    return X_scaled, y

def reshape_data(X, img_width, img_height):
    """
    Reshaping data into format required by the CNN.
    :param X: Input data
    :param img_width: Width of the image
    :param img_height: Height of the image
    :return: Reshaped data
    """
    return X.reshape(-1, img_width, img_height, 1)

def create_cnn_model(input_shape):
    """
    Creating CNN model.
    :param input_shape: Shape of the input data
    :return: CNN model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_cnn():
    logger = setup_logger(__name__, os.path.join(config.LOG_DIR, 'cnn_training.log'))
    logger.info("Starting CNN model training...")
    
    create_directory(config.MODEL_DIR)

    data = load_csv_data(config.PROCESSED_DATA_FILE)
    X, y = preprocess_data(data)
    X = reshape_data(X, img_width=64, img_height=64)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_SEED)

    model = create_cnn_model(input_shape=(64, 64, 1))
    optimizer = Adam(learning_rate=config.CNN_PARAMS['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # To setup callbacks
    checkpoint = ModelCheckpoint(os.path.join(config.MODEL_DIR, 'cnn_model.h5'), monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=config.CNN_PARAMS['epochs'], batch_size=config.CNN_PARAMS['batch_size'], validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])
    
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    metrics = calculate_metrics(y_val, y_pred)
    log_metrics(metrics, log_file=os.path.join(config.LOG_DIR, 'cnn_metrics.log'))
    plot_confusion_matrix(y_val, y_pred, labels=[0, 1], output_dir=config.LOG_DIR, filename='cnn_confusion_matrix.png')
    plot_metrics(history, metric='accuracy')
    plot_metrics(history, metric='loss')

    logger.info("CNN model training and evaluation completed.")

if __name__ == "__main__":
    train_cnn()
