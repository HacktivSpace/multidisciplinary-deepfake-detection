import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import logging

class CNNModel:
    @staticmethod
    def build(input_shape, num_classes):
        """
        Building Convolutional Neural Network (CNN) model.
        :param input_shape: Shape of the input data (height, width, channels)
        :param num_classes: Number of classes for the output layer
        :return: Compiled CNN model
        """
        logger = logging.getLogger('cnn_model_logger')
        logger.info(f"Building CNN model with input shape {input_shape} and {num_classes} output classes.")
        
        try:
            model = Sequential()
            
            # Convolutional Layer 1
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            logger.info("Added first convolutional layer.")
            
            # Convolutional Layer 2
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            logger.info("Added second convolutional layer.")
            
            # Convolutional Layer 3
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            logger.info("Added third convolutional layer.")
            
            # Flattening Layer
            model.add(Flatten())
            logger.info("Added flattening layer.")
            
            # Fully Connected Layer 1
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            logger.info("Added first fully connected layer with dropout.")
            
            # Fully Connected Layer 2
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            logger.info("Added second fully connected layer with dropout.")
            
            # Output Layer
            model.add(Dense(num_classes, activation='softmax'))
            logger.info("Added output layer.")
            
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            logger.info("Compiled the CNN model.")
            
            return model
        except Exception as e:
            logger.error(f"Error building CNN model: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('cnn_model_logger')
    logger.info("Starting to build the CNN model for testing purposes.")
    
    # Usage for testing
    input_shape = (64, 64, 3)
    num_classes = 2  
    model = CNNModel.build(input_shape, num_classes)
    
    logger.info("CNN model built successfully.")
