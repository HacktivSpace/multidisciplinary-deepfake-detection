from sklearn.model_selection import train_test_split
import pandas as pd
import logging

class DataSplitter:
    def __init__(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Initializing DataSplitter.
        :param test_size: Proportion of the dataset to include in the test split
        :param val_size: Proportion of the dataset to include in the validation split
        :param random_state: Seed used by the random number generator
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.logger = logging.getLogger('data_splitter_logger')

    def split(self, data, target_column):
        """
        Splitting data into training, validation, and testing sets.
        :param data: DataFrame containing the data to split
        :param target_column: Name of the target column
        :return: Tuple containing the training, validation, and testing sets (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info(f"Splitting data with target column '{target_column}'")
        
        try:
            # To separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # First split to get test set
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            self.logger.info(f"Initial split: {X_train_val.shape[0]} train/val samples, {X_test.shape[0]} test samples")

            # To calculate proportion of remaining data to allocate to validation
            val_size_adjusted = self.val_size / (1 - self.test_size)

            # Second split to get validation set from remaining training data
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=self.random_state)
            self.logger.info(f"Second split: {X_train.shape[0]} train samples, {X_val.shape[0]} validation samples")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            self.logger.error(f"Error during data splitting: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('data_splitter_logger')

    # usage
    data_path = 'path/to/processed_data.csv'  
    data = pd.read_csv(data_path)
    target_column = 'label' 
    splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(data, target_column)

    logger.info(f"Training set size: {X_train.shape[0]} samples")
    logger.info(f"Validation set size: {X_val.shape[0]} samples")
    logger.info(f"Test set size: {X_test.shape[0]} samples")
