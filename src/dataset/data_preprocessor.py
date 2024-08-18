import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# To set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('logs/data_preprocessing.log')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)

class DataPreprocessor:
    def __init__(self):
        """
        Initializing DataPreprocessor with standard scaler, label encoder, and imputer.
        """
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        logger.info("DataPreprocessor initialized with StandardScaler, LabelEncoder, and SimpleImputer.")

    def preprocess(self, data, target_column):
        """
        Preprocessing the data by filling missing values, scaling numerical features, and encoding categorical features.
        :param data: DataFrame containing the data to preprocess
        :param target_column: Name of the target column
        :return: DataFrame containing the preprocessed data, Series containing the preprocessed target
        """
        logger.info("Starting preprocessing.")
        try:
            # To separate features and target
            features = data.drop(columns=[target_column])
            target = data[target_column]
            logger.debug(f"Features and target separated. Features shape: {features.shape}, Target shape: {target.shape}")

            # To fill missing values
            features = pd.DataFrame(self.imputer.fit_transform(features), columns=features.columns)
            logger.debug("Missing values filled.")

            numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
            features[numerical_features] = self.scaler.fit_transform(features[numerical_features])
            logger.debug("Numerical features scaled.")

            # To encode categorical features
            categorical_features = features.select_dtypes(include(['object'])).columns
            for col in categorical_features:
                features[col] = self.label_encoder.fit_transform(features[col])
                logger.debug(f"Categorical feature '{col}' encoded.")

            if target.dtype == 'object':
                target = self.label_encoder.fit_transform(target)
                logger.debug("Target encoded.")

            logger.info("Preprocessing completed successfully.")
            return features, target

        except Exception as e:
            logger.error(f"Error occurred during preprocessing: {e}")
            raise

    def transform(self, data):
        """
        Transforming new data using the already fitted scaler, imputer, and label encoder.
        :param data: DataFrame containing the data to transform
        :return: DataFrame containing the transformed data
        """
        logger.info("Starting data transformation.")
        try:

            data = pd.DataFrame(self.imputer.transform(data), columns=data.columns)
            logger.debug("Missing values filled in new data.")

            numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
            data[numerical_features] = self.scaler.transform(data[numerical_features])
            logger.debug("Numerical features scaled in new data.")

            categorical_features = data.select_dtypes(include(['object'])).columns
            for col in categorical_features:
                data[col] = self.label_encoder.transform(data[col])
                logger.debug(f"Categorical feature '{col}' encoded in new data.")

            logger.info("Data transformation completed successfully.")
            return data

        except Exception as e:
            logger.error(f"Error occurred during data transformation: {e}")
            raise
