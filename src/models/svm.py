from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging

class SVMModel:
    @staticmethod
    def build(kernel='linear', C=1.0):
        """
        Building Support Vector Machine (SVM) model.
        :param kernel: Specifies the kernel type to be used in the algorithm
        :param C: Regularization parameter
        :return: SVM model pipeline
        """
        logger = logging.getLogger('svm_model_logger')
        logger.info(f"Building SVM model with kernel={kernel}, C={C}.")
        
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel=kernel, C=C, probability=True))
            ])
            logger.info("SVM model built successfully.")
            return pipeline
        except Exception as e:
            logger.error(f"Error building SVM model: {e}", exc_info=True)
            raise

    @staticmethod
    def save(model, model_path):
        """
        Saving SVM model to file.
        :param model: Trained SVM model
        :param model_path: Path to save the model
        """
        logger = logging.getLogger('svm_model_logger')
        logger.info(f"Saving SVM model to {model_path}.")
        
        try:
            joblib.dump(model, model_path)
            logger.info("SVM model saved successfully.")
        except Exception as e:
            logger.error(f"Error saving SVM model: {e}", exc_info=True)
            raise

    @staticmethod
    def load(model_path):
        """
        Loading SVM model from file.
        :param model_path: Path to load the model from
        :return: Loaded SVM model
        """
        logger = logging.getLogger('svm_model_logger')
        logger.info(f"Loading SVM model from {model_path}.")
        
        try:
            model = joblib.load(model_path)
            logger.info("SVM model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading SVM model: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('svm_model_logger')
    logger.info("Starting to build, save, and load SVM model for testing purposes.")
    
    model = SVMModel.build(kernel='rbf', C=1.0)
    model_path = 'svm_model_test.pkl'
    SVMModel.save(model, model_path)
    loaded_model = SVMModel.load(model_path)
    
    logger.info("SVM model build, save, and load process completed successfully.")
