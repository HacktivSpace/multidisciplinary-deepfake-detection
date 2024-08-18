import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

class BayesianModel(BaseEstimator, ClassifierMixin):
    def __init__(self, prior_mean=0, prior_std=1):
        """
        Initializing Bayesian model with prior mean and standard deviation.
        :param prior_mean: Mean of the prior distribution
        :param prior_std: Standard deviation of the prior distribution
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.mean_ = None
        self.std_ = None
        self.classes_ = None
        self.logger = logging.getLogger('bayesian_model_logger')
        self.logger.info("Initialized BayesianModel with prior_mean=%s, prior_std=%s", prior_mean, prior_std)

    def fit(self, X, y):
        """
        To fit the Bayesian model to training data.
        :param X: Training data features
        :param y: Training data labels
        :return: Self
        """
        self.logger.info("Fitting Bayesian model...")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.mean_ = np.zeros((len(self.classes_), X.shape[1]))
        self.std_ = np.zeros((len(self.classes_), X.shape[1]))

        for idx, label in enumerate(self.classes_):
            X_class = X[y == label]
            self.mean_[idx, :] = X_class.mean(axis=0)
            self.std_[idx, :] = X_class.std(axis=0)

        self.logger.info("Model fitted with classes: %s", self.classes_)
        return self

    def predict_proba(self, X):
        """
        Predicting class probabilities for X.
        :param X: Input data
        :return: Class probabilities
        """
        self.logger.info("Predicting class probabilities...")
        log_prior = np.log(1.0 / len(self.classes_))
        log_likelihood = -0.5 * np.sum(((X[:, np.newaxis, :] - self.mean_) / (self.std_ + 1e-9)) ** 2, axis=2)
        log_likelihood -= np.log(self.std_ + 1e-9).sum(axis=1)
        log_posterior = log_likelihood + log_prior
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum(axis=1, keepdims=True)
        return posterior

    def predict(self, X):
        """
        Predicting class labels for X.
        :param X: Input data
        :return: Predicted class labels
        """
        self.logger.info("Predicting class labels...")
        proba = self.predict_proba(X)
        predictions = self.classes_[np.argmax(proba, axis=1)]
        self.logger.info("Predictions: %s", predictions)
        return predictions

    def predict_log_proba(self, X):
        """
        Predicting log-probabilities of the classes for input samples X.
        :param X: Input data
        :return: Log-probabilities of the classes
        """
        self.logger.info("Predicting log-probabilities...")
        log_proba = np.log(self.predict_proba(X))
        self.logger.info("Log-probabilities: %s", log_proba)
        return log_proba

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('bayesian_model_logger')
    logger.info("Testing BayesianModel...")
    
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])
    
    model = BayesianModel(prior_mean=0, prior_std=1)
    model.fit(X_train, y_train)
    
    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
    predictions = model.predict(X_test)
    logger.info("Test predictions: %s", predictions)
