import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.evaluation.cnn_evaluation import evaluate_cnn
from src.evaluation.transformer_evaluation import evaluate_transformer
from src.evaluation.svm_evaluation import evaluate_svm
from src.evaluation.bayesian_evaluation import evaluate_bayesian
from src.evaluation.vision_transformer_evaluation import evaluate_vision_transformer

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        """
        Setting up test variables and environment.
        """
        
        self.num_samples = 100
        self.num_features = 224 * 224 * 3
        self.num_classes = 2
        self.batch_size = 10

        X = np.random.randn(self.num_samples, 3, 224, 224).astype(np.float32)
        y = np.random.randint(0, self.num_classes, self.num_samples)
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.cnn_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 224 * 224, self.num_classes)
        )

        self.transformer_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 224 * 224, self.num_classes)
        )

        class DummyModel:
            def predict(self, X):
                return np.random.randint(0, self.num_classes, len(X))

        self.svm_model = DummyModel()
        self.bayesian_model = DummyModel()
        self.vision_transformer_model = self.transformer_model  

    def test_evaluate_cnn(self):
        """
        Testing CNN model evaluation.
        """
        device = 'cpu'
        metrics = evaluate_cnn(self.cnn_model, self.dataloader, device)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)

    def test_evaluate_transformer(self):
        """
        Testing Transformer model evaluation.
        """
        device = 'cpu'
        metrics = evaluate_transformer(self.transformer_model, self.dataloader, device)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)

    def test_evaluate_svm(self):
        """
        Testing SVM model evaluation.
        """
        X_test = np.random.randn(self.num_samples, self.num_features)
        y_test = np.random.randint(0, self.num_classes, self.num_samples)
        metrics = evaluate_svm(self.svm_model, X_test, y_test)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)

    def test_evaluate_bayesian(self):
        """
        Testing Bayesian model evaluation.
        """
        X_test = np.random.randn(self.num_samples, self.num_features)
        y_test = np.random.randint(0, self.num_classes, self.num_samples)
        metrics = evaluate_bayesian(self.bayesian_model, X_test, y_test)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)

    def test_evaluate_vision_transformer(self):
        """
        Testing Vision Transformer model evaluation.
        """
        device = 'cpu'
        metrics = evaluate_vision_transformer(self.vision_transformer_model, self.dataloader, device)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)

if __name__ == "__main__":
    unittest.main()
