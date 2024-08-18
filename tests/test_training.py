import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.training.cnn_training import train_cnn
from src.training.transformer_training import train_transformer
from src.training.svm_training import train_svm
from src.training.bayesian_training import train_bayesian
from src.training.vision_transformer_training import train_vision_transformer

class TestTraining(unittest.TestCase):
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
        self.device = 'cpu'

    def test_train_cnn(self):
        """
        Testing CNN model training.
        """
        model, optimizer, criterion = train_cnn(self.dataloader, self.device, num_epochs=1)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        self.assertIsInstance(criterion, torch.nn.Module)

    def test_train_transformer(self):
        """
        Testing Transformer model training.
        """
        model, optimizer, criterion = train_transformer(self.dataloader, self.device, num_epochs=1)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        self.assertIsInstance(criterion, torch.nn.Module)

    def test_train_svm(self):
        """
        Testing SVM model training.
        """
        model = train_svm(self.dataloader, num_epochs=1)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(callable(getattr(model, 'predict', None)))

    def test_train_bayesian(self):
        """
        Testing Bayesian model training.
        """
        model = train_bayesian(self.dataloader, num_epochs=1)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(callable(getattr(model, 'predict', None)))

    def test_train_vision_transformer(self):
        """
        Testing Vision Transformer model training.
        """
        model, optimizer, criterion = train_vision_transformer(self.dataloader, self.device, num_epochs=1)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        self.assertIsInstance(criterion, torch.nn.Module)

if __name__ == "__main__":
    unittest.main()
