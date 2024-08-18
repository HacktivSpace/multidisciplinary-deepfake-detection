import unittest
import torch
from torchsummary import summary
from src.models.cnn import CNNModel
from src.models.transformer import TransformerModel
from src.models.svm import SVMModel
from src.models.bayesian import BayesianModel
from src.models.vision_transformer import VisionTransformer

class TestModel(unittest.TestCase):
    def setUp(self):
        """
        Setting up test variables and environment.
        """
        self.input_shape = (3, 224, 224)  
        self.num_classes = 10  

    def test_cnn_model(self):
        """
        Testing CNN model architecture.
        """
        model = CNNModel(num_classes=self.num_classes)
        model.eval()
        sample_input = torch.randn(1, *self.input_shape)
        output = model(sample_input)
        self.assertEqual(output.shape[1], self.num_classes)
        summary(model, self.input_shape)

    def test_transformer_model(self):
        """
        Testing Transformer model architecture.
        """
        model = TransformerModel(
            input_dim=self.input_shape[1] * self.input_shape[2],
            model_dim=512,
            num_heads=8,
            num_layers=6,
            output_dim=self.num_classes
        )
        model.eval()
        sample_input = torch.randn(1, self.input_shape[1] * self.input_shape[2])
        output = model(sample_input)
        self.assertEqual(output.shape[1], self.num_classes)
        summary(model, (self.input_shape[1] * self.input_shape[2],))

    def test_svm_model(self):
        """
        Testing SVM model architecture.
        """
        model = SVMModel()
        sample_input = torch.randn(1, self.input_shape[1] * self.input_shape[2]).numpy()
        output = model.predict(sample_input)
        self.assertEqual(len(output), 1)
        self.assertIn(output[0], range(self.num_classes))

    def test_bayesian_model(self):
        """
        Testing Bayesian model architecture.
        """
        model = BayesianModel()
        sample_input = torch.randn(1, self.input_shape[1] * self.input_shape[2]).numpy()
        output = model.predict(sample_input)
        self.assertEqual(len(output), 1)
        self.assertIn(output[0], range(self.num_classes))

    def test_vision_transformer_model(self):
        """
        Testing Vision Transformer model architecture.
        """
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=self.num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072
        )
        model.eval()
        sample_input = torch.randn(1, *self.input_shape)
        output = model(sample_input)
        self.assertEqual(output.shape[1], self.num_classes)
        summary(model, self.input_shape)

if __name__ == "__main__":
    unittest.main()
