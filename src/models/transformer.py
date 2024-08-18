import torch
from torch import nn
import math
import logging

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        """
        Initializing Transformer model.
        :param input_dim: Dimension of the input features
        :param model_dim: Dimension of the transformer model
        :param num_heads: Number of attention heads
        :param num_layers: Number of transformer layers
        :param output_dim: Dimension of the output (number of classes)
        :param dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()
        self.logger = logging.getLogger('transformer_model_logger')
        self.logger.info(f"Initializing Transformer model with input_dim={input_dim}, model_dim={model_dim}, num_heads={num_heads}, num_layers={num_layers}, output_dim={output_dim}, dropout={dropout}.")
        
        try:
            self.embedding = nn.Linear(input_dim, model_dim)
            self.positional_encoding = PositionalEncoding(model_dim, dropout)
            encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads, dim_feedforward=model_dim * 4, dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
            self.decoder = nn.Linear(model_dim, output_dim)
            self.logger.info("Transformer model initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing Transformer model: {e}", exc_info=True)
            raise
    
    def forward(self, src):
        """
        To forward pass of transformer model.
        :param src: Input tensor
        :return: Output tensor
        """
        self.logger.info(f"Performing forward pass with input tensor of shape {src.shape}.")
        
        try:
            src = self.embedding(src) * math.sqrt(src.size(1))
            src = self.positional_encoding(src)
            output = self.transformer_encoder(src)
            output = self.decoder(output.mean(dim=1))
            self.logger.info(f"Forward pass completed with output tensor of shape {output.shape}.")
            return output
        except Exception as e:
            self.logger.error(f"Error during forward pass: {e}", exc_info=True)
            raise

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializing Positional Encoding.
        :param d_model: Dimension of the model
        :param dropout: Dropout rate
        :param max_len: Maximum length of the input sequences
        """
        super(PositionalEncoding, self).__init__()
        self.logger = logging.getLogger('positional_encoding_logger')
        self.logger.info(f"Initializing Positional Encoding with d_model={d_model}, dropout={dropout}, max_len={max_len}.")
        
        try:
            self.dropout = nn.Dropout(p=dropout)
            
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
            self.logger.info("Positional Encoding initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing Positional Encoding: {e}", exc_info=True)
            raise
    
    def forward(self, x):
        """
        To forward pass of positional encoding.
        :param x: Input tensor
        :return: Tensor with positional encoding added
        """
        self.logger.info(f"Performing forward pass with input tensor of shape {x.shape}.")
        
        try:
            x = x + self.pe[:x.size(0), :]
            x = self.dropout(x)
            self.logger.info(f"Forward pass of Positional Encoding completed with output tensor of shape {x.shape}.")
            return x
        except Exception as e:
            self.logger.error(f"Error during forward pass of Positional Encoding: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('transformer_model_logger')
    logger.info("Starting to initialize and build the Transformer model for testing purposes.")
    
    input_dim = 512
    model_dim = 512
    num_heads = 8
    num_layers = 6
    output_dim = 2  
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
    
    logger.info("Transformer model initialized and built successfully.")
