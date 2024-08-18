import torch
from torch import nn
import math
import logging

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1, emb_dropout=0.1):
        """
        Initializing Vision Transformer model.
        :param img_size: Size of the input image
        :param patch_size: Size of the patches to divide the image into
        :param num_classes: Number of output classes
        :param dim: Dimension of the transformer model
        :param depth: Number of transformer layers
        :param heads: Number of attention heads
        :param mlp_dim: Dimension of the feedforward network
        :param dropout: Dropout rate for transformer layers
        :param emb_dropout: Dropout rate for the embedding
        """
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by the patch size."
        
        logging.info("Initializing Vision Transformer with img_size=%s, patch_size=%s, num_classes=%s, dim=%s, depth=%s, heads=%s, mlp_dim=%s, dropout=%s, emb_dropout=%s",
                     img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout)

        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        """
        To forward pass of Vision Transformer model.
        :param img: Input image tensor
        :return: Output tensor
        """
        logging.info("Performing forward pass of Vision Transformer...")
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(x.size(0), -1, 3 * p * p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        output = self.mlp_head(x)
        logging.info("Forward pass complete.")
        return output

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        """
        Initializing Transformer model used within Vision Transformer.
        :param dim: Dimension of the transformer model
        :param depth: Number of transformer layers
        :param heads: Number of attention heads
        :param mlp_dim: Dimension of the feedforward network
        :param dropout: Dropout rate for transformer layers
        """
        super(Transformer, self).__init__()
        logging.info("Initializing Transformer with dim=%s, depth=%s, heads=%s, mlp_dim=%s, dropout=%s", dim, depth, heads, mlp_dim, dropout)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                MultiHeadAttention(dim, heads, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        """
        To forward pass of Transformer model.
        :param x: Input tensor
        :return: Output tensor
        """
        logging.info("Performing forward pass of Transformer...")
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        logging.info("Forward pass of Transformer complete.")
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        """
        Initializing Multi-Head Attention mechanism.
        :param dim: Dimension of the transformer model
        :param heads: Number of attention heads
        :param dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        logging.info("Initializing Multi-Head Attention with dim=%s, heads=%s, dropout=%s", dim, heads, dropout)
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        To forward pass of Multi-Head Attention mechanism.
        :param x: Input tensor
        :return: Output tensor
        """
        logging.info("Performing forward pass of Multi-Head Attention...")
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, -1)
        output = self.to_out(out)
        logging.info("Forward pass of Multi-Head Attention complete.")
        return output

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        """
        Initializing FeedForward network.
        :param dim: Dimension of the transformer model
        :param hidden_dim: Dimension of the hidden layer
        :param dropout: Dropout rate
        """
        super(FeedForward, self).__init__()
        logging.info("Initializing FeedForward with dim=%s, hidden_dim=%s, dropout=%s", dim, hidden_dim, dropout)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        To forward pass of FeedForward network.
        :param x: Input tensor
        :return: Output tensor
        """
        logging.info("Performing forward pass of FeedForward...")
        output = self.net(x)
        logging.info("Forward pass of FeedForward complete.")
        return output
