import torch.nn as nn
from .mlp_builder import mlp_builder

class ImgEncoder(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hidden_dims, 
        out_dim, 
        dropout_probs,
        batchnorm):
        super().__init__()
        
        self.mlp = mlp_builder(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            dropout_probs=dropout_probs,
            batchnorm=batchnorm
        )

    def forward(self, x):
        z = self.mlp(x)
        return z
    
