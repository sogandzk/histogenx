import torch.nn as nn

def mlp_builder(in_dim, hidden_dims, out_dim, dropout_probs, batchnorm):
    """
    Build a flexible MLP with ReLU, optional BatchNorm, and per-layer dropout
    """
    assert len(hidden_dims) == len(dropout_probs), "hidden_dims and dropout_probs must match"
    
    layers = []

    for h, dp in zip(hidden_dims, dropout_probs):
        layers.append(nn.Linear(in_dim, h))
        if batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dp))
        in_dim = h

    layers.append(nn.Linear(in_dim, out_dim))
    
    return nn.Sequential(*layers)
