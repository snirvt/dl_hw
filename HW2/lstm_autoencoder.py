

import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder



class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=128):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, hidden_dim)#.to(device)
        self.decoder = Decoder(hidden_dim, n_features)#.to(device)
    
    def get_seq_len(self,X):
        seq_len = []
        for x in X:
            seq_len.append(len(x))
        return seq_len
    
    def forward(self, X):
        seq_len = self.get_seq_len(X)
        X = self.encoder(X)
        X = self.decoder(X, seq_len)
        return X


# data = [[torch.zeros(1, 3) for _ in range(2)] ,[torch.ones(1, 3) for _ in range(3)]]
# E = LSTMAutoencoder(n_features=3, hidden_dim = 2)
# E(data)

