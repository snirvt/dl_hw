
import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):  
    def __init__(self, n_features, hidden_dim = 128, latent_dim=32, pixel_by_pixel = False,
    bidirectional=False, num_layers = 1):
        super(Encoder, self).__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.pixel_by_pixel = pixel_by_pixel
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        self.dim_coeff = (self.bidirectional + 1) * self.num_layers * 2
        
        self.LSTM = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional = bidirectional,
            )

        self.BN = nn.BatchNorm1d(self.dim_coeff * self.hidden_dim)  
        self.DO = nn.Dropout(p=0.3)  
        self.dense_1 = nn.Linear(self.dim_coeff * self.hidden_dim, 2*self.hidden_dim)
        self.relu = nn.ReLU()
        self.BN_2 = nn.BatchNorm1d(2*self.hidden_dim)
        self.dense_2 = nn.Linear(2*self.hidden_dim, self.latent_dim)
        self.dense_test = nn.Linear((self.bidirectional + 1) * self.hidden_dim, self.latent_dim)

    def forward(self, X):
        if self.pixel_by_pixel:
            X = X.reshape(X.shape[0],-1,1)
        out, (h_n, c_n) = self.LSTM(X)
        out_full = self.dense_test(out)
        h_n = h_n.permute(1,2,0).reshape(-1, h_n.shape[0] * h_n.shape[-1])
        c_n = c_n.permute(1,2,0).reshape(h_n.shape)
        out = torch.concat([h_n, c_n], axis=1)
        out = self.BN(out)
        out = nn.Dropout(p=0.2)(out)
        out = self.dense_1(out)
        out = self.relu(out)
        out = self.BN_2(out)
        out = self.dense_2(out)
        return out.reshape(-1,1,self.latent_dim), out_full
