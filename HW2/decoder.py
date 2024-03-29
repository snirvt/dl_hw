
import numpy as np
import torch
import torch.nn as nn




class Decoder(nn.Module):  
    def __init__(self, n_features , seq_len, hidden_dim ,
     bidirectional=False, num_layers = 1, out_dim = 28):

        super(Decoder, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers  = num_layers
        self.out_dim = out_dim
        self.dim_coeff = (self.bidirectional + 1) * self.num_layers * 2


        self.LSTM = nn.LSTM(
            input_size = n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional = self.bidirectional,
            )

        self.BN = nn.BatchNorm1d(self.dim_coeff * self.hidden_dim)
        self.DO = nn.Dropout(p=0.3)
        self.dense_1 = nn.Linear(self.seq_len*(self.bidirectional + 1)*self.hidden_dim, 2*self.hidden_dim)
        self.relu = nn.ReLU()
        self.BN_2 = nn.BatchNorm1d(2*self.hidden_dim)
        self.dense_test = nn.Linear((self.bidirectional + 1) * self.hidden_dim, self.out_dim)

    def forward(self, Z):
        Z = Z.repeat(1, self.seq_len, 1)
        out, (h_n, c_n) = self.LSTM(Z)
        h_n_last = h_n[-1,:,:].squeeze()
        out = self.dense_test(out)
        return out, h_n_last

