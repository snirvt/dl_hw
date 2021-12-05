
import numpy as np
import torch
import torch.nn as nn




class Decoder(nn.Module):  
    def __init__(self, n_features, hidden_dim = 128):
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        self.LSTM = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )
              
    def forward(self, Z, seq_len):
        X = []
        for i,z in enumerate(Z):
            X.append([])
            h_n = torch.zeros(1, 1, self.hidden_dim)
            c_n = torch.zeros(1, 1, self.hidden_dim)
            for _ in range(seq_len[i]):
                # Step through the sequence one element at a time.
                # after each step, hidden contains the hidden state.
                z_out, (h_n, c_n) = self.LSTM(z.view(1, 1, -1), (h_n, c_n)) # repeating the same z but different h_n, c_n
                X[i].append(z_out)
        return X
                








