
import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):  
    def __init__(self, n_features, hidden_dim = 128):
        super(Encoder, self).__init__()
        
        self.n_features = n_features
        
        self.hidden_dim = hidden_dim
        
        self.LSTM = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )
              
    def forward(self, X):
        z = []
        for xT in X:
            h_n = torch.zeros(1, 1, self.hidden_dim)
            c_n = torch.zeros(1, 1, self.hidden_dim)
            for xt in xT:
                # Step through the sequence one element at a time.
                # after each step, hidden contains the hidden state.
                out, (h_n, c_n) = self.LSTM(xt.view(1, 1, -1), (h_n, c_n))
            z.append(out)
        return z
                
        

    
# n_features = 3    
# hidden_dim = 2

# lstm = nn.LSTM(input_size=n_features,
#                 hidden_size=hidden_dim,
#                 num_layers=1,
#                 batch_first=True)
# inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5 dim = 3

# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# output, (h_n, c_n) = lstm(inputs)
# z = h_n[0,-1,:]
# output[-1]
 
 
# inputs = [torch.randn(1, 3) for _ in range(3)]  # make a sequence of length 5 dim = 3
# # inputs = torch.cat(inputs).view(len(inputs), 2, -1)

# data = [[torch.zeros(1, 3) for _ in range(2)] ,[torch.ones(1, 3) for _ in range(20)]]



# z = []
# for xT in data:
#     h_n = torch.zeros(1, 1, 2)
#     c_n = torch.zeros(1, 1, 2)
#     for xt in xT:
#         # Step through the sequence one element at a time.
#         # after each step, hidden contains the hidden state.
#         out, (h_n, c_n) = lstm(xt.view(1, 1, -1), (h_n, c_n))
#     z.append(out)    
    
    
 
# E = Encoder(n_features=3, hidden_dim = 2)
# E(data)