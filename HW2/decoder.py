
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
        # self.dense_1 = nn.Linear(self.dim_coeff *self.hidden_dim, 2*self.hidden_dim)
        self.dense_1 = nn.Linear(self.seq_len*(self.bidirectional + 1)*self.hidden_dim, 2*self.hidden_dim)
        self.relu = nn.ReLU()
        self.BN_2 = nn.BatchNorm1d(2*self.hidden_dim)
        # self.dense_2 = nn.Linear(2*self.hidden_dim, self.out_dim) # out_dim = 28*28

        self.dense_test = nn.Linear((self.bidirectional + 1) * self.hidden_dim, self.out_dim)

    def forward(self, Z):
        Z = Z.repeat(1, self.seq_len, 1)
        out, (h_n, c_n) = self.LSTM(Z)
        h_n_last = h_n[-1,:,:].squeeze()
        # h_n =  h_n.permute(1,2,0).reshape(-1, h_n.shape[0] * h_n.shape[-1])
        # c_n = c_n.permute(1,2,0).reshape(h_n.shape)
        # out = torch.concat([h_n, c_n], axis=1)
        # out = nn.Dropout(p=0.2)(out)
        # out = self.dense_1(out)
        # out = self.relu(out)
        # out = self.dense_2(out)
        out = self.dense_test(out)
        return out, h_n_last



# class Decoder(nn.Module):
#     def _init_(self, args):
#         super(Decoder, self)._init_()

#         self.batch_size = args.batch_size
#         self.input_size = args.input_size
#         self.embedding_dim = args.hidden_state_size

#         self.lstm_dec = nn.LSTM(
#             input_size=self.embedding_dim,
#             hidden_size=self.embedding_dim,
#             num_layers=1,
#             batch_first=True
#         )

#         self.output_layer = nn.Linear(self.embedding_dim, self.input_size)

#     def forward(self, x):
#         z, (hidden_state, cell_state) = self.lstm_dec(x)
#         return self.output_layer(z), hidden_state


