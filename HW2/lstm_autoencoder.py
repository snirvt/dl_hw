

import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder



class LSTMAutoencoder_mnist(nn.Module):
    def __init__(self, n_features, seq_len, latent_dim=32, hidden_dim_enc=128, hidden_dim_dec=128, 
    pix_by_pix=False, bidirectional_enc=False, num_layers_enc = 1, bidirectional_dec=False, num_layers_dec = 1,
     out_dim = 28):
        super().__init__()

        self.encoder = Encoder(n_features, hidden_dim_enc, latent_dim, pix_by_pix, bidirectional_enc, num_layers_enc)
        self.decoder = Decoder(n_features = latent_dim, seq_len = seq_len, hidden_dim = hidden_dim_dec, out_dim=out_dim,
        bidirectional = bidirectional_dec, num_layers = num_layers_dec)

        self.seq_len = seq_len
        self.pix_by_pix = pix_by_pix
        self.dim_coeff = (bidirectional_dec + 1) * num_layers_dec
        self.dense = nn.Linear(hidden_dim_dec, 10) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        Z, _ = self.encoder(X)
        X, hidden_decoder = self.decoder(Z)
        X = self.sigmoid(X)
        pred = self.dense(hidden_decoder)
        return X, pred


class LSTMAutoencoder_syn(nn.Module):
    def __init__(self, n_features, seq_len, hidden_dim_enc=128, hidden_dim_dec=128 ,latent_dim=32,
     bidirectional_enc=False, num_layers_enc = 1, bidirectional_dec=False, num_layers_dec = 1, out_dim = 1):

        super().__init__()
        self.encoder = Encoder(n_features = n_features, hidden_dim = hidden_dim_enc, latent_dim = latent_dim,
            pixel_by_pixel = False, bidirectional=bidirectional_enc, num_layers = num_layers_enc)
        
        self.decoder = Decoder(n_features = latent_dim, seq_len = seq_len, hidden_dim = hidden_dim_dec, 
            bidirectional=bidirectional_dec, num_layers = num_layers_dec, out_dim = out_dim)
        
        self.seq_len = seq_len

    def forward(self, X):
        Z, _ = self.encoder(X)
        X, _ = self.decoder(Z)
        return X



class LSTMAutoencoder_sp500(nn.Module):
    def __init__(self, n_features, seq_len, hidden_dim_enc=128, hidden_dim_dec=128 ,latent_dim=32,
     bidirectional_enc=False, num_layers_enc = 1,
      bidirectional_dec=False, num_layers_dec = 1, out_dim = 1, sub_seq_size = -1):

        super().__init__()
        self.encoder = Encoder(n_features = n_features, hidden_dim = hidden_dim_enc, latent_dim = latent_dim,
            pixel_by_pixel = False, bidirectional=bidirectional_enc, num_layers = num_layers_enc)
        
        self.decoder = Decoder(n_features = latent_dim, seq_len = sub_seq_size, hidden_dim = hidden_dim_dec, 
            bidirectional=bidirectional_dec, num_layers = num_layers_dec, out_dim = out_dim)
        
        self.latent_dim = latent_dim
        if sub_seq_size != -1:
            self.seq_len = seq_len
        else:
            self.seq_len = sub_seq_size
        self.sub_seq_size = sub_seq_size

        self.dense = nn.Linear(self.latent_dim + 1, 1) 

    def forward(self, X):
        for sub_part in range(self.seq_len // self.sub_seq_size):
            sub_x = X[:,sub_part * self.sub_seq_size : (sub_part + 1)*self.sub_seq_size,:] 
            sub_z, full_sub_z = self.encoder(sub_x)
            sub_x, _ = self.decoder(sub_z)

            if sub_part == 0:
                X_reconstructed = sub_x
                Z = full_sub_z
            else:
                X_reconstructed = torch.cat([X_reconstructed, sub_x],axis=1)
                Z = torch.cat([Z, full_sub_z],axis=1)
        if self.seq_len % self.sub_seq_size > 0:
            sub_x = X[:,-(self.seq_len % self.sub_seq_size): ,:] 
            sub_z, full_sub_z = self.encoder(sub_x)
            sub_x, _ = self.decoder(sub_z)
            X_reconstructed = torch.cat([X_reconstructed, sub_x],axis=1).flatten()[:-(self.seq_len % self.sub_seq_size+1)]
            Z = torch.cat([Z, full_sub_z],axis=1)

        X_t = X[:,:-1,:]
        Z_past = torch.cat([X_t, Z[:,:-1,:]],axis=2)
        Z_future = torch.cat([X[:,-1:,:], Z[:,-1:,:]],axis=2)

        Y_t_pred = self.dense(Z_past)
        future_y = self.dense(Z_future)
        return X_reconstructed.reshape(X.shape), Y_t_pred, future_y