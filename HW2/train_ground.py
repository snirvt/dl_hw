

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from lstm_autoencoder import LSTMAutoencoder

import torch
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor



def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()#.to(device)
    history = dict(train=[], val=[])
    best_model_wts = deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for seq_true in train_dataset:
            optimizer.zero_grad()
            # seq_true = seq_true#.to(device)
            seq_pred = model(seq_true.view(-1, 28, len(seq_true)))
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                # seq_true = seq_true#.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history


# data = [[torch.zeros(1, 3) for _ in range(2)] ,[torch.ones(1, 3) for _ in range(3)]]
# model = LSTMAutoencoder(n_features=3, hidden_dim = 2)

# train_model(model, data, data, 5)






mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())


train_loader = torch.utils.data.DataLoader(mnist_trainset,
                            batch_size=128,
                            shuffle=True,
                            num_workers=1)

test_loader = torch.utils.data.DataLoader(mnist_testset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=1)


model = LSTMAutoencoder(n_features=28, hidden_dim = 2)

train_model(model, train_loader, test_loader, 5)


