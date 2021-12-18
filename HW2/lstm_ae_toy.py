import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from copy import deepcopy

from lstm_autoencoder import LSTMAutoencoder_syn
from ex_1_toy_maker import get_toy_data, toy_data_splitter

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model,lr,gradient_clipping, train_loader, val_loader, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_AE = nn.MSELoss(reduction='mean').to(device)
    history = dict(train=[], val=[], train_acc=[], val_acc=[])
    best_model_wts = deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        train_accuracies = []
        for batch_seq in train_loader:
            optimizer.zero_grad()
            batch_seq = batch_seq.to(device)
            seq_pred = model(batch_seq.reshape(len(batch_seq),model.seq_len,-1))
            loss_AE = criterion_AE(seq_pred.reshape(batch_seq.shape), batch_seq)
            loss = loss_AE 
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for batch_seq in val_loader:
                # seq_true = seq_true#.to(device)
                batch_seq = batch_seq.to(device)
                seq_pred = model(batch_seq.reshape(len(batch_seq),model.seq_len,-1))
                loss_AE = criterion_AE(seq_pred.reshape(len(batch_seq),batch_seq.shape[-1]), batch_seq)
                loss = loss_AE 
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train_loss: {train_loss:0.3f}, val_loss {val_loss:0.3f}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history


''' Get Data'''
data = get_toy_data(n = 10000, T = 50)
train_loader, val_loader, test_loader = toy_data_splitter(data, batch_size = 512)


''' Grid Serach '''
grid_search = False
if grid_search:
    best_val = float('inf')
    best_model = []
    history_dict = {}
    n_epochs = 100
    for lr in [0.1, 0.01, 0.003]:
        for gradient_clipping in [1, 0.5, 0.1]:
            for hidden_state_size in [32, 64, 128]:
                model = LSTMAutoencoder_syn(n_features=1, seq_len=50, hidden_dim = hidden_state_size, latent_dim=32, bidirectional_enc=True,
                num_layers_enc = 2, bidirectional_dec=False, num_layers_dec = 1, out_dim = 50)
                model = model.to(device)
                hyper_param_str = f'lr: {lr}, gc:{gradient_clipping}, hs: {hidden_state_size}'
                model, history = train_model(model,lr,gradient_clipping, train_loader, test_loader, n_epochs)
                history_dict[hyper_param_str + '_model'] = model
                history_dict[hyper_param_str + '_history'] = history
                min_val = np.min(history['val'])
                print(hyper_param_str + f': min_val: {min_val}')
                if min_val < best_val:
                    print('New Best')
                    best_val = np.min(history['val'])
                    best_model = model
                    history_dict['best_model'] = model
                    history_dict['best_params'] = f'lr: {lr}, gc:{gradient_clipping}, hs: {hidden_state_size}'
    np.save('grid_history.npy', history_dict) 
''' '''



''' Single Run '''
n_epochs = 100
model = LSTMAutoencoder_syn(n_features=1, seq_len=50, hidden_dim_enc=128, hidden_dim_dec=128, latent_dim=32, bidirectional_enc=True,
num_layers_enc = 2, bidirectional_dec=True, num_layers_dec = 1, out_dim = 1)
model = model.to(device)
model, history = train_model(model,lr=0.003,gradient_clipping=1,
 train_loader=train_loader, val_loader=val_loader,n_epochs=n_epochs)

plt.plot(history['train'], label='Train Loss')
plt.plot(history['val'], label='Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_toy.png')
plt.close()
'''' '''



''' Create Plots '''
seq = next(iter(test_loader))
for i,seq_num in enumerate([0, 1, 2]):
    pred_seq = model(seq[seq_num].reshape(1,50,1).to(device))
    pred_seq = pred_seq.reshape(50)
    plt.plot(seq[seq_num].numpy(), label='Test Signal')
    plt.plot(pred_seq.detach().to('cpu').numpy(), label='Test prediction')
    plt.title(f'Ex: {i}')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.savefig(f'syn{seq_num}.png')
    plt.close()
''' '''