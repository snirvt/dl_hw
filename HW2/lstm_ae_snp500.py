
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from copy import deepcopy

from lstm_autoencoder import LSTMAutoencoder_sp500

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model,lr,gradient_clipping, train_loader, val_loader, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    AE_reg = 0.1
    PRED_reg = 1
    criterion_AE = nn.MSELoss(reduction='mean').to(device)
    criterion_MSE = nn.MSELoss(reduction='mean').to(device)
    history = dict(train=[], val=[], pred_train = [], pred_val = [])
    best_model_wts = deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        train_accuracies = []
        pred_train = []
        for batch_seq in train_loader:
            optimizer.zero_grad()
            batch_seq = batch_seq.to(device)
            X = batch_seq.reshape(len(batch_seq),model.seq_len,-1)
            Y_t = X[:,1:,:]
            seq_pred, Y_t_pred, _ = model(X)
            assert(~np.isnan(seq_pred.detach().cpu().numpy()).any())

            loss_AE = AE_reg * criterion_AE(seq_pred.reshape(batch_seq.shape), batch_seq)
            loss_MSE = PRED_reg * criterion_MSE(Y_t_pred, Y_t)
             
            loss = loss_AE + loss_MSE
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            train_losses.append(loss_AE.item())
            pred_train.append(loss_MSE.item())

        val_losses = []
        pred_val = []
        model = model.eval()
        with torch.no_grad():
            for batch_seq in val_loader:
                # seq_true = seq_true#.to(device)
                batch_seq = batch_seq.to(device)
                X = batch_seq.reshape(len(batch_seq),model.seq_len,-1)
                Y_t = X[:,1:,:]
                seq_pred,Y_t_pred, _ = model(X)
                loss_AE = AE_reg * criterion_AE(seq_pred.reshape(batch_seq.shape), batch_seq)
                loss_MSE = PRED_reg * criterion_MSE(Y_t_pred, Y_t)
                loss = loss_AE + loss_MSE
                val_losses.append(loss_AE.item())
                pred_val.append(loss_MSE.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        pred_train_loss = np.mean(pred_train)
        pred_val_loss = np.mean(pred_val)
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        history['pred_train'].append(pred_train_loss)
        history['pred_val'].append(pred_val_loss)

        if val_loss + pred_val_loss < best_loss:
            best_loss = val_loss + pred_val_loss
            best_model_wts = deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train_loss: {train_loss:0.3f}, val_loss {val_loss:0.3f}, pred_train_loss: {pred_train_loss:0.3f}, pred_val_loss {pred_val_loss:0.3f}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history


''' 3.3.1 '''
stocks = pd.read_csv('data/sp500/SP 500 Stock Prices 2014-2017.csv')
stocks.shape

google = stocks[stocks['symbol'] == 'GOOGL']
amazon = stocks[stocks['symbol'] == 'AMZN']

plt.plot(google['high'].values)
plt.title('Google Stocks - High')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('figures_Q3/google_high.png')
plt.close()


plt.plot(amazon['high'].values)
plt.title('Amazon Stocks - High')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('figures_Q3/amazon_high.png')
plt.close()


symbols = set(stocks['symbol'])

highs = torch.zeros((477,1007))
row_symbol = {}
cnt = 0
for symbol in symbols:
    high = stocks[stocks['symbol'] == symbol]['high'].values
    if len(high) == 1007 and ~np.isnan(high).any():
        highs[cnt] = torch.Tensor(high)
        row_symbol[cnt] = symbol
        cnt+=1
# np.save('highs.npy', highs)
# np.save('row_symbol.npy', row_symbol)

# ''' shortcut '''
# highs = np.load('highs.npy')
# row_symbol = np.load('row_symbol.npy', allow_pickle=True).item()

key_list = list(row_symbol.keys())

data_idx = np.random.choice(range(len(key_list)), len(key_list), replace=False)
Train_data = highs[data_idx[:int(0.5*len(data_idx))],:]
Val_data = highs[data_idx[int(0.5*len(data_idx)):int(0.8*len(data_idx))],:]
Test_data = highs[data_idx[int(0.8*len(data_idx)):],:]


mean_train = Train_data.mean(axis=1).reshape(-1,1)
std_train = Train_data.std(axis=1).reshape(-1,1)
mean_val = Val_data.mean(axis=1).reshape(-1,1)
std_val = Val_data.std(axis=1).reshape(-1,1)
mean_test = Test_data.mean(axis=1).reshape(-1,1)
std_test = Test_data.std(axis=1).reshape(-1,1)

Train_data = (Train_data - mean_train)/std_train
Val_data =   (Val_data - mean_val)/std_val
Test_data =  (Test_data - mean_test)/std_test


batch_size = 64
train_loader = torch.utils.data.DataLoader(Train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
val_loader = torch.utils.data.DataLoader(Val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
test_loader = torch.utils.data.DataLoader(Test_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

seq_len = Train_data.shape[1]

n_epochs = 150
model = LSTMAutoencoder_sp500(n_features=1, seq_len=seq_len, hidden_dim_enc=64, hidden_dim_dec=64, latent_dim=32, bidirectional_enc=True,
num_layers_enc = 2, bidirectional_dec=True, num_layers_dec = 1, sub_seq_size = 53, out_dim = 1)

model = model.to(device)
model, history = train_model(model,lr=0.003,gradient_clipping=1,
 train_loader=train_loader, val_loader=val_loader,n_epochs=n_epochs)




''' ae loss plot'''
plt.plot(history['train'], label='Train Loss')
plt.plot(history['val'], label='Validation Loss')
plt.title('Reconstruction Loss Over Time')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.savefig('figures_Q3/ae_loss_sp500.png')
plt.close()
'''' '''


''' pred loss plot'''
plt.plot(history['pred_train'], label='Train Loss')
plt.plot(history['pred_val'], label='Validation Loss')
plt.title('Prediction Loss Over Time')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.savefig('figures_Q3/pred_loss_sp500.png')
plt.close()
'''' '''



''' Create Plots '''
seq = next(iter(test_loader))

'''3.3.2 autoencoder'''
for i,seq_num in enumerate([0, 1, 2]):
    pred_seq_reconstruct = model(seq[seq_num].reshape(1,seq_len,1).to(device))[0]
    pred_seq_reconstruct = pred_seq_reconstruct.squeeze()
    plt.plot(seq[seq_num]*std_test[i]+mean_test[i], label='Test Signal')
    plt.plot(pred_seq_reconstruct.detach().to('cpu')*std_test[i]+mean_test[i], label='Test prediction')
    plt.title(f'Ex: {i}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'figures_Q3/sp500_{seq_num}_332.png')
    plt.close()
''' '''


''' 3.3.3 1 step'''
for i,seq_num in enumerate([0, 1, 2]):
    pred_seq_1_step = model(seq[seq_num].reshape(1,seq_len,1).to(device))[1]
    pred_seq_1_step = pred_seq_1_step.squeeze()
    plt.plot(seq[seq_num]*std_test[i]+mean_test[i], label='Test Signal')
    plt.plot(pred_seq_1_step.detach().to('cpu')*std_test[i]+mean_test[i], label='Test prediction')
    plt.title(f'Ex: {i}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'figures_Q3/sp500_{seq_num}_333.png')
    plt.close()
''' '''


''' 3.3.4 n/2 steps (Auto-regression)'''
for i,seq_num in enumerate([0, 1, 2]):
    X = seq[seq_num].to(device)
    X_half = X[:len(X)//2]
    model.seq_len = len(X)//2
    res = []
    for j in range(len(X)//2+1):
        pred_seq = model(X_half.reshape(1,len(X_half),1).to(device),True)[2].to(device)
        pred_seq = pred_seq.reshape(1)
        res.append(pred_seq.to('cpu').detach()[0])
        X_half = torch.cat([X_half[1:],pred_seq])
    res = X[:len(X)//2].to('cpu').detach().tolist() + res
    plt.plot(seq[seq_num]*std_test[i]+mean_test[i], label='Test Signal')
    plt.plot(torch.Tensor(res)*std_test[i]+mean_test[i], label='Test prediction')
    plt.title(f'Ex: {i}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'figures_Q3/sp500_{seq_num}_334.png')
    plt.close()
    model.seq_len = len(X)
''' '''



