

import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

from lstm_autoencoder import LSTMAutoencoder_mnist



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion_AE = nn.MSELoss().to(device)
    criterion_CE = nn.CrossEntropyLoss().to(device)

    history = dict(train=[], val=[], train_acc=[], val_acc=[])
    best_model_wts = deepcopy(model.state_dict())
    best_loss = float('inf')
    CE_coeff = 0.01
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        train_accuracies = []
        for images, labels in train_dataset:
            optimizer.zero_grad()
            images = images.to(device)
            seq_pred, pred = model(images.squeeze())
            loss_AE = criterion_AE(seq_pred.reshape(len(images),1,images.shape[-2],images.shape[-1]), images)

            pred = pred.reshape(images.shape[0],10)
            labels = labels.to(device)
            loss_CE = criterion_CE(pred, labels)
            acc = torch.sum(torch.argmax(pred,axis=1) == labels).item()/len(pred)

            loss = loss_AE + CE_coeff*loss_CE
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_losses.append(loss.item())
            train_accuracies.append(acc)
        val_losses = []
        val_accuracies = []
        model = model.eval()
        with torch.no_grad():
            for images, labels in val_dataset:
                images = images.to(device)
                seq_pred, pred = model(images.reshape(len(images),model.seq_len,-1))
                loss_AE = criterion_AE(seq_pred.reshape(len(images),1,images.shape[-2],images.shape[-1]), images)
                pred = pred.reshape(images.shape[0],10)
                labels = labels.to(device)
                loss_CE = criterion_CE(pred, labels)
                acc = torch.sum(torch.argmax(pred,axis=1) == labels).item()/len(pred)
                loss = loss_AE + CE_coeff*loss_CE
                val_losses.append(loss.item())
                val_accuracies.append(acc)
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = np.mean(train_accuracies)
        val_acc = np.mean(val_accuracies)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train_loss: {train_loss:0.3f}, train_acc {train_acc:0.3f}, val_loss {val_loss:0.3f}, val_acc {val_acc:0.3f}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history


def normalize(img):
    img *= (1.0/img.max())
    return img

transform = transforms.Compose([
                transforms.ToTensor(),
                normalize   
            ])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_trainset,
                            batch_size=1024,
                            shuffle=True,
                            num_workers=1)

test_loader = torch.utils.data.DataLoader(mnist_testset,
                            batch_size=1024,
                            shuffle=False,
                            num_workers=1)

pix_by_pix = False
if not pix_by_pix:
    model = LSTMAutoencoder_mnist(n_features=28,seq_len=28, latent_dim=512, hidden_dim_enc = 64, hidden_dim_dec = 64,
    pix_by_pix = pix_by_pix, bidirectional_enc=True, num_layers_enc = 2, bidirectional_dec=False, num_layers_dec = 1) 
    model = model.to(device)
    model, history = train_model(model, train_loader, test_loader, 50)
else:
    model = LSTMAutoencoder_mnist(n_features=1,seq_len=784,latent_dim=128, hidden_dim_enc = 32, hidden_dim_dec = 32,
    pix_by_pix=pix_by_pix, bidirectional_enc=True, num_layers_enc = 2, bidirectional_dec=False, num_layers_dec = 1, out_dim = 1)
    model = model.to(device)
    model, history = train_model(model, train_loader, test_loader, 50)

''' Loss + Accuracy'''
plt.plot(history['train'], label='Train Loss')
plt.plot(history['val'], label='Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.savefig('figures_Q2/loss_mnist.png')
plt.close()

plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy Over Time')
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('figures_Q2/acc_mnist.png')
plt.close()
'''' '''


''' Create prediction plots '''
im, lable = next(iter(test_loader))
for im_num,seq_num in enumerate([0, 1, 2]):
    plt.imshow(im[im_num].permute(1, 2, 0).numpy(), cmap='gray')
    plt.savefig(f'figures_Q2/original_pix_by_pix_{seq_num}.png')
    plt.close()
    pred, lable = model(im[im_num].to(device))
    lable_pred = np.argmax(lable.detach().to('cpu').numpy())
    plt.imshow(pred.reshape(28, 28).detach().to('cpu').numpy(), cmap='gray')
    plt.title(f'Ex: {im_num}, prediction: {lable_pred}')
    plt.savefig(f'figures_Q2/reconstructed_pix_by_pix_{seq_num}.png')
    plt.close()
''' '''


