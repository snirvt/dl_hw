
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import utils
from copy import deepcopy


from neural_network import NeuralNetwork
from dense_layer import Dense
from res_layer import ResLayer
from losses import CrossEntropy
from activations import Tanh, Relu, Sigmoid
from utils import split_to_batches,shuffle_and_batch
from configurations import config



lr = 0.1
# nn = NeuralNetwork([2,64,64,2],Tanh(), lr, Dense)
lr = 0.2
nn = NeuralNetwork([2,64,64,2],Sigmoid(), lr, ResLayer)

dataset_name = 'SwissRollData'
# dataset_name =  'PeaksData'
from data.dataloader import DataLoader
from utils import shuffle_and_batch, train_test_split, get_index

data = DataLoader.load_dataset(dataset_name)
X_train, y_train, X_test, y_test = train_test_split(data)

m = np.mean(X_train,axis=1).reshape(-1,1)
sd = np.std(X_train,axis=1).reshape(-1,1)

# X_train = (X_train - m)/sd
# X_test = (X_test - m)/sd


X_train, y_train, train_loader, = shuffle_and_batch(X_train, y_train)
X_test, y_test, test_loader, = shuffle_and_batch(X_test, y_test)

batch_X_train, batch_y_train = train_loader
batch_X_test, batch_y_test = test_loader



def accuracy(y_hat, y):
    return (sum(sum(np.round(y_hat) == y))/2) / y.shape[-1]




best_loss = float('inf')
best_model = []
err_train = []
err_test = []
acc_train = []
acc_test = []


for i in range(100):
    if i % 25 == 0:
        nn.lr *= 0.95
        
    err_train.append(0)
    err_test.append(0)
    acc_train.append(0)
    acc_test.append(0)
        
    for batch_x, batch_y in zip(batch_X_train, batch_y_train):
        y_hat = nn(batch_x)
        nn.backprop(batch_y)
        loss_train = CrossEntropy()(y_hat, batch_y)
        acc = accuracy(y_hat, batch_y)
        err_train[i] += loss_train
        acc_train[i] += acc
        
    for batch_x, batch_y in zip(batch_X_test, batch_y_test):
        y_test_hat = nn(batch_x)
        loss_test = CrossEntropy()(y_test_hat, batch_y)
        acc_val = accuracy(y_test_hat, batch_y)
        err_test[i] += loss_test
        acc_test[i] += acc_val

        
        
    err_train[i] /= len(batch_X_train)
    err_test[i]  /= len(batch_X_test)
    acc_train[i] /= len(batch_X_train)
    acc_test[i]  /= len(batch_X_test)    
              
    if err_test[i] < best_loss:
        best_loss = err_test[i]
        best_model = deepcopy(nn)
    print('loss: {:.4f}, acc: {:.4f}, loss val: {:.4f}, acc val: {:.4f}'.format(err_train[i],acc_train[i], err_test[i],acc_test[i]))
    
y_pred = best_model(X_train)

plt.scatter(X_train[0,:], X_train[1,:], c = np.round(y_pred[0]))
plt.show()

plt.scatter(X_train[0,:], X_train[1,:], c = np.round(y_train[0]))
plt.show()



plt.rcParams.update({'font.size': 22})

plt.plot(err_train, label='Train Error')
plt.plot(err_test, label='Test Error')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.ylim(0,1)
plt.legend()
plt.show()


plt.plot(acc_train, label='Train Accuracy')
plt.plot(acc_test, label='Test Accuracy')
plt.title('Accuracy - lr: {},  batch size: {}'.format(lr,config.BATCH_SIZE))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.legend()
plt.show()








