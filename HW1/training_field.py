
import numpy as np
from numpy.lib import utils


from neural_network import NeuralNetwork
from losses import CrossEntropy
from activations import Tanh, Relu

import matplotlib.pyplot as plt
from utils import split_to_batches

from copy import deepcopy

nn = NeuralNetwork([2,10,10,2],Tanh(), 0.1)
# nn(np.zeros((2,10)))

# X_train = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 50).T
# X_train = np.concatenate((X_train, np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 50).T), axis=1)
# X_test = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 50).T
# X_test = np.concatenate((X_test, np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 50).T), axis=1)


# y_train = np.zeros((2, 100)).astype(np.float32)
# y_train[0,:50] = 1
# y_train[1,50:] = 1
# y_test = np.zeros((2, 100))
# y_test[0,:50] = 1
# y_test[1,50:] = 1
# plt.scatter(X_train[0,:], X_train[1,:], c = y_train[0,:])
# plt.legend()
# plt.show()



dataset_name = 'SwissRollData'
# dataset_name =  'PeaksData'
from data.dataloader import DataLoader
from utils import shuffle_and_batch, train_test_split, get_index

data = DataLoader.load_dataset(dataset_name)
X_train, y_train, X_test, y_test = train_test_split(data)

m = np.mean(X_train,axis=1).reshape(-1,1)
sd = np.std(X_train,axis=1).reshape(-1,1)

X_train = (X_train - m)/sd
X_test = (X_test - m)/sd


# batch_X_train, batch_y_train = split_to_batches(X_train, y_train)




# X_train = np.array([[0,0],[1,0],[0,1], [1,1]]).T
# y_train = np.array([[0,1],[1,0],[1,0],[0,1]]).T
# X_test = np.array([[0,0],[1,0],[0,1], [1,1]]).T
# y_test = np.array([[0,1],[1,0],[1,0],[0,1]]).T

best_loss = float('inf')
best_model = []
for i in range(100):
    if i % 10 == 0:
        nn.lr *= 0.95
    # for batch_x, batch_y in zip([X_train], [y_train]):
    batch_x , batch_y = X_train, y_train
    y_hat = nn(batch_x)
    nn.backprop(batch_y)
    # nn.model[0].W[0]
    # y_hat_test = nn(X_test)
    # if i % 25 == 0:
    loss_train = CrossEntropy()(y_hat, batch_y)
    acc = (sum(sum(np.round(y_hat) == batch_y))/2) / batch_y.shape[-1]
    if loss_train < best_loss:
        best_loss = loss_train
        best_model = deepcopy(nn)
    # loss_test = CrossEntropy()(y_hat_test, y_test)
    print('loss: {}, acc: {}'.format(loss_train,acc))#, loss_test)
    

    

    
y_pred = best_model(X_train)

plt.scatter(X_train[0,:], X_train[1,:], c = np.round(y_pred[0]))
plt.show()












