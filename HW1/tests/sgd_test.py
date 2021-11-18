#!/usr/bin/env python3

from optimizer import SGD
import matplotlib.pyplot as plt
import numpy as np


from utils import mse

optimizer = SGD(lr=0.1)
X = np.linspace(-5,5, 100)
y = 2*X - 12 + np.random.randn(X.shape[0])
y = y.reshape(-1,1)

W = np.zeros((2,1))

X = np.concatenate((X.reshape(-1,1) , np.ones((X.shape[0],1))), axis=1)
loss = []
for _ in range(100):
    y_pred = X @ W
    loss.append(mse(y,y_pred))
    grad = X.T @ ((y_pred-y) * (1/X.shape[0]))
    W = optimizer.step(grad, W)

y_pred = X @ W
loss.append(mse(y,y_pred))
    
print(f'W0 = {W[0]}')
print(f'W1 = {W[1]}')
print(f'Prediction: {y_pred}')
print(f'Error: {loss[-1]}')
print(f'LSE: {loss}')
fig, axs = plt.subplots(2)
axs[0].plot(loss)
axs[0].set_title('Least squares error per iteration')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Loss value')
axs[1].plot(X[:,0], y_pred, color='r')
axs[1].scatter(X[:,0], y)
axs[1].legend(['Prediction', 'Actual y'])
axs[1].set_title('Graphs')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
fig.tight_layout()
plt.savefig('../figures/sgd_test.png', transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()
