import numpy as np

class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, grad, W):
        return np.subtract(W, self.lr * grad)