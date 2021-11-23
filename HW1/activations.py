
import numpy as np


class Tanh:
    def __init__(self):
        self.activate = np.tanh
        self.deriv = lambda x: np.subtract(np.ones(x.shape), np.multiply(np.tanh(x), np.tanh(x)))
        # self.deriv = lambda x: np.subtract(np.ones(x.shape), np.multiply(x, x))

    def __call__(self, X):
        return self.activate(X)
    
    
    
class Sigmoid:
    def __init__(self):
        self.activate = lambda x: 1/(1 + np.exp(-x))
        self.deriv = lambda x: self.activate(x) * (1 - self.activate(x))
    
    def __call__(self, X):
        return self.activate(X)
    
    
    
class Relu():
    def __init__(self):
        self.activate = lambda x: x * (x > 0)
        self.deriv = lambda x: 1. * (x > 0)
        
            
    def __call__(self, X):
        return self.activate(X)