
import numpy as np

class CrossEntropy():
    def __init__(self):
        pass
    
    def loss(self, y_hat, C):
        m = C.shape[-1]  # batchsize
        loss = np.sum(C * np.log(y_hat))
        return (-1 / m) * loss
    
    def __call__(self, y_hat, C):
        return self.loss(y_hat, C)



