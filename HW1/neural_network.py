
import numpy as np

from dense_layer import Dense
from softmax_regression import SoftmaxRegression
from activations import Tanh, Relu

from copy import deepcopy

from losses import CrossEntropy

class NeuralNetwork():
    def __init__(self, dim_list, activation=Tanh(), lr=0.01):
        self.dim_list = dim_list
        self.lr = lr
        self.activation = activation
        self.model = []
        self.g_X = []
        self.g_W = []
        self.g_b = []
        self.X_list = []
        self.ff_res = np.nan
        
        self.init_model()

    
    def init_model(self): # dim_list = [2,3,2]
        for i in range(len(self.dim_list)-2):
            self.model.append(Dense(self.dim_list[i], self.dim_list[i+1], self.activation))
        self.model.append(SoftmaxRegression(self.dim_list[-2], self.dim_list[-1]))    
        
    def feed_forward(self, X):
        result = deepcopy(X)
        self.X_list = [result]
        for i in range(len(self.model)):
            result = self.model[i](result)
            self.X_list.append(result)
        self.ff_res = result
        return result
    
    def loss(self, y_hat, C):
        m = C.shape[-1]  # batchsize
        loss = np.sum(C * np.log(y_hat))
        return (-1 / m) * loss
        
    def step_layer(self, layer):
        layer.W -= self.lr * layer.g_W
        layer.b -= self.lr * layer.g_b
        
    def backprop(self, C, update=True):
        self.g_X = []
        self.g_W = []
        self.g_b = []
        self.model[-1].gradient(self.X_list[-2],self.X_list[-1], C) ## softmax has no b
        g_X = self.model[-1].g_X
        g_W = self.model[-1].g_W
        g_b = self.model[-1].g_b
        self.g_X.append(g_X)
        self.g_W.append(g_W)
        self.g_b.append(g_b)
                        
        # self.model[-1].W -= self.lr * self.model[-1].W
        if update:
            self.step_layer(self.model[-1])
        
        for i in reversed(range(len(self.model)-1)):
            self.model[i].jacTMV_W(self.X_list[i], g_X)
            self.model[i].jacTMV_b(self.X_list[i], g_X)
            g_X = self.model[i].jacTMV_x(self.X_list[i], g_X)
            self.g_X.append(g_X)
            self.g_W.append(self.model[i].g_W)
            self.g_b.append(self.model[i].g_b)
            if update:
                self.step_layer(self.model[i])
        
        

    def __call__(self, X):
        return self.feed_forward(X)
    


