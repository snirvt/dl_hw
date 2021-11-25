import numpy as np
from copy import deepcopy

from dense_layer import Dense
from res_layer import ResLayer
from softmax_regression import SoftmaxRegression
from activations import Tanh, Relu
from losses import CrossEntropy

class NeuralNetwork():
    def __init__(self, dim_list, activation=Tanh(), lr=0.01, type = Dense):
        self.dim_list = dim_list
        self.lr = lr
        self.activation = activation
        self.type = type
        self.model = []
        self.X_list = []

        self.init_gradient_tracks()
        self.init_model()

    
    def init_model(self): 
        if self.type == Dense:
            for i in range(len(self.dim_list)-2):
                self.model.append(Dense(self.dim_list[i], self.dim_list[i+1], self.activation))
            self.model.append(SoftmaxRegression(self.dim_list[-2], self.dim_list[-1])) 
        if self.type == ResLayer:
            for i in range(len(self.dim_list)-2):
                self.model.append(ResLayer(self.dim_list[0], self.dim_list[i+1], self.activation))
            self.model.append(SoftmaxRegression(self.dim_list[0], self.dim_list[-1])) 
            
    def feed_forward(self, X):
        result = deepcopy(X)
        self.X_list = [result]
        for i in range(len(self.model)):
            result = self.model[i](result)
            self.X_list.append(result)
        return result
    
        
    def step_layer(self, layer):
        if isinstance(layer, Dense) or isinstance(layer, SoftmaxRegression):
            layer.W -= self.lr * layer.g_W
            layer.b -= self.lr * layer.g_b
        if isinstance(layer, ResLayer):
            layer.W1 -= self.lr * layer.g_W1
            layer.W2 -= self.lr * layer.g_W2
            layer.b -= self.lr * layer.g_b
        
    def init_gradient_tracks(self):
        self.g_X = []
        self.g_W = []
        self.g_b = []
        
    def gradient_track(self, layer):
        self.g_X.append(layer.g_X)
        self.g_W.append(layer.g_W) 
        self.g_b.append(layer.g_b)
        return layer.g_X
         
    def backprop(self, C, update=True):
        self.init_gradient_tracks()
        self.model[-1].gradient(self.X_list[-2],self.X_list[-1], C)
        g_X = self.gradient_track(self.model[-1])
           
        if update:
            self.step_layer(self.model[-1])
        
        for i in reversed(range(len(self.model)-1)):
            g_X = self.model[i].gradient(self.X_list[i], g_X)
            _ = self.gradient_track(self.model[i])
            if update:
                self.step_layer(self.model[i])
        
    def __call__(self, X):
        return self.feed_forward(X)
    


