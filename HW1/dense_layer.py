    
    
    
    
    
import numpy as np

class Dense():
    def __init__(self, input_dim, output_dim, activation):
        self.W = self.init_weights(output_dim, input_dim)
        self.b = np.random.randn(output_dim, 1)
        self.g_X = None
        self.g_W = None
        self.g_b = None
        self.activation = activation

    def jacTMV_x(self, x, v):
        linear = self.linear(x)
        act_deriv = self.activation.deriv(linear)
        act_hadamard = np.multiply(act_deriv, v)
        self.g_X =  self.W.T @ act_hadamard
        return self.g_X


    def jacTMV_b(self, x, v):
        m = x.shape[1]
        # m=1
        linear = self.linear(x)
        act_deriv = self.activation.deriv(linear)
        act_hadamard = np.multiply(act_deriv, v)
        self.g_b =  np.sum((1 / m) * act_hadamard, axis=1, keepdims=True)
        return self.g_b


    def jacTMV_W(self, x, v):
        linear = self.linear(x)
        act_deriv = self.activation.deriv(linear)
        act_hadamard = np.multiply(act_deriv, v)
        self.g_W = act_hadamard @ x.T
        return self.g_W

    # initializes the weights
    def init_weights(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim) * np.sqrt(input_dim * output_dim)
    
    def linear(self, X):
        linear = self.W @ X + self.b
        return linear

    def feed_forward(self, X):
        linear_act = self.activation(self.linear(X))
        return linear_act

    def __call__(self, X):
        return self.feed_forward(X)

