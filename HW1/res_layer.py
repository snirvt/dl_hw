    

import numpy as np

class ResLayer():
    def __init__(self, input_dim, output_dim, activation):
        self.W1 = self.init_weights(output_dim, input_dim)
        self.W2 = self.init_weights(input_dim, output_dim)
        self.b = np.random.randn(output_dim, 1)
        self.g_X = None
        self.g_W1 = None
        self.g_W2 = None
        self.g_b = None
        self.activation = activation

    def jac_x(self, x):
        linear = self.linear_1(x)
        act_deriv = self.activation.deriv(linear)
        W2_diag_W1 = 0
        for i in range(x.shape[1]): 
            diag = np.diag(act_deriv[:,i])
            W2_diag_W1 += self.W2 @ diag @ self.W1
        eye = np.eye(W2_diag_W1.shape[0])
        self.g_X = eye + W2_diag_W1
        return self.g_X


    def jac_b(self, x):
        linear = self.linear_1(x)
        act_deriv = self.activation.deriv(linear)
        W2_diag = 0
        for i in range(x.shape[1]): 
            diag = np.diag(act_deriv[:,i])
            W2_diag += self.W2 @ diag
        self.g_b =  W2_diag
        return self.g_b


        
    def jacMV_W1(self, x, v):
        linear = self.linear_1(x)
        act_deriv = self.activation.deriv(linear)
        W2_diag_vxT = 0
        
        for i in range(x.shape[1]): 
            # kron = np.kron(x[:,i].T, np.eye(x.shape[0]-1))
            diag = np.diag(act_deriv[:,i])
            vxT = v @ x[:,i:i+1]
            W2_diag_vxT += self.W2 @ diag @ vxT

        self.g_W1 = W2_diag_vxT
        return self.g_W1
    
    
    def jacMV_W2(self, x, v):
        linear = self.linear_1(x)
        act = self.activation(linear)
        self.g_W2 = v @ act
        # linear = self.linear_1(x)
        # act = self.activation(linear)
        # kron = np.kron(act.T, np.eye(act.shape[0]))
        # self.g_W2 = v @ kron
        return self.g_W2


    # initializes the weights
    def init_weights(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim) * np.sqrt(input_dim * output_dim)
    
    def linear_1(self, X):
        linear = self.W1 @ X + self.b
        return linear
    
    def linear_2(self, v):
        linear = self.W2 @ v
        return linear
    
    def feed_forward(self, X):
        Linear_1 = self.linear_1(X)
        linear_1_act = self.activation(Linear_1)
        Linear_2 = self.linear_2(linear_1_act)
        residual = X + Linear_2
        return residual

    def __call__(self, X):
        return self.feed_forward(X)
