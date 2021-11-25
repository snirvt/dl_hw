    

import numpy as np

class ResLayer():
    def __init__(self, input_dim, output_dim, activation):
        self.W1 = self.init_weights(output_dim, input_dim)
        self.W2 = self.init_weights(input_dim, output_dim)
        self.b = np.random.randn(output_dim, 1)
        self.g_X = None
        self.g_W1 = None
        self.g_W2 = None
        self.g_W = None
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

    
    def jacTMV_x(self, X, v):
        jac_x = self.jac_x(X)
        return jac_x.T @ v



    def jac_x_TMV(self, x, v):
        linear = self.linear_1(x)
        act_deriv = self.activation.deriv(linear)
        W2_diag_W1 = 0
                
        (self.W2 @ (linear * (self.W1 @ v)))
        for i in range(x.shape[1]): 
            diag = np.diag(act_deriv[:,i])
            W2_diag_W1 += self.W2 @ diag @ self.W1
        eye = np.eye(W2_diag_W1.shape[0])
        self.g_X = eye + W2_diag_W1
        return self.g_X


    def gradient(self,X,v):
        self.jacTMV_W1_batch(X, v)
        self.jacTMV_W2_batch(X, v)
        self.jacTMV_b_batch(X, v)
        self.jacTMV_x_batch(X, v)
        self.g_W = (self.g_W1, self.g_W2)
        return self.g_X


    # def jacTMV_x(self, X, v):
    #     linear = self.linear_1(X)
    #     act_deriv = self.activation.deriv(linear)
    #     W2_diag = self.W2 * act_deriv
    #     W2_diag_W1 = W2_diag @ self.W1
    #     W2_diag_I = np.eye(W2_diag_W1.shape[0]) + W2_diag_W1
    #     result = W2_diag_I.T @ v
    #     return result



    def jac_b(self, x):
        linear = self.linear_1(x)
        act_deriv = self.activation.deriv(linear)
        W2_diag = 0
        for i in range(x.shape[1]): 
            diag = np.diag(act_deriv[:,i])
            W2_diag += self.W2 @ diag
        self.g_b =  W2_diag
        return self.g_b

    def jacTMV_b(self, x, v):
        jac_b = self.jac_b(x)
        return jac_b.T @ v
    
    def jacTMV_b_batch(self, x, v):
        linear = self.linear_1(x)
        act_deriv = self.activation.deriv(linear)
        self.g_b = np.sum(act_deriv * (self.W2.T @ v), axis=1).reshape(-1,1)
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


    def jacTMV_W1_batch(self, x, v):
        linear = self.linear_1(x)
        act_deriv = self.activation.deriv(linear)
        self.g_W1 = (act_deriv * (self.W2.T @ v)) @ x.T
        return self.g_W1


    def jacTMV_W2_batch(self, x, v):
        linear = self.linear_1(x)
        act = self.activation(linear)
        self.g_W2 = v @ act.T
        return self.g_W2

    def jacTMV_x_batch(self, x, v):
        linear = self.linear_1(x)
        act_deriv = self.activation.deriv(linear)
        self.g_X = v + (self.W1.T @ (act_deriv * (self.W2.T @ v)))
        return self.g_X
        




    def jacTMV_X_new(self, x, v):
        linear = self.linear_1(x)
        deriv = self.activation.deriv(linear)
        deriv_W1 = deriv * self.W1
        deriv_W1_v = deriv_W1 @ v
        identity = np.eye(np.shape(deriv_W1_v[0]))
        self.g_W1 = identity + deriv_W1_v
        return self.g_W1

    
    def jacTMV_W1_kron_trick(self, x,v): ## worksonly with 1 sample x
        linear = self.linear_1(x)
        act_deriv = self.activation.deriv(linear)
        diag_w2_v = 0
        # w2_diag_kron = 0
        for i in range(x.shape[1]): 
            # kron = np.kron(x[:,i].T, np.eye(x.shape[0]-1))
            diag = np.diag(act_deriv[:,i])
            W2Tv = self.W2.T @ v
            diag_w2_v += diag @ W2Tv
            # w2_diag_kron += self.W2 @ diag @ kron
        
        self.g_W1 = diag_w2_v @ x.T # identity of kron 
        # self.g_W1 = w2_diag_kron
        return self.g_W1
    
    def jacTMV_W1(self, x, v):
        jac_W1 = self.jac_W1(x)
        self.g_W1 = jac_W1.T @ v
        return self.g_W1
    
    
    
    
    def jacTMV_W2(self, x, v):
        linear = self.linear_1(x)
        act = self.activation(linear)
        self.g_W2 = v @ act.T
        # linear = self.linear_1(x)
        # act = self.activation(linear)
        # kron = np.kron(act.T, np.eye(act.shape[0])).T
        # self.g_W2 = kron @ v.T
        return self.g_W2

    
    # def jacTMV_W2(self, x, v):
    #     linear = self.linear_1(x)
    #     act = self.activation(linear)
    #     self.g_W2 = v @ act
    #     # linear = self.linear_1(x)
    #     # act = self.activation(linear)
    #     # kron = np.kron(act.T, np.eye(act.shape[0]))
    #     # self.g_W2 = v @ kron
        #  return self.g_W2
    
    
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
