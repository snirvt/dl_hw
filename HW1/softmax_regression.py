import numpy as np
import numpy.matlib as npm
class SoftmaxRegression():
    def __init__(self, input_dim, output_dim):
        self.W = self.init_weights(output_dim,input_dim)
        self.b = self.init_weights(output_dim, 1)
        self.g_W = 0
        self.g_b = 0
        self.g_X = 0

    # initializes the weights
    def init_weights(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim)/(np.sqrt(input_dim*output_dim))
    
    # updates the gradient wrt to weights and data
    def gradient(self, X, y_hat, C):
        m = C.shape[-1]  # batchsize
        derivative = np.subtract(y_hat, C)
        self.g_W = (1 / m) * (X @ derivative.T).T
        self.g_b  = np.sum((1 / m) * derivative, axis=1, keepdims=True)
        self.g_X = (1 / m) * (self.W.T @ derivative)
        
        
    def predict_labels(self, output):
        return np.asarray([p[0] for p in self.softmax_predict(output)])

    def softmax_predict(self, output):
        max_args = np.argmax(output, axis=1).reshape(-1, 1)
        return max_args
        
    def feed_forward(self, X):
        linear = self.W @ X + self.b
        max = np.max(linear, axis=0, keepdims=True)
        exp_linear = np.exp(linear - max)
        result = exp_linear / np.sum(exp_linear, axis=0, keepdims=True)
        return result
    
    def __call__(self, X):
        return self.feed_forward(X)
    
