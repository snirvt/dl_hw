import numpy as np

class Softmax:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = self.init_weights(input_dim, output_dim)

    def init_weights(self, prev_layer, next_layer):
        return np.random.randn(prev_layer, next_layer)

    def __call__(self, X):
        linear = X.T @ self.W
        max = np.max(linear, axis=1, keepdims=True)
        exp_linear = np.exp(linear - max)
        result = exp_linear / np.sum(exp_linear, axis=1, keepdims=True)
        return result

    # returns an array with each 1 representing an index with label 1
    def predict(self, output):
        return np.asarray([p[0] for p in self.softmax_predict(output)])

    # arg max for softmax prediction
    def softmax_predict(self, output):
        max_args = np.argmax(output, axis=1).reshape(-1, 1)
        return max_args