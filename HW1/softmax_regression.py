import numpy as np

class SoftmaxRegression():
    def __init__(self, input_dim, output_dim):
        self.W = self.init_weights(input_dim, output_dim)
        self.g_W = None
        self.g_X = None

    # initializes the weights
    def init_weights(self, input_dim, output_dim):
        return np.random.randn(input_dim, output_dim)

    # updates the gradient wrt to weights and data
    def gradient(self, X, C):
        m = X.shape[-1]  # batchsize
        linear = X.T @ self.W  # linear mapping
        max = np.max(linear, axis=1, keepdims=True)  # normalization
        exp_linear = np.exp(linear - max)  # applying normalization
        result = exp_linear / np.sum(exp_linear, axis=1, keepdims=True)  # softmax
        derivative = np.subtract(result, C)
        self.g_W = 1 / m * (X @ derivative)
        self.g_X = 1 / m * (self.W @ derivative.T)

    # returns the loss
    def loss(self, X, C):
        m = X.shape[-1]  # batchsize
        linear = X.T @ self.W  # linear mapping
        max = np.max(linear, axis=1, keepdims=True)  # normalization
        exp_linear = np.exp(linear - max)  # applying normalization
        result = exp_linear / np.sum(exp_linear, axis=1, keepdims=True)  # softmax
        loss = np.sum(C * np.log(result))
        return -1 / m * loss

    # predics the labels themselves
    def predict_labels(self, output):
        return np.asarray([p[0] for p in self.softmax_predict(output)])

    # arg max for softmax prediction
    def softmax_predict(self, output):
        max_args = np.argmax(output, axis=1).reshape(-1, 1)
        return max_args

    def step(self, optimizer):
        optimizer.step(self.g_W, self.W)

    # softmax
    def __call__(self, X):
        linear = X.T @ self.W
        max = np.max(linear, axis=1, keepdims=True)
        exp_linear = np.exp(linear - max)
        result = exp_linear / np.sum(exp_linear, axis=1, keepdims=True)
        return result