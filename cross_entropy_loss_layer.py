import numpy as np

class CrossEntropyLossLayer():
    def __init__(self, W):
        self.W = W
        self.g_W = None

    # saves in g_W the gradient with respect to weights
    def gradient_wrt_weights(self, X, C):
        m = X.shape[-1]                                                 # batchsize
        linear = X.T @ self.W                                           # linear mapping
        max = np.max(linear, axis=1, keepdims=True)                     # normalization
        exp_linear = np.exp(linear - max)                               # applying normalization
        result = exp_linear / np.sum(exp_linear, axis=1, keepdims=True) # softmax
        derivative = np.subtract(result, C)
        self.g_W = 1 / m * (X @ derivative)

    # returns the gradient with respect to data
    def gradient_wrt_data(self, X, C):
        m = X.shape[-1]                                                             # batchsize
        linear = self.W.T @ X                                                       # linear mapping
        exp_linear = np.exp(linear)
        summed_exp_linear = np.array(exp_linear.sum(axis=1))
        rep_mat_res = np.matlib.repmat(summed_exp_linear, exp_linear.shape[0], 1)
        division = np.divide(exp_linear, rep_mat_res)
        derivative = np.subtract(division, C.T)
        return 1/m * self.W @ derivative

    # returns the loss
    def __call__(self, X, C):
        m = X.shape[-1]                                                 # batchsize
        linear = X.T @ self.W                                           # linear mapping
        max = np.max(linear, axis=1, keepdims=True)                     # normalization
        exp_linear = np.exp(linear - max)                               # applying normalization
        result = exp_linear / np.sum(exp_linear, axis=1, keepdims=True) # softmax
        loss = np.sum(C * np.log(result))
        return -1/m * loss

    def predict(self, X):
        linear = X.T @ self.W
        max = np.max(linear, axis=1, keepdims=True)
        exp_linear = np.exp(linear - max)
        result = exp_linear / np.sum(exp_linear, axis=1, keepdims=True)
        return result

    def predict_labels(self, output):
        return np.asarray([p[0] for p in self.softmax_predict(output)])

    # arg max for softmax prediction
    def softmax_predict(self, output):
        max_args = np.argmax(output, axis=1).reshape(-1, 1)
        return max_args

    def step(self, optimizer):
        optimizer.step(self.g_W, self.W)