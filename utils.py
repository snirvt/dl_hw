from sklearn.utils import shuffle
import numpy as np
from configurations import config


# splits the data to X_train, y_train, X_test and y_test
def train_test_split(data):
    X_train = data['Yt']
    y_train = data['Ct']
    X_test = data['Yv']
    y_test = data['Cv']
    return X_train, y_train, X_test, y_test

# shuffles and splits the data to batches
def shuffle_and_batch(X_train, y_train, X_test, y_test):
    n_labels = y_train.shape[0]
    y_train_indices_T = get_indices_transpose(y_train)  # indices of label '1'
    X_train_T = X_train.T
    X_train, y_train = shuffle(X_train_T, y_train_indices_T)
    y_train = encode_one_hot(y_train, n_labels).T  # (2, 20000)
    X_train = X_train.T  # (2, 20000)
    train_loader = split_to_batches(X_train, y_train)
    test_loader = split_to_batches(X_test, y_test)
    return X_train, y_train, train_loader, test_loader

# one hot encoding
def encode_one_hot(labels, n_labels):
    encoding = np.zeros((labels.size, n_labels))
    encoding[np.arange(labels.size), labels] = 1
    return encoding

# returns the indices where label=1 and squeezes it so we get one dimension array
def get_indices_transpose(labels):
    return np.asarray([np.where(label == 1) for label in labels.T]).squeeze()

# returns an array with each 1 representing an index with label 1
def get_index(labels):
    return np.asarray([np.where(l == 1) for l in labels]).squeeze()

# splits the data into batches according to config.BATCH_SIZE
def split_to_batches(X, y):
    return (np.array_split(X, round(X.shape[-1] / config.BATCH_SIZE), axis = 1),
            np.array_split(y, round(y.shape[-1] / config.BATCH_SIZE), axis = 1))