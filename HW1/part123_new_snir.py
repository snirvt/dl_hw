from os import access
from utils import shuffle_and_batch, train_test_split, get_index
# from cross_entropy_loss_layer import CrossEntropyLossLayer
from softmax_regression import SoftmaxRegression
from data.dataloader import DataLoader
from itertools import product
from optimizer import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from configurations import config


dataset_name = config.DATASETS_NAMES
dataset_name = 'SwissRollData'
dataset_name =  'PeaksData'

data = DataLoader.load_dataset(dataset_name)
X_train, y_train, X_test, y_test = train_test_split(data)



# X_train = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 5000).T
# X_train = np.concatenate((X_train, np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 5000).T), axis=1)
# X_test = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 5000).T
# X_test = np.concatenate((X_test, np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 5000).T), axis=1)

# y_train = np.zeros((2, 10000)).astype(np.float32)
# y_train[0,:5000] = 1
# y_train[1,5000:] = 1
# y_test = np.zeros((2, 10000))
# y_test[0,:5000] = 1
# y_test[1,5000:] = 1

X_train = np.concatenate((X_train, np.ones((1,X_train.shape[1]))), axis=0)
X_test = np.concatenate((X_test, np.ones((1,X_test.shape[1]))), axis=0)

input_dim = X_train.shape[0]
output_dim = y_train.shape[0]
# W = np.random.randn(input_dim, output_dim)
net = SoftmaxRegression(input_dim, output_dim)
optimizer = SGD(lr=0.001)
config.BATCH_SIZE = 500



# n_epochs = config.NUM_EPOCHS
n_epochs = 100
val_loss = []
val_accuracy = []
train_accuracy = []
train_loss = []
Xtrain, Ytrain, train_loader, val_loader = shuffle_and_batch(X_train, y_train, X_test, y_test)                                                                                                          

for epoch in range(n_epochs):
    train_batches, train_labels = train_loader
    val_batches, val_labels = val_loader
    n_correct = 0
    n_cnt = 0
    batch_loss = []
    if epoch % 10 == 0:
        optimizer.lr *= 0.995
        
    # for batch, labels in tqdm(zip(train_batches, train_labels),total=len(train_batches)):                    
    for batch, labels in zip(train_batches, train_labels):                    
        labels = labels.T
        batch_loss.append(net.loss(batch, labels))
        
        net.gradient(batch, labels)
        W_old, grad = net.W, net.g_W
        net.W = optimizer.step(grad, W_old)

        # calc accuracy
        labels = get_index(labels)
        prob_prediction = net(batch)
        label_prediction = net.predict_labels(prob_prediction)
        n_correct += sum(label_prediction == labels)
        n_cnt += len(label_prediction)
    train_loss.append(np.sum(batch_loss))
    train_accuracy.append(n_correct/n_cnt)
    
    n_correct = 0
    n_cnt = 0
    batch_loss = []
    for batch, labels in zip(val_batches, val_labels):
        labels = labels.T
        batch_loss.append(net.loss(batch, labels))
        # calc accuracy
        labels = get_index(labels)
        prob_prediction = net(batch)
        label_prediction = net.predict_labels(prob_prediction)
        n_correct += sum(label_prediction == labels)
        n_cnt += len(label_prediction)
    val_loss.append(np.sum(batch_loss))
    val_accuracy.append(n_correct/n_cnt)
    print('epoch: {}'.format(epoch))
    print('train loss: {}, val loss: {}'.format(train_loss[-1], val_loss[-1]))
    print('train accuracy: {}, val accuracy: {}'.format(train_accuracy[-1], val_accuracy[-1]))
        # train_accuracy = np.append(train_accuracy, label_prediction == labels, axis=0)
        
    
plt.figure(1)
plt.subplot(211)
plt.plot(range(len(train_loss)), train_loss, label='loss')
plt.plot(range(len(val_loss)), val_loss, label='val loss')
plt.legend()
plt.subplot(212)
plt.plot(range(len(train_accuracy)), train_accuracy, label='accuracy')
plt.plot(range(len(val_accuracy)), val_accuracy, label='val accuracy')
plt.legend()
plt.show() 




def run_tests(dataset_hidden_layers=None,
              learning_rates = [0.1, 0.01, 0.001],
              batch_sizes=[32, 64, 128],
              optimizer=None,
              epochs=50):
    len_learning_rates = len(learning_rates)
    len_batch_sizes = len(batch_sizes)
    for dataset_name in config.DATASETS_NAMES:
        if dataset_hidden_layers == None:
            hidden_layers = []
        else:
            hidden_layers = dataset_hidden_layers[dataset_name]
        data = DataLoader.load_dataset(dataset_name)
        X_train, y_train, X_test, y_test = train_test_split(data)
        Y = np.argmax(y_train, axis=0)
        sets = {'train': (X_train, y_train), 'test': (X_test, y_test)}

        sns.scatterplot(x=X_train[0],
                        y=X_train[1],
                        hue=Y,
                        linewidth=0.1,
                        palette='pastel',).set_title(
            f'2d visualization of {dataset_name}'
        )

        fig = plt.figure(figsize=(len_batch_sizes * 3,
                                  len_learning_rates * 3))
        fig.suptitle(f'Accuracy as a function of epoch'
                     f'\nDataset: {dataset_name}', fontsize=20)
        for i, (lr, batch_size) in tqdm(enumerate(list(product(learning_rates,
                                                               batch_sizes))),
                                        desc=dataset_name,
                                        total=len_batch_sizes * len_learning_rates):
            input_dim = X_train.shape[0]
            output_dim = y_train.shape[0]
            # W = np.random.randn(input_dim, output_dim)
            net = SoftmaxRegression(input_dim, output_dim)
            optimizer = SGD(lr=lr)
            val_epoch_accuracy, train_epoch_accuracy = train(net, optimizer, X_train, y_train, X_test, y_test)
            ax = fig.add_subplot(len_learning_rates, len_batch_sizes, i + 1)
            ax.set_title(f'lr: {lr}, batch size: {batch_size}')
            ax.set_ylim([0, 1])
            ax.plot(range(config.NUM_EPOCHS), val_epoch_accuracy, label=dataset_name)
            ax.plot(range(config.NUM_EPOCHS), train_epoch_accuracy, label=dataset_name)
            ax.legend(['Train', 'Validation'])
        plt.savefig(f'./figures/{dataset_name}_crossentropy.png', transparent=True, bbox_inches='tight', pad_inches=1)
        plt.show()

def train(net, optimizer, X_train, y_train, X_test, y_test):
    Xtrain, Ytrain, train_loader, val_loader = shuffle_and_batch(X_train,
                                                                  y_train,
                                                                  X_test,
                                                                  y_test)
    n_epochs = config.NUM_EPOCHS
    train_batches, train_labels = train_loader
    val_batches, val_labels = val_loader
    train_epoch_accuracy = []
    val_epoch_accuracy = []
    for epoch in range(n_epochs):
        train_accuracy = []
        train_loss = []
        for batch, labels in tqdm(zip(train_batches, train_labels),
                                  total=len(train_batches)):
            labels = labels.T
            batch_loss = net.loss(batch, labels)
            train_loss.append(batch_loss)
            net.gradient(batch, labels)
            W_old, grad = net.W, net.g_W
            net.W = optimizer.step(grad, W_old)

            # calc accuracy
            labels = get_index(labels)
            prob_prediction = net(batch)
            label_prediction = net.predict_labels(prob_prediction)
            train_accuracy = np.append(train_accuracy, label_prediction == labels, axis=0)

        print(f'Epoch #{epoch} train acc: {np.mean(train_accuracy)}')

        train_epoch_accuracy = np.append(train_epoch_accuracy, np.mean(train_accuracy))

        # validation
        val_accuracy = []
        for v_batch, v_labels in tqdm(zip(val_batches, val_labels),
                                  total=len(val_batches)):
            v_labels = v_labels.T
            val_probs_prediction = net(v_batch)
            val_label_prediction = net.predict_labels(val_probs_prediction)
            v_labels = get_index(v_labels)
            val_accuracy = np.append(val_accuracy, val_label_prediction == v_labels, axis=0)

        print(f'Epoch #{epoch} val acc: {np.mean(val_accuracy)}')
        val_epoch_accuracy = np.append(val_epoch_accuracy, np.mean(val_accuracy))

    return val_epoch_accuracy, train_epoch_accuracy




if __name__ == '__main__':
    HIDDEN_LAYERS = {
        "SwissRollData": [0],
        "PeaksData": [0],
        "GMMData": [0]
    }

    run_tests(optimizer='SGD')
