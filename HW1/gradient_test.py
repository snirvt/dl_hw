from softmax_regression import SoftmaxRegression
from losses import CrossEntropy
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
    
    
def gradient_test_W():
    X = np.random.randn(2, 1)              # 5 - features, 10 - batchsize
    d = np.random.randn(3, 2)               # 5 - features, 3 - labels
    d = d / np.linalg.norm(d)               # d - random vec such that ||d|| = O(1)

    labels = np.random.randint(3, size=1)  # randomly creates 3 labels for 10 samples
    C = np.zeros((3,labels.size))          # 10 samples, 3 labels (10, 3)
    C[labels,np.arange(labels.size)] = 1   # columns in c are one-hot encoded
    loss = CrossEntropy()
    sm = SoftmaxRegression(2, 3)
    f_x = sm(X)
    f_x_loss = loss(f_x, C)
    W = deepcopy(sm.W)
    order_1, order_2 = [], []
    epsilons = np.geomspace(0.5, 0.5 ** 20, 20)

    for epsilon in epsilons:
        epsilon_d = epsilon * d
        # sm.W = W + epsilon_d
        sm.W = W + epsilon_d
        X_pertube = X
        # X_pertube = X + epsilon_d
        f_eps = sm(X_pertube)
        f_eps_loss = loss(f_eps, C)
        f_order_1 = abs(f_eps_loss- f_x_loss)
        sm.W = deepcopy(W)
        sm.gradient(X,f_x, C)
        g_W = sm.g_W

        epsilon_d_grad = np.dot(epsilon_d.ravel().T, g_W.ravel())
        
        f_order_2 = abs(f_eps_loss - f_x_loss - epsilon_d_grad)
        order_1.append(f_order_1)
        order_2.append(f_order_2)
    l = range(20)
    plt.title('Grad test in semilogarithmic plot wrt W')
    plt.plot(l, order_1, label='Zero order approx')
    plt.plot(l, order_2, label='First order approx')
    plt.yscale('log')
    plt.legend()
    # plt.savefig('./figures/grad_test.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
    
gradient_test_W()
    
    
def gradient_test_b():
    X = np.random.randn(2, 1)              # 5 - features, 10 - batchsize
    d = np.random.randn(3, 1)               # 5 - features, 3 - labels
    d = d / np.linalg.norm(d)               # d - random vec such that ||d|| = O(1)

    labels = np.random.randint(3, size=1)  # randomly creates 3 labels for 10 samples
    C = np.zeros((3,labels.size))          # 10 samples, 3 labels (10, 3)
    C[labels,np.arange(labels.size)] = 1   # columns in c are one-hot encoded
    loss = CrossEntropy()
    sm = SoftmaxRegression(2, 3)
    f_x = sm(X)
    f_x_loss = loss(f_x, C)
    b = deepcopy(sm.b)
    order_1, order_2 = [], []
    epsilons = np.geomspace(0.5, 0.5 ** 20, 20)

    for epsilon in epsilons:
        epsilon_d = epsilon * d
        sm.b = b + epsilon_d
        X_pertube = X
        f_eps = sm(X_pertube)
        f_eps_loss = loss(f_eps, C)
        f_order_1 = abs(f_eps_loss- f_x_loss)
        sm.b = deepcopy(b)
        sm.gradient(X,f_x, C)
        g_b = sm.g_b

        epsilon_d_grad = np.dot(epsilon_d.ravel().T, g_b.ravel())
        
        f_order_2 = abs(f_eps_loss - f_x_loss - epsilon_d_grad)
        order_1.append(f_order_1)
        order_2.append(f_order_2)
    l = range(20)
    plt.title('Grad test in semilogarithmic plot wrt b')
    plt.plot(l, order_1, label='Zero order approx')
    plt.plot(l, order_2, label='First order approx')
    plt.yscale('log')
    plt.legend()
    # plt.savefig('./figures/grad_test.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
    
gradient_test_b()        
    
    
    
    
    
def gradient_test_X():
    X = np.random.randn(2, 1)              # 5 - features, 10 - batchsize
    d = np.random.randn(2, 1)               # 5 - features, 3 - labels
    d = d / np.linalg.norm(d)               # d - random vec such that ||d|| = O(1)

    labels = np.random.randint(3, size=1)  # randomly creates 3 labels for 10 samples
    C = np.zeros((3,labels.size))          # 10 samples, 3 labels (10, 3)
    C[labels,np.arange(labels.size)] = 1   # columns in c are one-hot encoded
    loss = CrossEntropy()
    sm = SoftmaxRegression(2, 3)
    f_x = sm(X)
    f_x_loss = loss(f_x, C)
    order_1, order_2 = [], []
    epsilons = np.geomspace(0.5, 0.5 ** 20, 20)

    for epsilon in epsilons:
        epsilon_d = epsilon * d
        X_pertube = X + epsilon_d
        f_eps = sm(X_pertube)
        f_eps_loss = loss(f_eps, C)
        f_order_1 = abs(f_eps_loss- f_x_loss)
        sm.g_X = deepcopy(sm.g_X)
        sm.gradient(X,f_x, C)
        g_X = sm.g_X

        epsilon_d_grad = np.dot(epsilon_d.ravel().T, g_X.ravel())
        
        f_order_2 = abs(f_eps_loss - f_x_loss - epsilon_d_grad)
        order_1.append(f_order_1)
        order_2.append(f_order_2)
    l = range(20)
    plt.title('Grad test in semilogarithmic plot wrt X')
    plt.plot(l, order_1, label='Zero order approx')
    plt.plot(l, order_2, label='First order approx')
    plt.yscale('log')
    plt.legend()
    # plt.savefig('./figures/grad_test.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
    
gradient_test_X()