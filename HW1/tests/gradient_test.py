from softmax_regression import SoftmaxRegression
import matplotlib.pyplot as plt
import numpy as np

def gradient_test():
    X = np.random.randn(5, 10)              # 5 - features, 10 - batchsize
    d = np.random.randn(5, 3)               # 5 - features, 3 - labels
    d = d / np.linalg.norm(d)               # d - random vec such that ||d|| = O(1)

    labels = np.random.randint(3, size=10)  # randomly creates 3 labels for 10 samples
    C = np.zeros((labels.size, 3))          # 10 samples, 3 labels (10, 3)
    C[np.arange(labels.size), labels] = 1   # columns in c are one-hot encoded

    loss_func = SoftmaxRegression(5, 3)

    f_x = loss_func.loss(X, C)
    loss_func.gradient(X, C)
    g_x = loss_func.g_W
    W = loss_func.W

    order_1, order_2 = [], []

    epsilons = np.geomspace(0.5, 0.5 ** 20, 20)

    for epsilon in epsilons:
        epsilon_d = epsilon * d
        loss_func.W = W + epsilon_d
        f_eps = loss_func.loss(X, C)
        f_order_1 = abs(f_eps - f_x)
        epsilon_d_grad = np.dot(epsilon_d.ravel().T, g_x.ravel())
        f_order_2 = abs(f_eps - f_x - epsilon_d_grad)
        order_1.append(f_order_1)
        order_2.append(f_order_2)
    l = range(20)
    plt.title('Grad test in semilogarithmic plot')
    plt.plot(l, order_1, label='Zero order approx')
    plt.plot(l, order_2, label='First order approx')
    plt.yscale('log')
    plt.legend()
    plt.savefig('./figures/grad_test.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == '__main__':
    gradient_test()