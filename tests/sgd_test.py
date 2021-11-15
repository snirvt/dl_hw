from optimizer import SGD
import matplotlib.pyplot as plt
import numpy as np

def sgd_test():
    optimizer = SGD()
    X = np.linspace(0,5, 100)
    y = 2*X + np.random.randn(X.shape[0])
    N_EPOCHS = 20
    W0_new = 0
    W1_new = 0
    loss = np.array([])

    for epoch in range(N_EPOCHS):
        y_pred = np.array([])
        error = np.array([])    # W0 error
        error_x = np.array([])  # W1 error

        W0 = W0_new
        W1 = W1_new

        for x_i in X:
            y_pred = np.append(y_pred, W0 + W1 * x_i)

        error = np.append(error, y_pred - y)
        error_x = np.append(error_x, error * X)
        loss_val = (error ** 2).mean()
        loss = np.append(loss, loss_val)

        W0_new = optimizer.step(sum(error), W0)
        W1_new = optimizer.step(sum(error_x), W1)

    print(f'W0 = {W0_new}')
    print(f'W1 = {W1_new}')
    print(f'Prediction: {y_pred}')
    print(f'Error: {error}')
    print(f'LSE: {loss}')
    fig, axs = plt.subplots(2)
    axs[0].plot(loss)
    axs[0].set_title('Least squares error per iteration')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Loss value')
    axs[1].plot(X, y_pred, color='r')
    axs[1].scatter(X, y)
    axs[1].legend(['Prediction', 'Actual y'])
    axs[1].set_title('Graphs')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.tight_layout()
    plt.savefig('../figures/sgd_test.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == '__main__':
    sgd_test()
