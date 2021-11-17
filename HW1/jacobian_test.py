
from dense_layer import Dense
from activations import Tanh
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from activations import Tanh


''' jacobian test X'''
def jacobian_test_x():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 10)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = Dense(3, 2, Tanh()) 
    fx = lin1(x)[:,0:1]
    u = np.random.randn(2, 1)
    gx = np.dot(fx.T, u)
    no_grad, x_grad = [], []

    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        x_perturbatzia = x + e_normalized_d
        fx_d = np.dot(lin1(x_perturbatzia)[:,0:1].T, u)
        jackMV_x = lin1.jacTMV_x(x, u)[:,0:1]
        no_grad.append(np.abs(fx_d - gx).squeeze())
        x_grad.append(np.abs(fx_d - gx - e_normalized_d.T @ jackMV_x).squeeze())
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt X')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()


# jacobian_test_x()



''' jacobian test b'''
def jacobian_test_b():
    d = np.random.rand(2, 1)
    x = np.random.rand(3, 10)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = Dense(3, 2, Tanh()) 
    fx = lin1(x)[:,0:1]
    u = np.random.randn(2, 1)
    gx = np.dot(fx.T, u)
    no_grad, x_grad = [], []
    org_b = deepcopy(lin1.b)
    for eps in eps_vals:
        lin1.b = deepcopy(org_b)
        e_normalized_d = eps * normalized_d
        # x_perturbatzia = x + e_normalized_d
        lin1.b += e_normalized_d
        fx_d = np.dot(lin1(x)[:,0:1].T, u)
        lin1.b = deepcopy(org_b)
        jacMV_b = lin1.jacTMV_b(x, u)[:,0:1]
        no_grad.append(np.abs(fx_d - gx).squeeze())
        x_grad.append(np.abs(fx_d - gx - e_normalized_d.T @ jacMV_b).squeeze())
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt bias')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    
# jacobian_test_b()


def jacobian_test_W():
    d = np.random.rand(2, 3)
    x = np.random.rand(3, 1) ## x must be 1 sample, since the gradient should only be for 1 sample (gx[:,0:1], gx_d[:,0:1]) 
    normalized_d = d / np.linalg.norm(d)
    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = Dense(3, 2, Tanh()) 
    fx = lin1(x)[:,0:1]
    u = np.random.randn(2, 1)
    gx = np.dot(fx.T, u)
    no_grad, x_grad = [], []
    org_W = deepcopy(lin1.W)
    for eps in eps_vals:
        lin1.W = deepcopy(org_W)
        e_normalized_d = eps * normalized_d
        lin1.W += e_normalized_d
        gx_d = np.dot(lin1(x)[:,0:1].T, u)
        lin1.W = deepcopy(org_W)
        jacMV_W = lin1.jacTMV_W(x, u)
        no_grad.append(np.abs(gx_d - gx).squeeze())
        x_grad.append(np.abs(gx_d - gx - e_normalized_d.reshape(1,-1) @ jacMV_W.reshape(-1,1)).squeeze())
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt Weights')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    
jacobian_test_x()
jacobian_test_b()
jacobian_test_W()



# import numpy as np

# W = np.array([[1,1,1],[1,1,1]])
# W2 = np.array([1,1,1]).reshape(-1,1)

# X = np.array([[-0.3,-0.3],[0.3,0.3],[0.3,0.3], [-0.1,-0.1]]).T

# v = np.array([0.5,0.7,0.8]).reshape(-1,1)


# W.shape
# W2.shape
# X.shape

# X @ W

# linear = W.T @ X

# (linear).shape

# np.tanh(linear)

# np.multiply(np.tanh(linear), v)

# np.multiply(np.tanh(linear), v).shape

# np.multiply(np.tanh(linear), v) @ X.T

# (np.multiply(np.tanh(linear), v) @ X.T).shape
# W.shape

# W @ (np.multiply(np.tanh(linear), v))

# (W @ (np.multiply(np.tanh(linear), v))).shape

# W.shape

# W2.T @ np.tanh(linear)







(np.tanh(linear) @ X).shape

np.diag(np.tanh(linear))



