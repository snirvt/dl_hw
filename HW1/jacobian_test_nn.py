
from dense_layer import Dense
from activations import Tanh
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from activations import Tanh
from neural_network import NeuralNetwork
from losses import CrossEntropy

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
        jacMV_x = lin1.jacTMV_x(x, u)[:,0:1]
        no_grad.append(np.abs(fx_d - gx).squeeze())
        x_grad.append(np.abs(fx_d - gx - e_normalized_d.T @ jacMV_x).squeeze())
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
    x = np.random.rand(3, 1)
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
    x = np.random.rand(3, 1)  
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






''' jacobian test NN X'''
def jacobian_test_nn_x():
    d = np.random.rand(2, 1)
    x = np.random.rand(2, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = NeuralNetwork([2,10,10,2], lr=0.1)
    fx = lin1(x)[:,0:1]
    no_grad, x_grad = [], []

    labels = np.random.randint(2, size=1)  
    C = np.zeros((2,labels.size))          
    C[labels,np.arange(labels.size)] = 1   

    loss = CrossEntropy()
    fx_loss = loss(fx, C)

    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        x_perturbatzia = x + e_normalized_d
        
        fx_d = lin1(x_perturbatzia)
        fx_d_loss = loss(fx_d, C)
        lin1.backprop(C, update=False)
        jacMV_x = lin1.g_X[-1]
        no_grad.append(np.abs(fx_d_loss - fx_loss))
        x_grad.append(np.abs(fx_d_loss - fx_loss - e_normalized_d.ravel().T @ jacMV_x.ravel()))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt X')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()


jacobian_test_nn_x()




''' jacobian test NN W'''
def jacobian_test_nn_W():
    d = np.random.rand(2, 2)
    x = np.random.rand(2, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = NeuralNetwork([2,2,10,2,2], lr = 0.1)
    fx = lin1(x)[:,0:1]
    no_grad, x_grad = [], []

    labels = np.random.randint(2, size=1)  
    C = np.zeros((2,labels.size))         
    C[labels,np.arange(labels.size)] = 1   

    loss = CrossEntropy()
    fx_loss = loss(fx, C)
    W = deepcopy(lin1.model[0].W)
    
    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        lin1.model[0].W = W + e_normalized_d
        fx_d = lin1(x)
        fx_d_loss = loss(fx_d, C)
        lin1.model[0].W = deepcopy(W)
        lin1.backprop(C, update=False)
        jacMV_W = lin1.g_W[-1]
        no_grad.append(np.abs(fx_d_loss - fx_loss))
        x_grad.append(np.abs(fx_d_loss - fx_loss - e_normalized_d.ravel().T @ jacMV_W.ravel()))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt X')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()


jacobian_test_nn_W()









''' jacobian test NN b'''
def jacobian_test_nn_b():
    d = np.random.rand(2, 1)
    x = np.random.rand(2, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = NeuralNetwork([2,2,10,2,2], lr=0.1)
    fx = lin1(x)[:,0:1]
    no_grad, x_grad = [], []

    labels = np.random.randint(2, size=1)  
    C = np.zeros((2,labels.size))          
    C[labels,np.arange(labels.size)] = 1  

    loss = CrossEntropy()
    fx_loss = loss(fx, C)
    b = deepcopy(lin1.model[0].b)
    
    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        lin1.model[0].b = b + e_normalized_d
        fx_d = lin1(x)
        fx_d_loss = loss(fx_d, C)
        lin1.model[0].b = deepcopy(b)
        lin1.backprop(C, update=False)
        jacMV_b = lin1.g_b[-1]
        no_grad.append(np.abs(fx_d_loss - fx_loss))
        x_grad.append(np.abs(fx_d_loss - fx_loss - e_normalized_d.ravel().T @ jacMV_b.ravel()))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt b')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()

jacobian_test_nn_b()









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