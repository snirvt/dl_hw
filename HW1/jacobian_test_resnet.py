
from res_layer import ResLayer
from neural_network import NeuralNetwork
from losses import CrossEntropy


from activations import Tanh
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from activations import Tanh

def jacobian_transpose_test_X():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = ResLayer(3, 3, Tanh())
    fx = lin1(x)[:,0:1]
    u = np.random.randn(3, 1)
    gx = np.dot(fx.T, u)
    no_grad, x_grad = [], []

    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        x_perturbatzia = x + e_normalized_d
        gx_d = np.dot(lin1(x_perturbatzia)[:,0:1].T, u)
        jacMV_x =  lin1.jacTMV_x_batch(x,u)
        no_grad.append(np.abs(gx_d - gx).squeeze())
        x_grad.append(np.abs(gx_d - gx - e_normalized_d.T @ jacMV_x).squeeze())
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt X')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    # plt.savefig('./figures/Jacobian_test_resnet_X.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# jacobian_transpose_test_X()

''' jacobian test X'''
def jacobian_test_X():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = ResLayer(3, 2, Tanh()) 
    fx = lin1(x)[:,0:1]
    no_grad, x_grad = [], []

    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        x_perturbatzia = x + e_normalized_d
        fx_d = lin1(x_perturbatzia)[:,0:1]
        jac_x = lin1.jac_x(x)
        no_grad.append(np.linalg.norm(fx_d - fx))
        x_grad.append(np.linalg.norm(fx_d - fx - jac_x @ e_normalized_d))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt X')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()

# jacobian_test_X()




def jacobian_transpose_test_b():
    d = np.random.rand(2, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)
    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = ResLayer(3, 2, Tanh()) 
    fx = lin1(x)[:,0:1]
    u = np.random.randn(3, 1)
    gx = np.dot(fx.T, u)

    no_grad, x_grad = [], []
    org_b = deepcopy(lin1.b)
    for eps in eps_vals:
        lin1.b = deepcopy(org_b)
        e_normalized_d = eps * normalized_d
        lin1.b += e_normalized_d
        gx_d = np.dot(lin1(x)[:,0:1].T, u)
        lin1.b = deepcopy(org_b)
        jacTMV_b =lin1.jacTMV_b_batch(x,u)
        no_grad.append(np.abs(gx_d - gx).squeeze())
        x_grad.append(np.abs(gx_d - gx - e_normalized_d.T @ jacTMV_b).squeeze())

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt bias')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    # plt.savefig('./figures/Jacobian_test_resnet_b.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# jacobian_transpose_test_b()
    

def jacobian_test_b():
    d = np.random.rand(2, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)
    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = ResLayer(3, 2, Tanh()) 
    fx = lin1(x)[:,0:1]
    no_grad, x_grad = [], []
    org_b = deepcopy(lin1.b)
    for eps in eps_vals:
        lin1.b = deepcopy(org_b)
        e_normalized_d = eps * normalized_d
        lin1.b += e_normalized_d
        fx_d = lin1(x)[:,0:1]
        lin1.b = deepcopy(org_b)
        jac_b = lin1.jac_b(x)
        no_grad.append(np.linalg.norm(fx_d - fx))
        x_grad.append(np.linalg.norm(fx_d - fx - jac_b @ e_normalized_d))

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


def jacobian_transpose_test_W1():
    d = np.random.rand(2, 3)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)
    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = ResLayer(3, 2, Tanh()) 
    fx = lin1(x)[:,0:1]
    u = np.random.randn(3, 1)
    gx = np.dot(fx.T, u)

    no_grad, x_grad = [], []
    org_W1 = deepcopy(lin1.W1)
    for eps in eps_vals:
        lin1.W1 = deepcopy(org_W1)
        e_normalized_d = eps * normalized_d
        lin1.W1 += e_normalized_d
        gx_d = np.dot(lin1(x)[:,0:1].T, u)
        lin1.W1 = deepcopy(org_W1)
        jacTMV_W1 = lin1.jacTMV_W1_batch(x, u)
        no_grad.append(np.abs(gx_d - gx).squeeze())
        x_grad.append(np.abs(gx_d - gx - e_normalized_d.flatten() @ jacTMV_W1.flatten()).squeeze())

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt W1')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    # plt.savefig('./figures/Jacobian_test_resnet_W1.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# jacobian_transpose_test_W1()




def jacobian_test_W1():
    d = np.random.rand(2, 3)
    x = np.random.rand(3, 1) ## x must be 1 sample, since the gradient should only be for 1 sample (gx[:,0:1], gx_d[:,0:1]) 
    normalized_d = d / np.linalg.norm(d)
    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = ResLayer(3, 2, Tanh()) 
    fx = lin1(x)[:,0:1]
    no_grad, x_grad = [], []
    org_W = deepcopy(lin1.W1)
    for eps in eps_vals:
        lin1.W1 = deepcopy(org_W)
        e_normalized_d = eps * normalized_d
        lin1.W1 += e_normalized_d
        fx_d = lin1(x)[:,0:1]
        lin1.W1 = deepcopy(org_W)
        # jacMV_W1 = lin1.jacMV_W1(x, e_normalized_d)
        jacMV_W1 = lin1.jacTMV_W1_batch(x, e_normalized_d)
        no_grad.append(np.linalg.norm(fx_d - fx))
        x_grad.append(np.linalg.norm(fx_d - fx - jacMV_W1))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt W1')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    
# jacobian_test_W1()







def jacobian_test_transpose_W2():
    d = np.random.rand(3, 2)
    x = np.random.rand(3, 1) 
    normalized_d = d / np.linalg.norm(d)
    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = ResLayer(3, 2, Tanh()) 
    fx = lin1(x)[:,0:1]
    u = np.random.randn(3, 1)
    gx = np.dot(fx.T, u)
    no_grad, x_grad = [], []
    org_W = deepcopy(lin1.W2)
    for eps in eps_vals:
        lin1.W2 = deepcopy(org_W)
        e_normalized_d = eps * normalized_d
        lin1.W2 += e_normalized_d
        gx_d = np.dot(lin1(x)[:,0:1].T, u)
        lin1.W2 = deepcopy(org_W)
        jacTMV_W2 = lin1.jacTMV_W2_batch(x, u)
        no_grad.append(np.abs(gx_d - gx).squeeze())
        x_grad.append(np.abs(gx_d - gx - e_normalized_d.reshape(1,-1) @ jacTMV_W2.reshape(-1,1)).squeeze())
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt W2')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    # plt.savefig('./figures/Jacobian_test_resnet_W2.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# jacobian_test_transpose_W2()

def jacobian_test_W2():
    d = np.random.rand(3, 2)
    x = np.random.rand(3, 1) ## x must be 1 sample, since the gradient should only be for 1 sample (gx[:,0:1], gx_d[:,0:1])
    normalized_d = d / np.linalg.norm(d)
    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = ResLayer(3, 2, Tanh())
    fx = lin1(x)
    no_grad, x_grad = [], []
    org_W = deepcopy(lin1.W2)
    for eps in eps_vals:
        lin1.W2 = deepcopy(org_W)
        e_normalized_d = eps * normalized_d
        lin1.W2 += e_normalized_d
        fx_d = lin1(x)
        lin1.W2 = deepcopy(org_W)
        jacMV_W2 = lin1.jacMV_W2(x, e_normalized_d)
        no_grad.append(np.linalg.norm(fx_d - fx))
        x_grad.append(np.linalg.norm(fx_d - fx - jacMV_W2))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt Weights')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    
    
    
    

def jacobian_transpose_test_nn_X():
    d = np.random.rand(2, 1)
    x = np.random.rand(2, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = NeuralNetwork([2,10,10,2], lr=0.1, type=ResLayer)

    fx = lin1(x)[:,0:1]
    u = np.random.randn(2, 1)
    gx = np.dot(fx.T, u)
    no_grad, x_grad = [], []
    
    
    labels = np.random.randint(2, size=1)  
    C = np.zeros((2,labels.size))          
    C[labels,np.arange(labels.size)] = 1  
    
    loss = CrossEntropy()
    fx_loss = loss(fx, C)

    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        x_perturbatzia = x + e_normalized_d
        # gx_d = np.dot(lin1(x_perturbatzia)[:,0:1].T, u)
        fx_d = lin1(x_perturbatzia)
        # jacMV_x =  lin1.jacTMV_x_batch(x,u)
        fx_d_loss = loss(fx_d, C)
        lin1.backprop(C, update=False)
        jacMV_x = lin1.g_X[-1]
        no_grad.append(np.abs(fx_d_loss - fx_loss))
        x_grad.append(np.abs(fx_d_loss - fx_loss - e_normalized_d.ravel().T @ jacMV_x.ravel()))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed neural network verification wrt X')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    # plt.savefig('./figures/Jacobian_test_resnet_X.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# jacobian_transpose_test_nn_X()

    
    
    
    
''' jacobian test NN W1'''
def jacobian_test_nn_W1():
    d = np.random.rand(10, 2)
    x = np.random.rand(2, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = NeuralNetwork([2,10,10,2], lr=0.1, type=ResLayer)
    fx = lin1(x)[:,0:1]
    no_grad, x_grad = [], []

    labels = np.random.randint(2, size=1)  
    C = np.zeros((2,labels.size))         
    C[labels,np.arange(labels.size)] = 1   

    loss = CrossEntropy()
    fx_loss = loss(fx, C)
    W1 = deepcopy(lin1.model[0].W1)
    
    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        lin1.model[0].W1 = W1 + e_normalized_d
        fx_d = lin1(x)
        fx_d_loss = loss(fx_d, C)
        lin1.model[0].W1 = deepcopy(W1)
        lin1.backprop(C, update=False)
        jacMV_W = lin1.g_W[-1][0]
        no_grad.append(np.abs(fx_d_loss - fx_loss))
        x_grad.append(np.abs(fx_d_loss - fx_loss - e_normalized_d.ravel().T @ jacMV_W.ravel()))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed neural network verification wrt W1')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
# jacobian_test_nn_W1()    
    
    
    
    
    
''' jacobian test NN W2'''
def jacobian_test_nn_W2():
    d = np.random.rand(2, 10)
    x = np.random.rand(2, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = NeuralNetwork([2,10,10,2], lr=0.1, type=ResLayer)
    fx = lin1(x)[:,0:1]
    no_grad, x_grad = [], []

    labels = np.random.randint(2, size=1)  
    C = np.zeros((2,labels.size))         
    C[labels,np.arange(labels.size)] = 1   

    loss = CrossEntropy()
    fx_loss = loss(fx, C)
    W2 = deepcopy(lin1.model[0].W2)
    
    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        lin1.model[0].W2 = W2 + e_normalized_d
        fx_d = lin1(x)
        fx_d_loss = loss(fx_d, C)
        lin1.model[0].W2 = deepcopy(W2)
        lin1.backprop(C, update=False)
        jacMV_W = lin1.g_W[-1][1]
        no_grad.append(np.abs(fx_d_loss - fx_loss))
        x_grad.append(np.abs(fx_d_loss - fx_loss - e_normalized_d.ravel().T @ jacMV_W.ravel()))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed neural network verification wrt W2')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
# jacobian_test_nn_W2()    
      
    
    
    
''' jacobian test NN b'''
def jacobian_test_nn_b():
    d = np.random.rand(10, 1)
    x = np.random.rand(2, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = NeuralNetwork([2,10,10,2], lr=0.1, type=ResLayer)
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
        jacMV_W = lin1.g_b[-1]
        no_grad.append(np.abs(fx_d_loss - fx_loss))
        x_grad.append(np.abs(fx_d_loss - fx_loss - e_normalized_d.ravel().T @ jacMV_W.ravel()))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed neural network verification wrt b')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
jacobian_test_nn_b()    
      