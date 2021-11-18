

from res_layer import ResLayer
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
    lin1 = ResLayer(3, 2, Tanh())
    fx = lin1(x)[:,0:1]
    u = np.random.randn(3, 1)
    gx = np.dot(fx.T, u)
    no_grad, x_grad = [], []

    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        x_perturbatzia = x + e_normalized_d
        gx_d = np.dot(lin1(x_perturbatzia)[:,0:1].T, u)
        jackMV_x = lin1.jacTMV_x(x, u)
        no_grad.append(np.abs(gx_d - gx).squeeze())
        x_grad.append(np.abs(gx_d - gx - e_normalized_d.T @ jackMV_x).squeeze())
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt X')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
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
        jacTMV_b = lin1.jacTMV_b(x, u)
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
        jacTMV_W1 = lin1.jacTMV_W1_kron_trick(x, u)
        no_grad.append(np.abs(gx_d - gx).squeeze())
        x_grad.append(np.abs(gx_d - gx - e_normalized_d.flatten() @ jacTMV_W1.flatten()).squeeze())

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt bias')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
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
        jacMV_W1 = lin1.jacMV_W1(x, e_normalized_d)
        no_grad.append(np.linalg.norm(fx_d - fx))
        x_grad.append(np.linalg.norm(fx_d - fx - jacMV_W1))
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt Weights')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
# jacobian_test_W1()







def jacobian_test_transpose_W2():
    d = np.random.rand(3, 2)
    x = np.random.rand(3, 1) ## x must be 1 sample, since the gradient should only be for 1 sample (gx[:,0:1], gx_d[:,0:1]) 
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
        jacTMV_W2 = lin1.jacTMV_W2(x, u)
        no_grad.append(np.abs(gx_d - gx).squeeze())
        x_grad.append(np.abs(gx_d - gx - e_normalized_d.reshape(1,-1) @ jacTMV_W2.reshape(-1,1)).squeeze())
    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.title('Jacobian transposed verification wrt Weights')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
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
    #

# jacobian_test_W2()

# if __name__ == '__main__':
#     jacobian_transpose_test_X()
