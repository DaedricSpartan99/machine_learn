import numpy as np
import random as rnd


"""
    Utils
"""

# define threshold constant
eps = 1e-5

"""
    y = samples vector
    x_n = arguments samples matrix

    grad_n: function pointer taking:
            - a scalar y_n
            - a vector of size Nx x_n
            - a vector of size Nx w_n

    initial_w = initial value of w (of size Nx)

    max_iters = maximum number of iterations
    gamma = step factor in GD

"""
def mini_batch_SGD(y, tx, grad_n, initial_w, max_iters, gamma):

    N = len(y) # length of samples
    Nx = len(tx[0]) # length of arguments
    w = initial_w
    old_w = initial_w - np.ones(Nx)

    # iterate
    while max_iters > 0 and (np.norm(w - old_w) / Nx) > eps: # TODO add threshold condition

        # generate random integer to consider
        NB = rnd.randint(1,N)
        B = random.sample(range(N), NB)

        # compute stochastic gradient
        g = np.zeros(Nx)
        for n in B:
            g += grad_n(grad_n(y[n], tx[n], w))
        g /= NB

        old_w = w

        # forward step w
        w -= gamma * g

        # decrement step
        max_iters -= 1

    return w

"""
    Compute the cost given a generic cost function.
    - y: predictions vector
    - xt: argument samples matrix
    - w: eights
    - fw: analytic expression for weight function.
                It should take the xt, w as parameters.
    - cost_fct: analytic expression of the cost function.
                It should take the errors vector containing ( y_n )
        
"""
def compute_cost(y, xt, w, fw, cost_fct):

    # Errors evaluation
    errors = y - fw(xt, w)

    # Compute the cost
    return np.mean(cost_fct(errors))






"""
    Particular case of RMSE implementations
"""

def MSE_fw(xt, w):

    # X^t * w
    #return np.transpose(xt) * w
    return np.dot(np.transpose(xt), w)

def MSE_cost_fct(errors):

    # euclidean_norm(errors) / 2
    return np.dot(errors, errors) / 2

# compute cost for RMSE particular case
def MSE_cost(y, xt, w):
    return compute_cost(y, xt, w, MSE_fw, MSE_cost_fct)





"""
    Methods implementations
"""

def ridge_regression(y, tx, lambda_):

    # Compute optimal weights
    T = np.dot(np.transpose(tx), tx) # dim(T) = M * M
    N = len(tx) # how many rows, TODO check
    M = len(T[0]) # how many columns

    # add lambda_ contribution, otherwise linear regression
    if lambda_ != 0:
        T += lambda_ * np.identity(M) / (2 * N)
        
    xy = np.dot(np.transpose(tx), y)

    # compute result following the formula: w * T = X^t * y
    w = np.linalg.solve(T, xy)

    cost_fct = lambda y, tx, w: MSE_cost(y, tx, w) - lambda_ * np.dot(w,w)
    
    return w, compute_cost(y, tx, w, MSE_fw, cost_fct) 


def least_squares(y, tx):
    
    # Compute weight as particular case of ridge regression, _lambda = 0
    return ridge_regression(y, tx, 0)

def logistic_sigmoid(z):
    arg = np.exp(z) 
    return arg / (1 + arg)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
     """
        Use the Logistic Regression method to find the best weights
        
        INPUT:
        |    y           | Predictions                   |
        |    tx          | Samples                       |
        |    initial_w   | Initial weights
        |    max_iters   | Maximum number of iterations
        |    gamma       | Step size
            
        OUTPUT:
            w           - Best weights
            loss        - Minimum loss
    """

    # gradient L_n formula: x_n * (sigma(x_n * w) - y_n)
    tx_t=tx.T
    grad=tx_t.dot( sigmoid (tx.dot(w))-y)
    
    # loss function
    loss = np.sum(np.log(1. + np.exp(np.dot(tx, w)))) - np.dot(y.T, np.dot(tx, w))

    # weight
    w = w - gamma * grad

    return w, loss

