import numpy as np
import random as rnd


"""
    Utils
"""

# define threshold constant
eps = 1e-10

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

# COMPUTE LOSS ET GRADIENT POUR least square GD

def compute_loss(y, tx, w):    #using MSE, give L not L_n
        e = y-tx.dot(w)
        cost=(e.dot(e.T))/(2*y.shape[0])  #I think we can remove the .T in e.T
        return cost

def compute_gradient(y, tx, w):
    e = y-tx.dot(w)
    tx_t=tx.T
    return -tx_t.dot(e)/len(y)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    #compute gradient and loss
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx, w)
        loss=compute_loss(y,tx,w)
        w = w-gamma*gradient    #update w by the gradient
         # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
        w_opt=ws[-1]
        loss_opt=losses[-1]
    return w_opt,loss_opt



"""
    Pure mini-batch SGD algorithm
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
    - cost_fct_n: analytic expression of the cost function per element L_n.
"""
def compute_cost(y, xt, w, cost_fct_n):

    # Errors evaluation
    #errors = y - fw(xt, w)

    # Compute the cost
    return np.mean(cost_fct_n(y, xt, w))






"""
    Particular case of RMSE implementations
"""

def MSE_fw(xt, w):

    # X * w
    return np.dot(xt, w)

# elementwise cost function
def MSE_cost_fct(y, xt, w):

    errors = y - MSE_fw(xt, w)

    # euclidean_norm(errors) / 2
    return np.power(errors,2) / 2

# compute cost for RMSE particular case
def MSE_cost(y, xt, w):
    return compute_cost(y, xt, w, MSE_cost_fct)





"""
    Methods implementations
"""

def ridge_regression(y, tx, lambda_):

    # Compute optimal weights
    N = len(tx) # how many rows, TODO check
    M = len(tx[0]) # how many columns
    T = np.dot(np.transpose(tx), tx)
    xy = np.dot(np.transpose(tx), y)

    # add lambda_ contribution, otherwise linear regression
    if np.abs(lambda_) > eps:   # we consider any lammda < eps to be equal to 0
        T += lambda_ * (2*N) * np.identity(M) # dim(T) = M * M

    w = np.linalg.solve(T, xy) # compute result following the formula: w * T = X^t * y
        
    #THIS PART STILL NEEDS TO BE CHECKED
    cost_fct = lambda y, tx, w: MSE_cost(y, tx, w) - lambda_ * np.dot(np.transpose(w),w)
    
    return w, compute_cost(y, tx, w, cost_fct) 



# logistic regression using mini-batch SGD
def logistic_regression_mb(y, tx, initial_w, max_iters, gamma):

    # gradient L_n formula: x_n * (sigma(x_n * w) - y_n)
    grad_n = lambda yn, txn, w: txn * (logistic_sigmoid(np.dot(ntx, w)) - yn)

    # compute optimal weight
    w = mini_batch_SDG(y, tx, grad_n, initial_w, max_iters, gamma)

    return w, MSE_cost(y, tx, w)



def least_squares(y, tx):
    
    # Compute weight as particular case of ridge regression, _lambda = 0
    return ridge_regression(y, tx, 0)

def logistic_sigmoid(z):
    arg = np.exp(z) 
    return arg / (1 + arg)
