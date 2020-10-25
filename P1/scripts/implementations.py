import numpy as np
import random as rnd
from proj1_helpers import *


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

"""
# COMPUTE LOSS & GRADIENT for least square GD

def compute_loss(y, tx, w):    #using MSE, give L not L_n
        e = y-tx.dot(w)
        cost=(e.dot(e.T))/(2*y.shape[0])  #I think we can remove the .T in e.T
        return cost

def compute_gradient(y, tx, w):
    e = y-tx.dot(w)
    tx_t=tx.T
    return -tx_t.dot(e)/len(y)
"""

### FUNCTION 1 ###


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
        #take the final ones
    w_opt=ws[-1]
    loss_opt=losses[-1]
    return w_opt,loss_opt

"""
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
"""

### FUNCTION 2 ###

def least_squares_SGD(y, tx, initial_w, batch_size = 1, max_iters, gamma):
    if (batch_size>len(y)):
        print("The batch size was bigger than the whole dataset, it was downsized to match that of the dataset")
        batch_size=len(y)
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size): 
            gradient = compute_gradient(minibatch_y,minibatch_tx,w) # Can compute the normal gradient for each minibatch
            loss=compute_loss(minibatch_y,minibatch_tx,w)
        # TODO: update w by gradient
        w = w-gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    w_opt=ws[-1]
    loss_opt=losses[-1]

    return w_opt,loss_opt


# I don't think we even need this function anymore, plus we deleted grad_n
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


"""
def compute_cost(y, xt, w, fw, cost_fct):

    # Errors evaluation
    errors = y - fw(xt, w)

    # Compute the cost
    return np.mean(cost_fct(errors))






"""
   # Particular case of RMSE implementations
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


"""
    Methods implementations
"""

### FUNCTION 4 ###

def ridge_regression(y, tx, lambda_):

    # Compute optimal weights
    T = np.dot(np.transpose(tx), tx) # dim(T) = M * M
    N = len(tx) # how many rows, TODO check
    M = len(T[0]) # how many columns

    # add lambda_ contribution, otherwise linear regression
    if np.abs(lambda_) < eps:   # we consider any lammda < eps to be equal to 0
        w = np.linalg.solve(tx,y)   
    else:
        T += lambda_ * (2*N) * np.identity(M)
        xy = np.dot(np.transpose(tx), y)
        w = np.linalg.solve(T, xy) # compute result following the formula: w * T = X^t * y

        
    #THIS PART STILL NEEDS TO BE CHECKED
    cost_fct = lambda y, tx, w: MSE_cost(y, tx, w) - lambda_ * np.dot(w,w)
    
    return w, compute_cost(y, tx, w, MSE_fw, cost_fct) 


### FUNCTION 3 ###

def least_squares(y, tx):
    
    # Compute weight as particular case of ridge regression, _lambda = 0
    return ridge_regression(y, tx, 0)

"""
def logistic_sigmoid(z):
    arg = np.exp(z) 
    return arg / (1 + arg)
"""

### FUNCTION 5 ###

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
     """
        Implementation of the Logistic Regression method 
        
        ARGUMENTS :
        _____________________________________________________
        |                |                                  |
        |    y           |    Predictions                   |
        |    tx          |    Samples                       |
        |    initial_w   |    Initial weights               |
        |    max_iters   |    Maximum number of iterations  |
        |    gamma       |    Step size                     |
        |________________|__________________________________|    
            
        RETURN VALUES :
        _____________________________________________________
        |                |                                  |
        |    w           |    Optimal weights               |
        |    loss        |    Final loss value              |
        |________________|__________________________________|
          
    """
    
    losses = []                   #loss array
    
    ws = [initial_w]              #weight array
    w = initial_w                 #current weight
    
    threshold = 1e-8
    
    """ --- ITERATIONS --- """
        
    for iter in range(max_iter):
        
        # gradient L_n formula: x_n * (sigma(x_n * w) - y_n)
        tx_t=tx.T
        grad=tx_t.dot( sigmoid (tx.dot(w))-y)

        # loss function
        loss = np.sum(np.log(1. + np.exp(np.dot(tx, w)))) - np.dot(y.T, np.dot(tx, w))

        # weight
        w = w - gamma * grad
        
        # array filling
        losses.append(loss)
        ws.append(w)
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    
    """ --- RETURN VALUES --- """
    return ws[-1], losses[-1]


### FUNCTION 6 ###

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
     """
        Implementation of the Regularized Logistic Regression method 
        
        ARGUMENTS :
        _____________________________________________________
        |                |                                  |
        |    y           |    Predictions                   |
        |    tx          |    Samples                       |
        |    lambda_     |    Penalty factor                |
        |    initial_w   |    Initial weights               |
        |    max_iters   |    Maximum number of iterations  |
        |    gamma       |    Step size                     |
        |________________|__________________________________|    
            
        RETURN VALUES :
        _____________________________________________________
        |                |                                  |
        |    w           |    Optimal weights               |
        |    loss        |    Final loss value              |
        |________________|__________________________________|
          
    """
    
    losses = []                   #loss array
    
    ws = [initial_w]              #weight array
    w = initial_w                 #current weight
    
    threshold = 1e-8
    
    """ --- ITERATIONS --- """
        
    for iter in range(max_iter):
        
        # gradient L_n formula: x_n * (sigma(x_n * w) - y_n)
        tx_t=tx.T
        grad=tx_t.dot( sigmoid (tx.dot(w))-y) + 2 * lambda_ * w

        # loss function
        loss = np.sum(np.log(1. + np.exp(np.dot(tx, w)))) - np.dot(y.T, np.dot(tx, w)) + lambda_ * np.linalg.norm(w) ** 2

        # weight
        w = w - gamma * grad
        
        # array filling
        losses.append(loss)
        ws.append(w)
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    
    """ --- RETURN VALUES --- """
    return ws[-1], losses[-1]

