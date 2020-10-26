import numpy as np
import random as rnd
from proj1_helpers import *

"""
    Utils
"""

# define threshold constant
eps = 1e-5

### FUNCTION 1 ###

def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    if initial_w is None:
        initial_w = np.ones(len(tx[0]))

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    #compute gradient and loss
    for n_iter in range(max_iters):
        
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w -= gamma * gradient    #update w by the gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)

    w_opt=ws[-1]
    loss_opt=losses[-1]
    return w_opt,loss_opt

### FUNCTION 2 ###

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size = 1):

    if initial_w is None:
        initial_w = np.ones(len(tx[0]))

    if (batch_size>len(y)):
        print("The batch size was bigger than the whole dataset, it was downsized to match that of the dataset")
        batch_size=len(y)

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size): 
            gradient = compute_gradient(minibatch_y,minibatch_tx,w) # Can compute the normal gradient for each minibatch
            loss = compute_loss(minibatch_y,minibatch_tx,w)

        w = w-gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)

    w_opt=ws[-1]
    loss_opt=losses[-1]

    return w_opt,loss_opt


"""
    Methods implementations
"""

### FUNCTION 4 ###

def ridge_regression(y, tx, lambda_):

    # Compute optimal weights
    T = np.dot(np.transpose(tx), tx) # dim(T) = M * M
    xy = np.dot(np.transpose(tx), y)
    N = len(tx) # hoow many rows
    M = len(T[0]) # how many columns

    # add lambda_ contribution, otherwise linear regression
    if np.abs(lambda_) > eps:   # we consider any lammda < eps to be equal to 0
        T += lambda_ * (2*N) * np.identity(M)

    w = np.linalg.solve(T,xy)   
        
    #THIS PART STILL NEEDS TO BE CHECKED
    cost_fct = lambda y, tx, w: MSE_cost(y, tx, w) - lambda_ * np.dot(w.T,w)
    
    return w, compute_cost(y, tx, w, cost_fct) 


### FUNCTION 3 ###

def least_squares(y, tx):
    
    # Compute weight as particular case of ridge regression, _lambda = 0
    return ridge_regression(y, tx, 0)

def sigmoid(z):
    arg = np.exp(-z) 
    return 1.0 / (arg + 1.0)

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
    
    if initial_w is None:
        initial_w = np.ones(len(tx[0]))

    losses = []                   #loss array
    
    ws = [initial_w]              #weight array
    w = initial_w                 #current weight
    
    threshold = 1e-8
    
    """ --- ITERATIONS --- """
        
    for iter in range(max_iters):
        
        # gradient L_n formula: x_n * (sigma(x_n * w) - y_n)
        tx_t = np.transpose(tx)
        z = np.dot(tx, w)

        # avoid data overflow by exponential
        z[z > 500] = 500
        z[z < -500] = -500
        
        # compute gradient
        grad = np.dot(tx_t, sigmoid(z) - y)

        # loss function
        loss = np.sum(np.log(1. + np.exp(z))) - np.dot(y.T, np.dot(tx, w))

        # weight
        w = w - gamma * grad
        
        # array filling
        losses.append(loss)
        ws.append(w)
        
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
    
    if initial_w is None:
        initial_w = np.ones(len(tx[0]))

    losses = []                   #loss array
    
    ws = [initial_w]              #weight array
    w = initial_w                 #current weight
    
    threshold = 1e-8
    
    """ --- ITERATIONS --- """
        
    for iter in range(max_iters):
        
        # gradient L_n formula: x_n * (sigma(x_n * w) - y_n)
        tx_t=tx.T
        z = tx.dot(w)
        z[z > 500] = 500
        z[z < -500] = -500

        grad=tx_t.dot( sigmoid (z)-y) + 2 * lambda_ * w

        # loss function
        loss = np.sum(np.log(1. + np.exp(z))) - np.dot(y.T, np.dot(tx, w)) + lambda_ * np.linalg.norm(w) ** 2

        # weight
        w = w - gamma * grad

        #print(np.linalg.norm(grad))
        
        # array filling
        losses.append(loss)
        ws.append(w)
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    
    """ --- RETURN VALUES --- """
    return ws[-1], losses[-1]

