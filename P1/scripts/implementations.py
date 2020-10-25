import numpy as np
import random as rnd

def mini_batch_SDG(y, tx, grad_n, initial_w, max_iters, gamma):

    N = len(y) # length of samples
    Nx = len(tx[0]) # length of arguments
    w = initial_w

    while max_iters > 0: # TODO add threshold condition

        # generate random integer to consider
        NB = rnd.randint(1,N)
        B = random.sample(range(N), NB)

        # compute stochastic gradient
        g = np.zeros(Nx)
        for n in B:
            g += grad_n(grad_n(y[n], tx[n]))
        g /= NB

        # forward step w
        w -= gamma * g
        max_iters -= 1

    return w

def ridge_regression(y, tx, lambda_):

    # Compute optimal weights
    xx = np.dot(np.transpose(tx), tx)
    # Add the lambda on the diagonal
    T = xx

    if lambda_ != 0:
        # add lambda_ contribution, otherwise linear regression
        T += lambda_ * np.identity(len(xx))
        
    xy = np.dot(np.transpose(tx), y)

    # compute result following the formula: w * T = X^t * y
    w = np.linalg.solve(T, xy)
    
    # compute loss
    loss = compute_RMSE(y, tx, w)

    return w, loss


def least_squares(y, tx):
    
    # Compute weight as particular case of ridge regression, _lambda = 0
    return ridge_regression(y, tx, 0)


