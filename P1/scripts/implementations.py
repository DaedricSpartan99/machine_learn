import numpy as np

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


