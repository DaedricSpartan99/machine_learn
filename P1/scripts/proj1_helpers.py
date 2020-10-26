# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

"""
     Conditioning            
"""

# remove nan
def erase_nan(xt):
    cursed = []
    mean = 0.0
    N = len(xt)
    M = len(xt[0])
    count = 0

    for i in range(N):
        for j in range(M):
            # if Nan
            if np.abs(xt[i,j] + 999) < 1e-10:
                cursed.append((i,j))
            else:
                mean += xt[i,j]
                count += 1

    mean /= count

    for (i,j) in cursed:
        xt[i,j] = mean

    return xt, mean, cursed

# restore nan
def restore_nan(xt, cursed):
    for (i,j) in cursed:
        xt[i,j] = -999
    return xt

# shrink closer to mean
def shrink_to(xt, mean, ratio):
    return (xt - mean) * ratio + mean

def shrink_back_from(xt, mean, ratio):
    return (xt - mean) / ratio + mean
            
# COMPUTE LOSS & GRADIENT for least square GD

def compute_loss(y, tx, w):    #using MSE, give L not L_n
        e = y-tx.dot(w)
        cost=(np.dot(e.T, e))/(2*len(y))  #I think we can remove the .T in e.T
        return cost

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    tx_t = np.transpose(tx)
    return - tx_t.dot(e)/ len(y)
# COMPUTE THE BATCHES FOR SGD

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
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

def compute_cost(y, xt, w, cost_fct_n):

    # Errors evaluation
    #errors = y - fw(xt, w)

    # Compute the cost
    return np.mean(cost_fct_n(y, xt, w))

# Particular case of RMSE implementations

def MSE_fw(xt, w):

    # X^t * w
    #return np.transpose(xt) * w
    return np.dot(xt, w)

# elementwise cost function
def MSE_cost_fct(y, xt, w):

    errors = y - MSE_fw(xt, w)

    # euclidean_norm(errors) / 2
    return np.power(errors,2) / 2

# compute cost for RMSE particular case
def MSE_cost(y, xt, w):
    return compute_cost(y, xt, w, MSE_cost_fct)



def logistic_sigmoid(z):
    arg = np.exp(z) 
    return arg / (1 + arg)
