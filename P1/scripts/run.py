#!/usr/bin/python

import argparse
from traintest import *
import numpy as np

# Name of the train file
TRAIN = 'train.csv'
try:
    # If can open, TRAIN becomes the file descriptor
    open(TRAIN, 'r')
except:
    raise NameError('Cannot open file %s! Are you sure it exists in this directory' % TRAIN)

# Name of the test file
TEST = 'test.csv'
try:
    # If can open, TEST becomes the file descriptor
    open(TEST, 'r')
except:
    raise NameError('Cannot open file %s! Are you sure it exists in this directory' % TEST)

methods = [[least_squares_GD, None, 500, 1e-6], 
           [least_squares_SGD, None, 500, 1e-3, 1], 
           [ridge_regression, 9e-6], 
           [least_squares], 
           [logistic_regression, None, 500, 1e-6], 
           [reg_logistic_regression, 9e-6, None, 500, 1e-6]
           ]

y, xt, ids = load_csv_data(TRAIN)

# filter data
xt = filter_nan(xt)

for args in methods:
    method = args[0]

    weights = []
    P = []

    for iter in range(5):
        w, accuracy = training(y, xt, method, args[1:])
        weights.append(w)
        P.append(accuracy)

    print('\nFor method %s: In total, there was %f of good predictions on the training set.\n' % (method, np.mean(P)))


""" TESTING """
#test(TEST, weights, 'submission.csv')

