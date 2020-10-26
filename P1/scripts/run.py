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

y, xt, ids = load_csv_data(TRAIN)

# filter -999 data and get best conditioning values
print("Erasing Nan")
(xt, mean, cursed) = erase_nan(xt)
print("Found %d Nan, substituted with %f" % (len(cursed), mean))
print("Performing a better conditioning")
xt = shrink_to(xt, mean, 0.001)

print("Max: ", xt.max())
print("Min: ", xt.min())
print("Conditioning: ", xt.max() / xt.min())

# y order of 1
# x order of mean
#w = np.ones(len(xt[0])) / mean
w = np.zeros(len(xt[0]))
lambda_ = 9e-6
gamma = 1e-6
iters = 500
                        
methods = [[least_squares_GD, w, iters, gamma], 
           [least_squares_SGD, w, iters, gamma, 1], 
           [ridge_regression, lambda_], 
           [least_squares], 
           [logistic_regression, w, iters, gamma], 
           [reg_logistic_regression, lambda_, w, iters, gamma]
           ]

input("Press enter to continue...")

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

