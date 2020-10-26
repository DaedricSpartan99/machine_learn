#!/usr/bin/python

import argparse
from traintest import *
import numpy as np

TRAIN = 'train.csv'
TEST = 'test.csv'

def check_file_exist(filename):
    # Name of the train file
    try:
        # If can open, TRAIN becomes the file descriptor
        open(filename, 'r')
    except:
        raise NameError('Cannot open file %s! Are you sure it exists in this directory' % TRAIN)

def main(test):
    check_file_exist(TRAIN)
    if test:
        return methods_test() 
    else:
        return predict()

def methods_test():
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
            w, accuracy, _ = training(y, xt, method, args[1:])
            weights.append(w)
            P.append(accuracy)

    print('\nFor method %s: In total, there was %f of good predictions on the training set.\n' % (method, np.mean(P)))



def predict():
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

    #input("Press enter to continue...")

    # use ridge with hard-coded lambdas
    lambdas = [9e-06, 0.0212, 1.65e-05, 0.00027, 2.42e-06, 0.000309, 4e-05, 3.63e-10]

    weights = []
    lambda_best = 0
    minloss_index = -1
    losses = []

    for i in range(20):

        minloss = float('inf')
        weights = []
        minloss_index = -1
        
        for (i, lambda_) in enumerate(lambdas):
            print("Aptempting ridge regression with lambda = ", lambda_)
            w, _, loss = training(y, xt, ridge_regression, lambda_, full = True)
            weights.append(w)
            losses.append((lambda_, loss))
            if loss < minloss:
                minloss = loss
                minloss_index = i


        lambda_best = lambdas[minloss_index]
        print("Best lambda: ", lambda_best)

        # generate other 8 lambdas around the minimum
        lambdas = np.linspace(lambda_best * (1.0 - 0.2 / (i+1)), lambda_best * (1.0 + 0.2 / (i+1)), 8)
    
    print("Lambdas: ", lambdas)
    print("Using weight: ", weights[minloss_index])
    
    print("\nLambda  loss")
    for (lamb, loss) in losses:
        print("%.12g  %12g" % (lamb, loss))

    print()

    #TODO  find local minimum lambda_
    
    print("Treating testing data")
    _, xt, ids = load_csv_data(TEST)

    # filter -999 data and get best conditioning values
    print("Erasing Nan")
    (xt, mean, cursed) = erase_nan(xt)
    print("Found %d Nan, substituted with %f" % (len(cursed), mean))
    print("Performing a better conditioning")
    xt = shrink_to(xt, mean, 0.001)

    predictions = test(xt, weights[minloss_index])

    print("Restoring old conditioning")
    xt = shrink_back_from(xt, mean, 0.001)
    print("Restoring Nan in their position")
    xt = restore_nan(xt, cursed)
    
    # submit file
    print("Creating file submission.csv")
    create_csv_submission(ids, predictions, 'submission.csv')

    print("File submission.csv created")
   



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simply ./run to get a valid submission, otherwise ./run -test to test methods. ')
    parser.add_argument('-test', action='store_true', help='Test mandatory methods', default=False)
    args = parser.parse_args()

    main(args.test)
