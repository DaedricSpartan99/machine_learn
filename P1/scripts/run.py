#!/usr/bin/python

import argparse
from traintest import *

# TODO, use cv too
def main(da, cv):

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

    
    # TODO implement cross validation
    lambdas = [9e-06, 0.0212, 1.65e-05, 0.00027, 2.42e-06, 0.000309, 4e-05, 3.63e-10]

    # Name of the training data
    TRAINING_DATA = ['train_jet_0_wout_mass.csv', 'train_jet_0_with_mass.csv',
                     'train_jet_1_wout_mass.csv', 'train_jet_1_with_mass.csv',
                     'train_jet_2_wout_mass.csv', 'train_jet_2_with_mass.csv',
                     'train_jet_3_wout_mass.csv', 'train_jet_3_with_mass.csv']
 
    # Name of the test data                   
    TESTING_DATA = ['test_jet_0_wout_mass.csv', 'test_jet_0_with_mass.csv',
                    'test_jet_1_wout_mass.csv', 'test_jet_1_with_mass.csv',
                    'test_jet_2_wout_mass.csv', 'test_jet_2_with_mass.csv',
                    'test_jet_3_wout_mass.csv', 'test_jet_3_with_mass.csv'] 
    
    # if need to split data
    if da:
        data_analysis_splitting(TRAIN, TEST, TRAINING_DATA, TESTING_DATA)

    """
    methods = [[least_squares_GD, w, 2000, 1e-6],
               [least_squares_SGD, w, 500, 1e-6, 1],
               [ridge_regression, 9e-6], 
               [least_squares], 
               [logistic_regression, w, 500, 1e-6], 
               [reg_logistic_regression, w, 500, 1e-6, 9e-6]]
    """

    #W = []
    #P = []
    
    #for param in methods:
    #method = param[0]
    #method_nb = 1 
    #method = methods[method_nb] 

    #method = "least_square"
    #method = "ridge_regression"
    method = "logistic_regression"

    diagnose = open('outputs/train_%s.txt' % (method), 'w')
    weights, prediction_train = training(TRAINING_DATA, lambdas, diagnose)
    #W.append(weights)
    #P.append(prediction_train)
    diagnose.close()

    print('\nFor method %s: In total, there was %f of good predictions on the training set.\n' % (method, prediction_train))

    """ TESTING """
    #test(TESTING_DATA, weights, 'submission.csv')


# parse entering argument to main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' This function implements the best run of team 81 on Kaggle. ')
    parser.add_argument('-da', action='store_true', help='Performs the Data Analysis and Splitting', default=False)
    parser.add_argument('-cv', action='store_true',
                        help='Performs the Cross-Validation. If not called, it will use hardcoded values.',
                        default=False)
    args = parser.parse_args()

    main(args.da, args.cv)
