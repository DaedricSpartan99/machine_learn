from proj1_helpers import *
from implementations import * #eps, ridge_regression, logistic_regression_mb

def wrong_pred_ratio(y, tx, w):
    """
        Return the percentage of wrong predictions (between 0 and 1)
    """

    P = np.dot(tx, w)
    N = len(P)
    
    # for all positive value: set 1, otherwise, set -1
    P[P > 0] = 1
    P[P <= 0] = -1
    
    # sum all matching values with Y
    diff = np.abs(P - y)
    correct = np.sum(diff < 1e-10)
    wrong = N - correct
    
    # compute ratio
    return float(wrong) / N



def training(y, xt, method, args):
    """
        Train on the data with the degree_star and lambda_star found by the cross-validation.

        At the end, we return the best weights and the percentage of correct prediction on the
        training set.
    """
    
    print("Testing: ", method.__name__)
    #print("index  accuracy  loss", file= outfile)

    # Load the file
    #y, xt, ids = load_csv_data(samples)
    N = len(y)

    w = np.ones(len(xt[0]))

    # split data into training and test accuracy
    # default 80%
    Nt = int(0.8 * N)
    B = rnd.sample(range(N), Nt)
    
    #w, loss = least_squares_GD(y, xt, w, 2000, 1e-6)

    #w, loss = least_squares_SGD(y, xt, w, 500, 1e-6, 1)

    #w, loss = least_squares(y, xt)
    
    #w, loss = ridge_regression(y, xt, 9e-03)

    #w, loss = logistic_regression(y[:Nt], xt[:Nt], w, 2000, 1e-6)
    
    #w, loss = reg_logistic_regression(y, xt, 9e-06, w, 500, 1e-6)

    w, loss = method(y[B], xt[B], *args)
    
    # complementary of B
    BC = list(range(N))
    for index in sorted(B, reverse=True):
        del BC[index]

    # make prediction with the remaining data
    ratio = wrong_pred_ratio(y[BC], xt[BC], w)

    accuracy = 100. * (1. - ratio)

    print('  Good prediction: %.3g' % (accuracy))
    print('  Loss: %.3g' % loss)

    #print("%d  %.3g  %.3g" % (idx, accuracy, loss), file=outfile)

    return w, 100 * accuracy


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def test(data, weights, output_name):

    # Load the file
    _, x_test, ids_test = load_csv_data(data)

    # Predict the labels
    y_pred = predict_labels(weights, x_test)

    # Write the file of predictions
    create_csv_submission(ids_test, y_pred, output_name)

    print(u'Data are ready to be submitted!')

