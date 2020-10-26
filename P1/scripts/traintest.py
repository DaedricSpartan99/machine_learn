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

    if N == 0:
        return 1.0
    
    # compute ratio
    return float(wrong) / N



def training(y, xt, method, *args, **kargs):
    
    print("Training method: ", method.__name__)

    N = len(y)

    w = np.ones(len(xt[0]))

    # split data into training and test accuracy
    # default 80%
    Nt = int(0.8 * N)

    if 'full' in kargs and kargs['full']:
        print("Executing full training mode")
        Nt = N

    B = rnd.sample(range(N), Nt)
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

    return w, accuracy, loss


def test(xt, weights):

    # predict y
    y = np.dot(xt, weights)

    # transform them in 1, -1
    y[ np.where(y <= 0) ] =  -1
    y[ np.where(y > 0) ] =  1

    return y


