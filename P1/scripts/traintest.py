from proj1_helpers import *
from implementations import ridge_regression

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



def training(samples, lambdas):
    """
        Train on the data with the degree_star and lambda_star found by the cross-validation.

        At the end, we return the best weights and the percentage of correct prediction on the
        training set.
    """
    weights = []  # weights array
    total = 0 # total number of samples
    meanGP = 0 # mean of good prediction

    for idx, data in enumerate(samples):
        # Print that we start the training
        #print(u'Training with file {0:s}'.format(data))
        #print(u'-----------------------------------------------------')
        # Load the file
        y, xt, ids = load_csv_data(data)
        N = len(y)

        # TODO Ridge Regression to get the best weights
        #x_train = build_poly_cross_terms(x_train, degree_star[idx],
        #                                  ct=ct[idx], sqrt=sqrt[idx], square=square[idx])

        w, _ = ridge_regression(y, xt, lambdas[idx])

        # Get the percentage of wrong prediction
        ratio = wrong_pred_ratio(y, tx, w)

        print(u'  Good prediction: {0:f}'.format(100. * (1. - ratio)))

        # Update the total number of entries tested/trained
        total += N
        # Update the mean value of good prections
        meanGP += (1 - val) * N

        weights.append(w)

    return weights, 100 * meanGP / total

def test(TESTING_DATA, weights, output_name):
    """
        Use the degree_star and lambda_star from the Cross-Validation as well as the weights
        from the Training to test on the TEST data-set. It will write the file of predictions
        ready to be submitted on Kaggle.
    """

    y_pred = []
    ids_pred = []

    # Support multiple input files
    for idx, data in enumerate(TESTING_DATA):
        # Print that we start the testing
        print("Testing with file %s" % data)
        print("-----------------------------------------------------")
        # Recreate the file
        data_file = data
        # Load the file
        _, x_test, ids_test = load_csv_data(data_file)

        # Build the polynomial
        #tx_test = build_poly_cross_terms(x_test, degree_star[idx],
        #                                 ct=ct[idx], sqrt=sqrt[idx], square=square[idx])

        # Predict the labels
        y_pred.append(predict_labels(weights[idx], tx_test))
        ids_pred.append(ids_test)

    # Put all the prediction together given the IDs
    ids = []
    pred = []

    idx = min(ids_pred[:][0])

    length = np.sum(len(i) for i in y_pred)

    print("Concatenate the predictions.")

    for i in range(length):
        for j in range(len(TESTING_DATA)):
            if len(ids_pred[j]) > 0:
                if ids_pred[j][0] == idx:
                    ids.append(idx)
                    pred.append(y_pred[j][0])
                    ids_pred[j] = np.delete(ids_pred[j], 0)
                    y_pred[j] = np.delete(y_pred[j], 0)
                    break

        if i % 100000 == 0:
            print(u'  {0:d}/{1:d} concatenated'.format(i, length))

        idx += 1

    # Transform the variables in ndarray
    pred = np.array(pred)
    ids = np.array(ids)

    # Write the file of predictions
    create_csv_submission(ids, pred, output_name)

    print(u'Data are ready to be submitted!')





