from proj1_helpers import *
from implementations import * #eps, ridge_regression, logistic_regression_mb

def get_headers(data_path):
    """
        Get the headers from the file given in parameter
    """

    f = open(data_path, 'r')
    reader = csv.DictReader(f)
    headers = reader.fieldnames

    return headers

def write_data(output, y, tx, ids, headers, type_):
    """
        Write the data into a CSV file
    """
    with open(output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=headers)
        writer.writeheader()
        if type_ == 'train':
            for r1, r2, r3 in zip(ids, y, tx):
                if r2 == 1:
                    pred = 's'
                elif r2 == -1:
                    pred = 'b'
                else:
                    pred = r2
                dic = {'Id': int(r1), 'Prediction': pred}
                for i in range(len(r3)):
                    dic[headers[i + 2]] = float(r3[i])
                writer.writerow(dic)
        elif type_ == 'test':
            for r1, r3 in zip(ids, tx):
                dic = {'Id': int(r1), 'Prediction': '?'}
                for i in range(len(r3)):
                    dic[headers[i + 2]] = float(r3[i])
                writer.writerow(dic)


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



def training(samples, lambdas, outfile):
    """
        Train on the data with the degree_star and lambda_star found by the cross-validation.

        At the end, we return the best weights and the percentage of correct prediction on the
        training set.
    """
    weights = []  # weights array
    total = 0 # total number of samples
    meanGP = 0 # mean of good prediction

    print("index  accuracy  loss", file= outfile)

    for idx, data in enumerate(samples):
        # Print that we start the training
        print(u'Training with file {0:s}'.format(data))
        print(u'-----------------------------------------------------')

        # Load the file
        y, xt, ids = load_csv_data(data)
        N = len(y)

        w = np.zeros(len(xt[0]))
    
        #w, loss = least_squares_GD(y, xt, w, 2000, 1e-6)

        #w, loss = least_squares_SGD(y, xt, w, 500, 1e-6, 1)

        #w, loss = least_squares(y, xt)
        
        w, loss = ridge_regression(y, xt, 9e-06)

        #w, loss = logistic_regression(y, xt, w, 500, 1e-6)
        
        #w, loss = reg_logistic_regression(y, xt, 9e-06, w, 500, 1e-6)

        # Get the percentage of wrong prediction
        ratio = wrong_pred_ratio(y, xt, w)

        accuracy = 100. * (1. - ratio)

        print('  Good prediction: %.3g' % (accuracy))
        print('  Loss: %.3g' % loss)

        print("%d  %.3g  %.3g" % (idx, accuracy, loss), file=outfile)

        # Update the total number of entries tested/trained
        total += N
        # Update the mean value of good prections
        meanGP += (1 - ratio) * N

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

        # Load the file
        _, x_test, ids_test = load_csv_data(data)

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


# TODO, modify a little bit once understood
def data_analysis_splitting(TRAIN, TEST, TRAINING_DATA, TESTING_DATA):
    """
        This long function is used for the data analysis and splitting.
        This function will split the data according as the description made in the report.

        We first split in 4 by the number of jets, then we split by the remaining NaNs in the
        first column. Then, we can write the new files.
    """

    print('START THE DATA ANALYSIS / SPLITTING FOR DATA-SETS')
    print('  Load the data. It may take a few seconds.')

    # First we load the data
    headers = get_headers(TRAIN)
    y_train, tx_train, ids_train = load_csv_data(TRAIN)
    y_test, tx_test, ids_test = load_csv_data(TEST)

    # Start the loop for the four kind of jets
    for jet in range(4):
        print("  Cleaning for Jet {0:d}".format(jet))

        # Get the new matrix with only the same jets
        # The information about the number of jets is in column 22
        tx_jet_train = tx_train[tx_train[:, 22] == jet]
        tx_jet_test = tx_test[tx_test[:, 22] == jet]

        # Cut the predictions for the same jet
        y_jet_train = y_train[tx_train[:, 22] == jet]
        y_jet_test = y_test[tx_test[:, 22] == jet]

        # Cut the ids for the same jet
        ids_jet_train = ids_train[tx_train[:, 22] == jet]
        ids_jet_test = ids_test[tx_test[:, 22] == jet]

        # Delete column 22 in Sample matrix
        tx_jet_train = np.delete(tx_jet_train, 22, 1)
        tx_jet_test = np.delete(tx_jet_test, 22, 1)

        # Delete column 24 (column 1 is ids, column 2 is pred) in headers
        headers_jet = np.delete(headers, 24)

        # Get all the columns with only NaNs
        nan_jet = np.ones(tx_jet_train.shape[1], dtype=bool)
        header_nan_jet = np.ones(tx_jet_train.shape[1] + 2, dtype=bool)
        for i in range(tx_jet_train.shape[1]):
            array = tx_jet_train[:, i]
            nbr_nan = len(array[array == -999])
            if nbr_nan == len(array):
                nan_jet[i] = False
                header_nan_jet[i + 2] = False

        # For Jet 0, there is a really big outlier in the column 3. So, we will remove it
        if jet == 0:
            to_remove = (tx_jet_train[:, 3] < 200)

        """ Start removing values """

        if jet == 0:
            tx_jet_train = tx_jet_train[to_remove, :]
            y_jet_train = y_jet_train[to_remove]
            ids_jet_train = ids_jet_train[to_remove]

            # We also remove the last column which is full of 0
            nan_jet[-1] = False
            header_nan_jet[-1] = False

        # Delete the columns in tX and headers
        tx_jet_train = tx_jet_train[:, nan_jet]
        tx_jet_test = tx_jet_test[:, nan_jet]

        headers_jet = headers_jet[header_nan_jet]

        # Get the NaNs in the mass
        nan_mass_jet_train = (tx_jet_train[:, 0] == -999)
        nan_mass_jet_test = (tx_jet_test[:, 0] == -999)
        header_nan_mass_jet = np.ones(len(headers_jet), dtype=bool)
        header_nan_mass_jet[2] = False

        # Write the files
        write_data(TRAINING_DATA[2 * jet], y_jet_train[nan_mass_jet_train], tx_jet_train[nan_mass_jet_train, :][:, 1:],
                   ids_jet_train[nan_mass_jet_train], headers_jet[header_nan_mass_jet], 'train')

        write_data(TRAINING_DATA[2 * jet + 1], y_jet_train[~nan_mass_jet_train], tx_jet_train[~nan_mass_jet_train, :],
                   ids_jet_train[~nan_mass_jet_train], headers_jet, 'train')

        write_data(TESTING_DATA[2 * jet], y_jet_test[nan_mass_jet_test], tx_jet_test[nan_mass_jet_test, :][:, 1:],
                   ids_jet_test[nan_mass_jet_test], headers_jet[header_nan_mass_jet], 'test')

        write_data(TESTING_DATA[2 * jet + 1], y_jet_test[~nan_mass_jet_test], tx_jet_test[~nan_mass_jet_test, :],
                   ids_jet_test[~nan_mass_jet_test], headers_jet, 'test')

    print("FINISHED SPLITTING THE DATA-SETS")




