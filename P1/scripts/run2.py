{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The autoreload extension is already loaded. To reload it, use:\n",
      "  %reload_ext autoreload\n"
     ]
    }
   ],
   "source": [
    "# Useful starting lines\n",
    "%matplotlib inline\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from proj1_helpers import *\n",
    "from implementations import *\n",
    "\n",
    "def main():\n",
    "    print(\"main\")\n",
    "    DATA_TRAIN_PATH = 'train.csv' \n",
    "    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)\n",
    "    \n",
    "    init_w=np.zeros(len(tX[0]))\n",
    "    max_iter, gamma = 500, 1e-9\n",
    "    #w, loss = least_squares_GD(y, tX, init_w, max_iter, gamma )\n",
    "    #w, loss = least_squares_SGD(y, tX, init_w, max_iter, gamma )\n",
    "    #w, loss = least_squares(y, tX)\n",
    "    #w, loss = logistic_regression(y, tX, init_w, max_iter, gamma )\n",
    "    #w, loss = reg_logistic_regression(y, tX, init_w, 0.001, max_iter, gamma )\n",
    "    w, loss = ridge_regression(y, tX, 23)\n",
    "\n",
    "    \"\"\"\n",
    "    Return the percentage of wrong predictions (between 0 and 1)\n",
    "    \"\"\"\n",
    "\n",
    "    P = np.dot(tX, w)\n",
    "    N = len(P)\n",
    "\n",
    "    # for all positive value: set 1, otherwise, set -1\n",
    "    P[P > 0] = 1\n",
    "    P[P <= 0] = -1\n",
    "\n",
    "    # sum all matching values with Y\n",
    "    diff = np.abs(P - y)\n",
    "    correct = np.sum(diff < 1e-10)\n",
    "    wrong = N - correct\n",
    "\n",
    "    # compute ratio\n",
    "    r= float(wrong) / N\n",
    "    print (\"ratio :\", r, \"\\n P:\", P)\n",
    "    print(\"correct:\", correct)\n",
    "    \n",
    "    DATA_TEST_PATH = 'test.csv' # TODO: download train data and supply path here \n",
    "    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)\n",
    "    \n",
    "    OUTPUT_PATH = 'submission2.csv' # TODO: fill in desired name of output file for submission\n",
    "    y_pred = predict_labels(w, tX_test)\n",
    "    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
