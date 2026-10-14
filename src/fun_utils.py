from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """

n_samples = y.size
n_tr = int(n_samples * tr_fraction)  # arrotonda per difetto
idx = np.random.permutation(n_samples)  # indici mescolati
idx_tr = idx[:n_tr]
idx_ts = idx[n_tr:]
return x[idx_tr, :], y[idx_tr], x[idx_ts, :], y[idx_ts]
