"""

Access to the MNIST dataset of handwritten digits.

"""

from __future__ import division

import os
import logging
import pickle
import gzip
import abc
import numpy as np
import os.path as path
# from learning.datasets import DataSet, datapath

_logger = logging.getLogger(__name__)

floatX = np.float32


class Preproc(object):
    __metaclass__ = abc.ABCMeta

    def preproc(self, X, Y):
        """ Preprocess data and return and X, Y tuple.

        Parameters
        ----------
        X, Y : ndarray

        Returns
        -------
        X, Y : ndarray
        """
        return X, Y

    def late_preproc(self, X, Y):
        """ Preprocess data and return and X, Y tuple.

        Parameters
        ----------
        X, Y : theano.tensor

        Returns
        -------
        X, Y : theano.tensor
        """
        return X, Y

#-----------------------------------------------------------------------------
def datapath(fname):
    """ Try to find *fname* in the dataset directory and return
        a absolute path.
    """
    candidates = [
        path.abspath(path.join(path.dirname(__file__), "../../data")),
        path.abspath("."),
        path.abspath("data"),
    ]
    if 'DATASET_PATH' in os.environ:
        candidates.append(os.environ['DATASET_PATH'])

    for c in candidates:
        c = path.join(c, fname)
        if path.exists(c):
            return c

    raise IOError("Could not find %s" % fname)

class DataSet(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, preproc=[]):
        self._preprocessors = []
        self.add_preproc(preproc)

    def add_preproc(self, preproc):
        """ Add the given preprocessors to the list of preprocessors to be used

        Parameters
        ----------
        preproc : {Preproc, list of Preprocessors}
        """
        if isinstance(preproc, Preproc):
            preproc = [preproc, ]

        for p in preproc:
            assert isinstance(p, Preproc)

        self._preprocessors += preproc

    def preproc(self, X, Y):
        """ Statically preprocess data.

        Parameters
        ----------
        X, Y : ndarray

        Returns
        -------
        X, Y : ndarray
        """
        for p in self._preprocessors:
            X, Y = p.preproc(X, Y)
        return X, Y

    def late_preproc(self, X, Y):
        """ Preprocess a batch of data

        Parameters
        ----------
        X, Y : theano.tensor

        Returns
        -------
        X, Y : theano.tensor
        """
        for p in self._preprocessors:
            X, Y = p.late_preproc(X, Y)
        return X, Y

class MNIST(DataSet):
    def    __init__(self, which_set='train', n_datapoints=None, fname="mnist.pkl.gz", preproc=[]):
        super(MNIST, self).__init__(preproc)

        _logger.info("Loading MNIST data")
        fname = datapath(fname)

        if fname[-3:] == ".gz":
            open_func = gzip.open
        else:
            open_func = open

        with open_func(fname) as f:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = pickle.load(f)

        if which_set == 'train':
            self.X, self.Y = self.prepare(train_x, train_y, n_datapoints)
        elif which_set == 'valid':
            self.X, self.Y = self.prepare(valid_x, valid_y, n_datapoints)
        elif which_set == 'test':
            self.X, self.Y = self.prepare(test_x, test_y, n_datapoints)
        elif which_set == 'salakhutdinov_train':
            train_x = np.concatenate([train_x, valid_x])
            train_y = np.concatenate([train_y, valid_y])
            self.X, self.Y = self.prepare(train_x, train_y, n_datapoints)
        elif which_set == 'salakhutdinov_valid':
            train_x = np.concatenate([train_x, valid_x])[::-1]
            train_y = np.concatenate([train_y, valid_y])[::-1]
            self.X, self.Y = self.prepare(train_x, train_y, n_datapoints)
        else:
            raise ValueError("Unknown dataset %s" % which_set)

        self.n_datapoints = self.X.shape[0]

    def prepare(self, x, y, n_datapoints):
        N = x.shape[0]
        assert N == y.shape[0]

        if n_datapoints is not None:
            N = n_datapoints

        x = x[:N]
        y = y[:N]

        one_hot = np.zeros((N, 10), dtype=floatX)
        for n in range(N):
            one_hot[n, y[n]] = 1.

        return x.astype(floatX), one_hot.astype(floatX)
