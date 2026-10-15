from math import dist

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        self._class_labels = np.unique(ytr)
        n_classes = self._class_labels.size
        n_features = xtr.shape[1]
        self._centroids = np.zeros(shape=(n_classes, n_features))
        for k, label in enumerate(
                self._class_labels):  # media di tutte le immagini della classe k
            self._centroids[k, :] = xtr[ytr == label, :].mean(axis=0)
            return self

    
    def predict(self, xts):
        if self._centroids is None:
            raise ValueError("Train the classifier first (call fit).")
        dist = euclidean_distances(xts, self._centroids) # (n_test,n_classi)
        idx_min = np.argmin(dist, axis=1) # indice del centroide più vicino
        return self._class_labels[idx_min]
