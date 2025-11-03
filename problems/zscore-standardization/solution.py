import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    X = np.array(X)
    mu = np.mean(X, axis=axis, keepdims=True)
    std = eps + np.maximum(eps, np.std(X, axis=axis, keepdims=True))
    z = (X - mu) / std
    return z
