import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    matrix = np.array(X)
    if matrix.ndim != 2:
        return None
    elif matrix.shape[0] < 2:
        return None
    
    N = matrix.shape[0]
    mu = np.mean(X, axis=0)
    X = X - mu
    cov = (1 / (N - 1)) * X.transpose(1, 0) @ X
    return cov
