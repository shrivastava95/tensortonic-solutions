import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    X = np.array(X)
    num = (X - np.min(X, axis=axis, keepdims=True))
    denum = np.max(X, axis=axis, keepdims=True) - np.min(X, axis=axis, keepdims=True)
    denum = np.where(denum == 0, eps, denum)

    return num / denum 
