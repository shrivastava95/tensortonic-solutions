import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    num = np.exp(x)
    denum = np.sum(num, axis=-1, keepdims=True)
    ans = num / denum
    return ans

