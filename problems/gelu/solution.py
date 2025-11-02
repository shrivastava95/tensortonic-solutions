import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    erf = np.vectorize(math.erf)
    x = np.array(x)
    sqrt2 = math.sqrt(2)
    z = 0.5 * ( x ) * ( 1 + erf(x / sqrt2) )
    return z
