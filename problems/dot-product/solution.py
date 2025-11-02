import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x, y = list(map(np.array, [x, y]))
    return (np.sum(x * y)).item()
    
