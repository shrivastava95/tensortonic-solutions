import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.array(x)
    if not x.shape:
        x = x.reshape(-1)

    return np.maximum(0, x)
