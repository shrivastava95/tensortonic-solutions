import numpy as np

def _stable_sigmoid(x):
    # cool thing i learnt here is that the argument to np.exp, if too large (for example: np.exp(749)) it will cause an overflow. which is why you want to compute the sigmoid operation in such a way that this scenario never happens.
    # feel free to raise a PR or an Issue in case you have a faster alternative to this function, i know my implementation is quite slow.
    x = np.array(x)
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + 1))


def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.array(x)
    if not x.shape:
        x = x.reshape(-1)
    return x * _stable_sigmoid(x)
