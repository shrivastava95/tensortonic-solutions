import numpy as np
from matplotlib import pyplot as plt

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y).reshape([-1, 1])
    
    N, D = X.shape

    w = np.zeros([D, 1]) / np.sqrt(D) # quick, dirty fan-in normalization for fun. Although we don't know enough invariants to make sure that this will have intended effect.
    b = np.zeros([1, 1])
    lr = 0.1 # arbitrary learning rate

    # okay, i guess we got to write the grads now. some math will get us this result:
    #
    # 1. dLi/dli (referred to as dli) = - ( (yi / pi)  * (1 - pi) * (pi)  - ((1 - yi) / (1 - pi)) * (pi) * (1 - pi) ) = (pi - yi) which is the famous simplification.
    # 2. dli/dwjk = (Xi.T)j                =>      so, dli/dw (referred to as dw) = (Xi.T) repeated H times along dim 2
    # => dLi/dw = (pi - yi) * (Xi.T)
    # 3. dli/db (referred to as db) = np.ones_like(b)
    # => dLi/db = (pi - yi) * (np.ones_like(b))
    # 
    # the above are the samplewise gradients, followed by the application of chain rule. for one sample (Xi, yi)
    # 
    # so, in order to compute the batched grads, we average samplewise grads over the batch.
    
    for step in range(steps):
        # forward pass
        l = X @ w + b
        p = _sigmoid(l)
        L = np.mean( -( (y * np.log(p)) + ((1 - y) * np.log(1 - p)) ), axis=0)

        # backprop using chain rule
        dl = p - y
        dw = dl * X
        db = dl

        # aggregate
        dl = np.mean(dl, axis=0) # idk what to reshape it to since its useless now.
        dw = np.mean(dw, axis=0).reshape([-1, 1])
        db = np.mean(db, axis=0).reshape([-1, 1])
        
        # compute updates
        w = w - lr * dw
        b = b - lr * db
    return w, b
