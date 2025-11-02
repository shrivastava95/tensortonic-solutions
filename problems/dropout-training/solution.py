import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    if rng is None:
        rng = np.random.random
    else:
        rng = rng.random
    
    mask = (rng(x.shape) <= (1-p))
    denom = 1.0 if p == 0.0 else (1 / (1 - p))
    x_masked = np.where(mask, x, 0.0) * denom
    pattern = np.where(mask, denom, 0.0)
    return x_masked, pattern


