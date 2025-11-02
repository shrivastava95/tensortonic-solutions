import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    if x.ndim not in [3, 4]:
        raise ValueError("yeah")
    x = np.array(x)
    return np.mean(np.mean(x, axis=-1), axis=-1)

