import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    g = grad
    w, m, v, g = list(map(np.array, [w, m, v, g]))
    m = beta1 * m + (1-beta1) * g
    v = beta2 * v + (1-beta2) * g * g

    w = w - lr * (weight_decay * w) - (lr * m) / (np.sqrt(v) + eps)
    return w, m, v
