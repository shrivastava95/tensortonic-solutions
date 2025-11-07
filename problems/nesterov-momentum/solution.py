import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    g = grad
    w, v, g = list(map(np.array, [w, v, g]))
    w_look = w - momentum * v
    v = momentum * v + lr * g
    w = w - v
    return w, v
