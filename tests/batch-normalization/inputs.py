import numpy as np

tests = {
    1: {
        "solver": "batch_norm_forward", 
        "args": [], 
        "kwargs": dict(x = np.array([[1., 2.],
              [3., 6.],
              [5., 10.]]),
gamma = [1., 0.5],
beta  = [0., 1.]), 
        "output": """y = batch-normalized activations where each column
(feature) has mean ≈ 0 and variance ≈ 1 before scaling.""", 
    },
}
