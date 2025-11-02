import numpy as np

tests = {
    1: {
        "solver": "leaky_relu", 
        "args": [], 
        "kwargs": dict(x = [-2, -1, 0, 1, 2], alpha = 0.1),
        "output": [-0.2, -0.1, 0.0, 1.0, 2.0],
    },
    2: {
        "solver": "leaky_relu", 
        "args": [], 
        "kwargs": dict(x = [-5, 5], alpha = 0.01),
        "output": [-0.05, 5.0],
    }
}
