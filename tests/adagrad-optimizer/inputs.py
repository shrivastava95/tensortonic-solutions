import numpy as np

tests = {
    1: {
        "solver": "adagrad_step", 
        "args": [], 
        "kwargs": dict(w=[1.0, 2.0], g=[0.1, -0.2], G=[0.0, 0.0], lr=0.1), 
        "output": ([0.9, 2.1], [0.01, 0.04]), 
    },
    2: {
        "solver": "adagrad_step", 
        "args": [], 
        "kwargs": dict(w=[1.0, 2.0], g=[0.0, 0.0], G=[0.1, 0.2], lr=0.1), 
        "output": ([1.0, 2.0], [0.1, 0.2]), 
    },
    3: {
        "solver": "adagrad_step", 
        "args": [], 
        "kwargs": dict(w=[0.0], g=[1.0], G=[100.0], lr=0.1), 
        "output": ([1.0, 2.0], [0.1, 0.2]), 
    },
}
