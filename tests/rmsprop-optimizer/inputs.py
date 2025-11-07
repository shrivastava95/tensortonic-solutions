import numpy as np

tests = {
    1: {
        "solver": "rmsprop_step", 
        "args": [], 
        "kwargs": dict(w=[1.0, 2.0], g=[0.2, -0.4], s=[0.0, 0.0], lr=0.1), 
        "output": ([0.684, 2.316], [0.004, 0.016]),
    },
    2: {
        "solver": "rmsprop_step", 
        "args": [], 
        "kwargs": dict(w=[5.0], g=[0.0], s=[0.1], lr=0.1), 
        "output": ([5.0], [0.09]),
    },
    3: {
        "solver": "rmsprop_step", 
        "args": [], 
        "kwargs": dict(w=[[1, 2]], g=[[0.1, 0.2]], s=[[0.01, 0.04]], lr=0.1), 
        "output": ([[0.9, 1.9], [0.01, 0.04]]),
    },
}
