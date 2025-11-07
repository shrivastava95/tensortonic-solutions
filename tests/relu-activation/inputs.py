import numpy as np

tests = {
    1: {
        "solver": "relu", 
        "args": [], 
        "kwargs": dict(x=[-2, -1, 0, 3]), 
        "output": [0.0, 0.0, 0.0, 3.0], 
    },
    2: {
        "solver": "relu", 
        "args": [], 
        "kwargs": dict(x=5.0), 
        "output": [5.0], 
    },
    3: {
        "solver": "relu", 
        "args": [], 
        "kwargs": dict(x=[[-1, 2], [3, -4]]), 
        "output": [[0.0, 2.0], [3.0, 0.0]], 
    },
}
