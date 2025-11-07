import numpy as np

tests = {
    1: {
        "solver": "clip_gradients", 
        "args": [], 
        "kwargs": dict(g=[0.1, 0.2, 0.2], max_norm=1.0), 
        "output": [0.1, 0.2, 0.2], 
    },
    2: {
        "solver": "clip_gradients", 
        "args": [], 
        "kwargs": dict(g=[6, 8], max_norm=5.0), 
        "output": [3.0, 4.0], 
    },
    3: {
        "solver": "clip_gradients", 
        "args": [], 
        "kwargs": dict(g=[[2, 2], [2, 2]], max_norm=2.0), 
        "output": [[1.0, 1.0], [1.0, 1.0]], 
    },
}
