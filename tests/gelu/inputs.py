import numpy as np

tests = {
    1: {
        "solver": "gelu", 
        "args": [], 
        "kwargs": dict(x = [-1.0, 0.0, 1.0]), 
        "output": [-0.158655, 0.0, 0.841345], 
    }, 
    2: {
        "solver": "gelu", 
        "args": [], 
        "kwargs": dict(x = 2.0), 
        "output": 1.954499, 
    },
    3: {
        "solver": "gelu", 
        "args": [], 
        "kwargs": dict(x = [[-2., -1.],[0., 1.]]), 
        "output": [[-0.045500,-0.158655],[0.000000,0.841345]], 
    },
}
