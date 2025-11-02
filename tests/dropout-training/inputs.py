import numpy as np

tests = {
    1: {
        "solver": "dropout", 
        "args": [], 
        "kwargs": dict(x=[1.0, 2.0, 3.0], p=0.0), 
        "output": "([1., 2., 3.], [1., 1., 1.])", 
    },
    2: {
        "solver": "dropout", 
        "args": [], 
        "kwargs": dict(x=[2.0, 4.0], p=0.5), 
        "output": "([0., 8.], [0., 2.]) or ([4., 0.], [2., 0.])", 
    },
}
