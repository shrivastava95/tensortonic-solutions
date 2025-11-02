import numpy as np

tests = {
    1: {
        "solver": "dot_product", 
        "args": [], 
        "kwargs": dict(x = [1,2,3], y = [4,5,6]), 
        "output": 32.0, 
    },
    2: {
        "solver": "dot_product", 
        "args": [], 
        "kwargs": dict(x = [1,0], y = [0,1]), 
        "output": 0.0, 
    },
    3: {
        "solver": "dot_product", 
        "args": [], 
        "kwargs": dict(x = [-1,2], y = [3, -1]), 
        "output": -5.0, 
    },
}
