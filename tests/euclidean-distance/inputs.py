import numpy as np

tests = {
    1: {
        "solver": "euclidean_distance", 
        "args": [], 
        "kwargs": dict(x = [3,4], y = [0,0]), 
        "output": 5.0, 
    },
    2: {
        "solver": "euclidean_distance", 
        "args": [], 
        "kwargs": dict(x = [1,2,3], y = [4,5,6]), 
        "output": 5.196152422706632, 
    },
    3: {
        "solver": "euclidean_distance", 
        "args": [], 
        "kwargs": dict(x = [0,0,0], y = [0,0,0]), 
        "output": 0.0, 
    },
}
