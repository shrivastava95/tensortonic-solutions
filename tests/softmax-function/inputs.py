import numpy as np

tests = {
    1: {
        "solver": "softmax", 
        "args": [np.array([1, 2, 3])], 
        "kwargs": {}, 
        "output": [0.09003057, 0.24472847, 0.66524096], 
    },
    2: {
        "solver": "softmax", 
        "args": [np.array([[1, 2, 3], [0, 0, 0]])], 
        "kwargs": {}, 
        "output": [[0.09003057, 0.24472847, 0.66524096], [0.3333, 0.3333, 0.3333]], 
    },
}
