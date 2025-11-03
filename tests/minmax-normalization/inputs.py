import numpy as np

tests = {
    1: {
        "solver": "minmax_scale", 
        "args": [], 
        "kwargs": dict(X = np.array([[1,2],[3,6],[5,10]])), 
        "output": [[0,0],[0.5,0.5],[1,1]], 
    },
}
