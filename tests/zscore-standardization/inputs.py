import numpy as np

tests = {
    1: {
        "solver": "zscore_standardize", 
        "args": [], 
        "kwargs": dict(X = np.array([[1,2],[3,6],[5,10]])), 
        "output": "Standardized columns (mean=0, std=1)", 
    },
}
