import numpy as np

tests = {
    1: {
        "solver": "calculate_eigenvalues", 
        "args": [], 
        "kwargs": dict(matrix=[[4, 1], [2, 3]]), 
        "output": [2.0, 5.0], 
    },
    2: {
        "solver": "calculate_eigenvalues", 
        "args": [], 
        "kwargs": dict(matrix=[[0, -1], [1, 0]]), 
        "output": "[-1j, 1j] (pure imaginary)", 
    },
    3: {
        "solver": "calculate_eigenvalues", 
        "args": [], 
        "kwargs": dict(matrix=[[1, 2, 3], [4, 5]]), 
        "output": "None (non-square)", 
    },
}
