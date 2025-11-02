import numpy as np

tests = {
    1: {
        "solver": "covariance_matrix", 
        "args": [], # put any input args here
        "kwargs": dict(X=[[1, 2], [2, 3], [3, 4]]), # put any input kwargs here
        "output": [[1.0, 1.0], [1.0, 1.0]], # put your public / custom test case outputs here
    },
    2: {
        "solver": "covariance_matrix", 
        "args": [], # put any input args here
        "kwargs": dict(X=[[1, 0], [0, 1]]), # put any input kwargs here
        "output": [[0.5, -0.5], [-0.5, 0.5]], # put your public / custom test case outputs here
    },
    3: {
        "solver": "covariance_matrix", 
        "args": [], # put any input args here
        "kwargs": dict(X=[[1, 2, 3]]), # put any input kwargs here
        "output": "None (only 1 sample)", # put your public / custom test case outputs here
    },
}
