import numpy as np

tests = {
    1: {
        "solver": "matrix_normalization",
        "args": [[[3, 4], [1, 0]]],
        "kwargs": {"axis": 1, "norm_type": "l2"},
        "output": [[0.6, 0.8], [1.0, 0.0]],
    },
    2: {
        "solver": "matrix_normalization",
        "args": [[[1, 2], [3, 4]]],
        "kwargs": {"axis": 0, "norm_type": "l1"},
        "output": [[0.25, 1/3], [0.75, 2/3]],
    },
    3: {
        "solver": "matrix_normalization",
        "args": [[[2, 8], [4, 2]]],
        "kwargs": {"axis": 1, "norm_type": "max"},
        "output": [[0.25, 1.0], [1.0, 0.5]],
    },
}
