import numpy as np

tests = {
    1: {
        "solver": "f1_micro",
        "args": [],
        "kwargs": {"y_true": [0,1,1], "y_pred": [0,1,0]},
        "output": 0.66667,
    },
    2: {
        "solver": "f1_micro",
        "args": [],
        "kwargs": {"y_true": [0,1,2,2], "y_pred": [0,1,2,2]},
        "output": 1.0,
    },
    3: {
        "solver": "f1_micro",
        "args": [],
        "kwargs": {"y_true": [2,2,1,0], "y_pred": [1,2,1,0]},
        "output": 0.75,
    }
}
