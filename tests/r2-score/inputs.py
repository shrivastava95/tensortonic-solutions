import numpy as np

tests = {
    1: {
        "solver": "r2_score",
        "args": [],
        "kwargs": dict(y_true=[3,4,5], y_pred=[2.9,4.1,5.0]), 
        "output": 0.99,
    },
    2: {
        "solver": "r2_score",
        "args": [],
        "kwargs": dict(y_true=[1,1,1], y_pred=[1,1,1]), 
        "output": 1.0,
    },
    3: {
        "solver": "r2_score",
        "args": [],
        "kwargs": dict(y_true=[1,1,1], y_pred=[0,2,1]), 
        "output": 0.0,
    },
}
