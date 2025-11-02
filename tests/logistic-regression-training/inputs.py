import numpy as np

tests = {
    1: {
        "solver": "train_logistic_regression",
        "args": [],
        "kwargs": {
            "X": [[0], [1], [2], [3]],
            "y": [0,0,1,1],
            "lr": 0.1,
            "steps": 500,
        },
        "output": "Accuracy >= 95%",
    }
}
