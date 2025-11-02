import numpy as np

tests = {
    1: {
        "solver": "hinge_loss", 
        "args": [], 
        "kwargs": {
            "y_true": [1, 1, -1],
            "y_score": [2, 0, 0],
            "margin": 1.0,
            "reduction": "mean"
        }, 
        "output": 0.66666667, 
    }
}
