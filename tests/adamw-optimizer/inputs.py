import numpy as np

tests = {
    1: {
        "solver": "adamw_step", 
        "args": [],
        "kwargs": dict(w=[1.0, -2.0], m=[0.0, 0.0], v=[0.0, 0.0], grad=[0.3, -0.7], lr=0.01, weight_decay=0.1), 
        "output":([0.967, -1.966], [0.03, -0.07], [0.00009, 0.00049]),
    },
    2: {
        "solver": "adamw_step", 
        "args": [],
        "kwargs": dict(w=[1.0, 2.0], m=[0.1, 0.2], v=[0.01, 0.04], grad=[0.0, 0.0], lr=0.01, weight_decay=0.1), 
        "output": ([0.99, 1.989], [0.09, 0.18], [0.00999, 0.03996]),
    },
    3: {
        "solver": "adamw_step", 
        "args": [],
        "kwargs": dict(w=[1.0, 2.0], m=[0.1, 0.2], v=[0.01, 0.04], grad=[0.1, 0.2], lr=0.01, weight_decay=0.0), 
        "output": ([0.99, 1.99], [0.1, 0.2], [0.01, 0.04]),
    },
}
