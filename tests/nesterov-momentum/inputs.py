import numpy as np

tests = {
    1: {
        "solver": "nesterov_momentum_step", 
        "args": [],
        "kwargs": dict(w=[1.0, -1.0], v=[0.0, 0.0], grad=[0.5, -0.25], lr=0.1, momentum=0.9), 
        "output": ([0.95, -0.975], [0.05, -0.025]), 
    },
    2: {
        "solver": "nesterov_momentum_step", 
        "args": [],
        "kwargs": dict(w=[1.0, 2.0], v=[0.5, -0.3], grad=[0.1, 0.2], lr=0.1, momentum=0.9), 
        "output": ([0.54, 2.25], [0.46, -0.25]), 
    },
    3: {
        "solver": "nesterov_momentum_step", 
        "args": [],
        "kwargs": dict(w=[2.0], v=[0.0], grad=[0.0], lr=0.1, momentum=0.9), 
        "output": ([2.0], [0.0]), 
    },
}
