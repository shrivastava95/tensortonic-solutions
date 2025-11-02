import numpy as np

tests = {
    1: {
        "solver": "linear_lr", 
        "args": [], 
        "kwargs": dict(step=0, total_steps=100, initial_lr=1e-3, final_lr=0.0, warmup_steps=10), 
        "output": 0.0, 
    },
    2: {
        "solver": "linear_lr", 
        "args": [], 
        "kwargs": dict(step=10, total_steps=100, initial_lr=1e-3, final_lr=0.0, warmup_steps=10),
        "output": 0.001, 
    },
    3: {
        "solver": "linear_lr", 
        "args": [], 
        "kwargs": dict(step=50, total_steps=100, initial_lr=1e-3, final_lr=0.0, warmup_steps=10),
        "output": 0.00055, 
    },
}
