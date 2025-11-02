import numpy as np

tests = {
    1: {
        "solver": "add_positional_encoding",
        "args": [
            np.zeros([3,4]), 
        ],
        "kwargs": {"base": 10000.0},
        "output": [[0.0000,1.0000,0.0000,1.0000],[0.8415,0.5403,0.0100,0.9999],[0.9093,-0.4161,0.0200,0.9998]]
    },
    2: {
        "solver": "add_positional_encoding",
        "args": [
            np.zeros([2, 3]), 
        ],
        "kwargs": {"base": 10000.0},
        "output": [[0.0000,1.0000,0.0000],[0.8415,0.5403,0.0022]]
    },
}
