import numpy as np

tests = {
    1: {
        "solver": "swish",
        "args": [ [0, 1, -1, 3] ], 
        "kwargs": dict(), 
        "output": [0.0, 0.731, -0.269, 2.857], 
    },
    2: {
        "solver": "swish",
        "args": [ 0.0 ], 
        "kwargs": dict(), 
        "output": [0.0], 
    },
    3: {
        "solver": "swish",
        "args": [ [[1, -1], [2, -2]] ], 
        "kwargs": dict(), 
        "output": [[0.731, -0.269], [1.762, -0.238]], 
    },
}
