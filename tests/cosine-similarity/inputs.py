import numpy as np

tests = {
    1: {
        "solver": "cosine_similarity",
        "args": [],
        "kwargs": {"a":[1,2,3] ,"b":[2,4,6]}, 
        "output": 1.0,
    },
    2: {
        "solver": "cosine_similarity",
        "args": [],
        "kwargs": {"a":[0,1] ,"b":[1,0]}, 
        "output": 0.0,
    }
}
