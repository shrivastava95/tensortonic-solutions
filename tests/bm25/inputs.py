import numpy as np

tests = {
    1: {
        "solver": "bm25_score", 
        "args": [], 
        "kwargs": dict(query_tokens=["machine","learning"], docs=[["introduction","to","machine","learning"], ["deep","learning","basics"], ["cooking","pasta","guide"]]), 
        "output": [1.34110, 0.49005, 0.00000], 
    },
    2: {
        "solver": "bm25_score", 
        "args": [],
        "kwargs": dict(query_tokens=["data"], docs=[["data","science"], ["big","data","analytics"], ["cooking","recipes"]]),
        "output": [0.49917, 0.42081, 0.00000], 
    },
}
