import numpy as np

tests = {
    1: {
        "solver": "scaled_dot_product_attention", 
        "args": [], 
        "kwargs": dict(
            Q = [[1, 0], [0, 1]], # 2x2 query matrix
            K = [[1, 0], [0, 1]], # 2x2 key matrix
            V = [[1, 2], [3, 4]], # 2x2 value matrix
            mask = None
        ), 
        "output": "No Output Given. Whatever.", 
    }
}
