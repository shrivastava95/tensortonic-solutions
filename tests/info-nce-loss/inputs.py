import numpy as np

tests = {
    1: {
        "solver": "info_nce_loss", 
        "args": [], 
        "kwargs": dict(Z1=[[1,0],[0,1]], Z2=[[1,0],[0,1]], temperature=0.1), 
        "output": "~0.0 (low loss)", 
    },
    2: {
        "solver": "info_nce_loss", 
        "args": [], 
        "kwargs": dict(Z1=[[1,0],[0,1]], Z2=[[0,1],[1,0]], temperature=0.1), 
        "output": "~693.1 (high loss)", 
    },
    3: {
        "solver": "info_nce_loss", 
        "args": [], 
        "kwargs": dict(Z1=[[1,0],[0,1]], Z2=[[1,0],[0,1]], temperature=1.0), 
        "output": "~0.69 (moderate loss)", 
    },
}
