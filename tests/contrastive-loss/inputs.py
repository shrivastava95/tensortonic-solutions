import numpy as np

tests = {
    1: {
        "solver": "contrastive_loss", 
        "args": [], 
        "kwargs": dict(a = [1., 0.], b = [1., 0.], y = [1], margin = 1.0), 
        "output": "d=0 → loss = 1*0² + 0*max(0,1-0)² = 0.0", 
    },
    2: {
        "solver": "contrastive_loss", 
        "args": [], 
        "kwargs": dict(a = [0., 0.], b = [0.5, 0.], y = [0], margin = 1.0), 
        "output": "d=0.5 → loss = 0*0.25 + 1*(1-0.5)² = 0.25", 
    },
    3: {
        "solver": "contrastive_loss", 
        "args": [], 
        "kwargs": dict(a = [[0.,0.],[1.,1.]], b = [[0.,0.],[2.,2.]], y = [1, 0], margin = 1.0), 
        "output": "[0.0, √2] → loss = [0.0, max(0,1-1.4142)²=0.0] → mean = 0.0", 
    },
}
