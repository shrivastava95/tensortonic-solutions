import numpy as np

tests = {
    1: {
        "solver": "classification_metrics", 
        "args": [], 
        "kwargs": dict(y_true=[0,1,2,2], y_pred=[0,1,0,2]), 
        "output": dict(accuracy=0.75, precision=0.75, recall=0.75, f1=0.75), 
    },
}
