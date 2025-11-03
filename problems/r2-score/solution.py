import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true, y_pred = list(map(np.array, [y_true, y_pred]))
    num = np.sum(np.square(y_true - y_pred))
    denum = np.sum(np.square(y_true - np.mean(y_true)))
    if denum == 0.0:
        frac = 0.0 if np.all(y_true == y_pred) else 1.0
    else:
        frac = num / denum
    r2 = 1 - frac
    return r2
