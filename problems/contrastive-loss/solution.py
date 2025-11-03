import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a, b, y = list(map(np.array, [a, b, y]))
    y = y.reshape(-1, 1)
    validate = np.all(((y == 1).astype(int) + (y == 0).astype(int)) > 0)
    if not validate:
        return None
    if a.ndim == 1:
        a, b = list(map(lambda item: item[None, :], [a, b]))

    d = np.linalg.norm(a - b, axis=-1, keepdims=True)
    li = y * (d ** 2) + (1 - y) * (np.maximum(0, margin - d) ** 2)
    match reduction:
        case "mean":
            return np.mean(li)
        case "sum":
            return np.sum(li)
        case _:
            return None
    

