import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    loss = margin - y_true * y_score
    loss = np.where(loss > 0, loss, 0)
    match reduction:
        case "mean":
            loss = np.mean(loss).item()
        case "sum":
            loss = np.sum(loss).item()
    return loss
