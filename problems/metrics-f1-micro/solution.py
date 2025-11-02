def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    return sum(map(int, [yt == yp for yt, yp in zip(y_true, y_pred)])) / len(y_true)
