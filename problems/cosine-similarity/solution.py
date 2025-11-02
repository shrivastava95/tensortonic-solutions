import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a, b = [np.array(item) for item in [a, b]]
    norm_a = np.linalg.norm(a).item()
    norm_b = np.linalg.norm(b).item()
    if 0.0 in [norm_a, norm_b]:
        return 0.0
    dot = np.sum(a * b) / (norm_a * norm_b)
    return dot.item()
