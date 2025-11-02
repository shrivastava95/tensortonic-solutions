import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    scores = np.array(scores)
    T = scores.shape[-1]
    mask = np.tri(T)
    # Explanation:
    # 1. np.tri(T) gives a lower triangular matrix of [T, T]. At index (i, j), if i < j then the value of the mask is 0 else it is 1.
    # 2. in causal self-attention, a position can attend to itself any future tokens.
    #    this means that a key at position (j) can contribute to query at position (i) iff. j <= i`
    #    notice that this is the inverse condition of how np.tri is constructed!
    masked_scores = np.where(np.tri(T), scores, mask_value)
    return masked_scores
