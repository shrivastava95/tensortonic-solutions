import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.array(g)
    g_norm = np.linalg.norm(g)
    if max_norm <= 0 or g_norm == 0:
        return g
    if g_norm.item() > max_norm:
        g = g * (max_norm / g_norm)
    return g
