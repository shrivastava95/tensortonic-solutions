import numpy as np

def _softmax(x, dim=-1):
    x = np.array(x)
    assert isinstance(dim, int)
    assert x.shape
    x = x - np.max(x, axis=dim, keepdims=True)
    x_exp = np.exp(x)  
    out = x_exp / np.sum(x_exp, axis=dim, keepdims=True)
    return out


def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1, Z2 = list(map(np.array, [Z1, Z2]))
    S = Z1 @ Z2.T / temperature
    L = - np.mean(np.log(np.diag(_softmax(S))))
    return L
