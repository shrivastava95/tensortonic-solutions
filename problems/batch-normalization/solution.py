import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)

    dims = x.shape
    if x.ndim == 2:
        x_hat = (x - np.mean(x, axis=0)) / np.sqrt(np.std(x, axis=0) ** 2 + eps)
    if x.ndim == 4:
        B, C, H, W = dims
        x_r = x.transpose(0, 2, 3, 1).reshape(-1, C) # B, H, W, C
        x_r_hat = (x_r - np.mean(x_r, axis=0)) / np.sqrt(np.std(x_r, axis=0) ** 2 + eps)
        x_hat = x_r_hat.reshape(B, H, W, C).transpose(0, 3, 1, 2)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    
    y = x_hat * gamma + beta
    return y
