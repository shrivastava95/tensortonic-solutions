import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(0, seq_len).reshape(-1, 1)
    i = ( np.arange(0, d_model).reshape(1, -1) // 2)
    angle = pos * np.exp( - np.log(base) * ( 2 * i / d_model))
    pe = np.zeros(angle.shape)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

def add_positional_encoding(x, base=10000.0):
    """
    Add PE to input x of shape (B, T, d_model); return same shape.
    """
    seq_len, d_model = x.shape[-2:]
    pe = positional_encoding(seq_len, d_model, base)
    x_pe = x + pe
    return x_pe
