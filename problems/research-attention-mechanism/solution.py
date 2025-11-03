import numpy as np
"""
Summarization:
The attention mechanism maps a query and a set of key-value pairs to an output, where all of them are vectors. The outputs are a weighted sum of the values, and the weights are determined by measuring how compatible each query-key pair is. In scaled dot product attention, This is done by computing the dot product of the query vector with all the key vectors, to obtain a logit for each query-key pair that expresses "how much should this value contribute to the output of this query?". These logits are then softmaxed across all keys in order to obtain weights that sum to 1 and lie in the range (0, 1).

However, during initialization, care must be taken to make sure that the standard deviations of the embeddings at initialization remain uniform across the depth of the network. This means that it is useful to make sure that the statistics of the attention outputs are similar to the 0-mean vector. Consider that the queries and keys may have their own respective variances and are of 0 mean. Say, the variance of the queries is Q and the variance of the keys is K. using the formula for the variance of the product of two random variables X and Y, we know that Var[XY] = Var[X] . Var[Y]. Observe that the query - key product is also exactly just that - a product of two random variables! Thus, the variance of the attention logits becomes dk * Q * K which means that, assuming Q and K are roughly the same and of unit std, the variance grew by a factor of dk! in order to maintain the variance and keep it from growing (which would saturate the softmax outputs), we need to scale the logits by a proportional factor of sqrt(dk).
"""
def _stable_softmax(x, axis=-1):
    """Numerically stable softmax implementation."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implement scaled dot-product attention mechanism.
    """
    Q, K, V = list(map(np.array, [Q, K, V]))
    scale = (1 / np.sqrt(K.shape[-1]))
    attn_logits = (Q @ K.T) * scale
    if mask is not None:
        attn_logits = np.where(np.tri(K.shape[-2]) == 0, -np.inf, attn_logits)
    attn_w = _stable_softmax(attn_logits, axis=-1)
    out = attn_w @ V
    return out, attn_w

