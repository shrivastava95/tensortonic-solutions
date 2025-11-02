# tensortonic-solutions

A collection of solutions / public test cases for [tensortonic.com](https://tensortonic.com)


# How to use
To start, clone the repo and implement your problems / tests in the `./problems` and `./tests` directory. The steps look something as follows:

1. Pick a tensortonic problem (e.g. [causal-masking](https://www.tensortonic.com/problems/causal-masking))
2. Copy the empty problem template to `./problems/causal-masking/solution.py`. Implement your solution.
```markdown
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
    #    this means that a key at position (j) can contribute to query at position (i) iff. j <= i
    #    notice that this is the inverse condition of how np.tri is constructed!
    masked_scores = np.where(np.tri(T), scores, mask_value)
    return masked_scores
```
3. Write the publicly available tests for causal-masking to `./tests/causal-masking/inputs.py`.
```markdown
tests = {
    1: {
        "solver": "apply_causal_mask",
        "args": [
            [[1,2,3],[4,5,6],[7,8,9]],
        ],
        "kwargs": {
            "mask_value": -1e9,
        },
        "output": [[1,-1e9,-1e9],[4,5,-1e9],[7,8,9]],
    },
}
```
4. Run `python run_tests.py --problem "causal-masking"`
