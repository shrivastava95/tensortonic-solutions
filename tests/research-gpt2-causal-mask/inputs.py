import numpy as np

output_1 = """Output:
[[ True, False, False, False],
 [ True, True, False, False],
 [ True, True, True, False],
 [ True, True, True, True]]

Explanation:
Position 0 can see positions 0
Position 1 can see positions 0, 1
Position 2 can see positions 0, 1, 2
Position 3 can see positions 0, 1, 2, 3
"""

tests = {
    1: {
        "solver": "causal_mask",
        "args": [],
        "kwargs": dict(seq_len=4),
        "output": """""",
    }
}
