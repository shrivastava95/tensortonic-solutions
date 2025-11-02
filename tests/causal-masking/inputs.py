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
    2: {
        "solver": "apply_causal_mask",
        "args": [
            [[[1,2,3],[4,5,6],[7,8,9]]],
        ],
        "kwargs": {
            "mask_value": -1e9,
        },
        "output": [[[1,-1e9,-1e9],[4,5,-1e9],[7,8,9]]],
    }
}
