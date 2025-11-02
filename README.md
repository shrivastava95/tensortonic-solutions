# tensortonic-solutions

A collection of solutions / public test cases for [tensortonic.com](https://tensortonic.com)


# How to use
To start, clone the repo and implement your problems / tests in the `./problems` and `./tests` directory. The steps look something as follows:

1. Pick a tensortonic problem (e.g. [causal-masking](https://www.tensortonic.com/problems/causal-masking))
2. Copy the empty problem template to `./problems/causal-masking/solution.py`. Implement your solution here.
```markdown

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
