import math
import numpy as np
from argparse import ArgumentParser
from importlib import import_module

parser = ArgumentParser()
parser.add_argument("--problem", required=True, help="Enter the name of the problem here.")
args = parser.parse_args()

if __name__ == "__main__":
    tests = import_module(f"tests.{args.problem}.inputs").tests
    for test_id, test in tests.items():
        solution = import_module(f"problems.{args.problem}.solution")
        solver = getattr(solution, test["solver"], None)
        inputs = {
                "args": test["args"],
                "kwargs" : test["kwargs"],
        }
        output = test["output"]
        assert solver is not None, f"error: problem {args.problem} with test case {test_id} has failed - solver={test['solver']} cannot be found in tests/{args.problem}/inputs.py"
        print()
        print(f"Solving test {test_id}...")
        candidate = solver(*inputs["args"], **inputs["kwargs"])
        print(f"\nexpected output: {np.array(output)}")
        try:
            print(f"\nyour output: {np.array(candidate)}")
        except:
            print(f"\nyour output: {candidate}")
        print()
        print(f"-"*50)
