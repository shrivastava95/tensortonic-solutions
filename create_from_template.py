import os
from argparse import ArgumentParser

def make_file_safely_with_content(filepath, content, replace_existing=False):
    assert not os.path.exists(filepath) or replace_existing, f"{filepath} exists. please pass the `--overwrite` argument to replace it by force."
    parent_dir = os.path.dirname(filepath)
    os.makedirs(parent_dir, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"wrote content to file: {filepath}")
        

parser = ArgumentParser()
parser.add_argument("--problem", required=True, help="the name of the problem you want to start with a template for.")
parser.add_argument("--overwrite", action="store_true", help="whether to overwrite existing problem files, if found conflict.")
args = parser.parse_args()

problem_path = f"problems/{args.problem}/solution.py"
problem_template = """"""

test_path = f"tests/{args.problem}/inputs.py"
test_template = """import numpy as np

tests = {
    1: {
        "solver": , # put the name of the function that the input is passed to here.
        "args": , # put any input args here
        "kwargs": , # put any input kwargs here
        "output": , # put your public / custom test case outputs here
    }
}
"""

for payloads in [[problem_path, problem_template, args.overwrite],[test_path,test_template, args.overwrite]]:
    make_file_safely_with_content(*payloads)



