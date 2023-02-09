"""Tool to verify that checkpoints match."""

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Compare checkpoints for testing.")
parser.add_argument("--new-model", help="The path to the new model.")
parser.add_argument(
    "--old-model", help="The path to the model we are comparing aganist."
)
parser.add_argument("--compare")


def get_compare_function(compare):
    """Eventually add configurable comaprison functions?"""

    def _cmp(a, b):
        return np.allclose(a.numpy(), b.numpy())

    return _cmp


def main(args):
    old = torch.load(args.old_model)
    new = torch.load(args.new_model)

    compare = get_compare_function(args.compare)

    if old.keys() != new.keys():
        raise ValueError(
            f"Parameter keys differ. Got: {args.old_model} -> "
            f"{sorted(old.keys())} and {args.new_model} -> {sorted(new.keys())}"
        )

    mismatched = set()
    for name, value in new.items():
        if not compare(value, old[name]):
            mismatched.add(name)

    if mismatched:
        raise ValueError(f"Parameters: {sorted(mismatched)} differ between models.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
