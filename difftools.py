import onnx
import torch
import numpy as np
from onnx import numpy_helper as nh
import copy
import pickle
import hashlib
import os
from rich import print as rprint
from typing import *
from typeguard import check_type
from onnx_functions import AugumentedOnnxModel

Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)


class ModelUpdate:

    supported_update_types = {
        "dense": Tensor,
        "low_rank": Tuple[Tensor, Tensor],
    }

    def __init__(self, initializer_name: str, update_type: str, delta):
        self.initializer_name = initializer_name

        if update_type not in self.supported_update_types:
            raise NotImplementedError(
                f"{update_type} is not yet supported as an update type"
            )

        # Better version of isinstance(), can handle lists of types
        check_type("Update Delta", delta, self.supported_update_types[update_type])

        # Standardize to store as numpy variable
        if update_type == "dense" and hasattr(delta, "numpy"):
            delta = delta.numpy()
        if update_type == "low_rank":
            delta = [d.numpy() for d in delta if hasattr(d, "numpy")]

        self.update_type = update_type
        self.delta = delta


def create_diff_file(updates: Union[List[ModelUpdate], ModelUpdate]) -> str:

    if not isinstance(updates, List):
        updates = [updates]

    # Each entry in changes should be a 3-tuple (initializer, update_type, delta)
    update_dict = {
        update.initializer_name: (update.update_type, update.delta)
        for update in updates
    }
    hash = hashlib.md5(pickle.dumps(update_dict)).hexdigest()

    filename = f"{hash}.modeldiff"
    with open(filename, "wb") as content_file:
        pickle.dump(update_dict, content_file)

    return filename


def apply_diff_file(augumented_onnx_model, diffname):
    new_model = copy.deepcopy(augumented_onnx_model)

    with open(diffname, "rb") as infile:
        update_dict = pickle.load(infile)

    seen_initializers = set()
    for initializer_name, (update_type, delta) in update_dict.items():
        if initializer_name not in seen_initializers:
            seen_initializers.add(initializer_name)
        else:
            raise ValueError(
                f"An initializer can appear at most once in a modeldiff file: {initializer_name} appears multiple times."
            )

        new_model.update_initializer(initializer_name, update_type, delta)

    return new_model


def print_human_readable_diff(diff_hash):
    with open(f"{diff_hash}.modeldiff.content", "rb") as infile:
        update_dict = pickle.load(infile)
    rprint(update_dict)
