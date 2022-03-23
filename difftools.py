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

Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)


class ModelUpdate:

    supported_update_types = {
        "dense": Tensor,
        "low_rank": Tuple[Tensor, Tensor],
    }

    def __init__(self, initializer_name: str, update_type: str, value):
        self.initializer_name = initializer_name

        if update_type not in self.supported_update_types:
            raise NotImplementedError(
                f"{update_type} is not yet supported as an update type"
            )

        # Better version of isinstance(), can handle lists of types
        check_type("Update Value", value, self.supported_update_types[update_type])

        # Standardize to store as numpy variable
        if update_type == "dense" and hasattr(value, "numpy"):
            value = value.numpy()
        if update_type == "low_rank":
            value = [v.numpy() for v in value if hasattr(v, "numpy")]

        self.update_type = update_type
        self.value = value


def create_diff_file(updates: Union[List[ModelUpdate], ModelUpdate]) -> str:

    if not isinstance(updates, List):
        updates = [updates]

    # Each entry in changes should be a 3-tuple (initializer, update_type, vlaue)
    update_dict = {
        update.initializer_name: (update.update_type, update.value)
        for update in updates
    }
    hash = hashlib.md5(pickle.dumps(update_dict)).hexdigest()

    filename = f"{hash}.modeldiff"
    with open(filename, "wb") as content_file:
        pickle.dump(update_dict, content_file)

    return filename


def apply_diff_file(model, diffname):
    new_model = copy.deepcopy(model)

    with open(diffname, "rb") as infile:
        update_dict = pickle.load(infile)

    for initializer_name, (update_type, value) in update_dict.items():
        initializer = new_model.get_weight_by_name(initializer_name)

        if update_type == "dense":
            initializer.raw_data = value.tobytes()
        elif update_type == "low_rank":
            low_rank_product = value[0] @ value[1]
            updated_value = nh.to_array(initializer) + low_rank_product
            initializer.raw_data = updated_value.tobytes()
        else:
            raise NotImplementedError(
                "Only the dense and low-rank update types are currently supported"
            )

    return new_model


def print_human_readable_diff(diff_hash):
    with open(f"{diff_hash}.modeldiff.content", "rb") as infile:
        update_dict = pickle.load(infile)
    rprint(update_dict)
