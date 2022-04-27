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
import constants

import CommitHeadInfo
from onnx_functions import AugumentedOnnxModel

Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)


class ModelUpdate:
    # Fetch Commit Head Info
    commit_head_info = CommitHeadInfo()

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


def create_diff_file(updates: Union[List[ModelUpdate], ModelUpdate], commit_head_info) -> str:

    if not isinstance(updates, List):
        updates = [updates]

    # Fetch Global Head's diff index
    with open("{}.modeldiff".format(commit_head_info.global_head), "rb") as infile:
        global_head_diff = pickle.load(infile)

    # Each entry in changes should be a 3-tuple (initializer, update_type, delta)
    update_dict = {
        "diff_index": global_head_diff['diff_index'] + 1,
        "previous_diff": commit_head_info.global_head,
        "next_diff": None,
        "initializers": {
            update.initializer_name: (update.update_type, update.delta)
            for update in updates
        }
    }
    hash = hashlib.md5(pickle.dumps(update_dict)).hexdigest()

    filename = os.path.join(constants.COMMITS_FOLDER, f"{hash}.modeldiff")

    # Update next diff pointer of current global head & persist
    global_head_diff['next_diff'] = filename
    with open(commit_head_info.global_head, "wb") as content_file:
            pickle.dump(global_head_diff, content_file)

    # Persist new diff
    with open(filename, "wb") as content_file:
            pickle.dump(update_dict, content_file)

    return filename


def apply_diff_file(augumented_onnx_model, diffname):
    new_model = copy.deepcopy(augumented_onnx_model)

    with open(diffname, "rb") as infile:
        update_dict = pickle.load(infile)

    seen_initializers = set()
    for initializer_name, (update_type, delta) in update_dict["initializers"].items():
        if initializer_name not in seen_initializers:
            seen_initializers.add(initializer_name)
        else:
            raise ValueError(
                f"An initializer can appear at most once in a modeldiff file: {initializer_name} appears multiple times."
            )

        new_model.update_initializer(initializer_name, update_type, delta)

    return new_model


def checkout(diff_file, commit_head_info):
    """
    Sequentially applies diffs in order till checkpoint is reached. Order of diffs determined by diff_index of diff_file
    :param diff_file:
    :return: ONNX model state at checkpoint defined by diff_file
    """
    with open(diff_file, "rb") as infile:
        checkpoint_diff = pickle.load(infile)
    checkpoint_diff_index = checkpoint_diff['diff_index']

    with open("{}.modeldiff".format(commit_head_info.global_head), "rb") as infile:
        global_head_diff = pickle.load(infile)

    with open("{}.modeldiff".format(commit_head_info.global_tail), "rb") as infile:
        global_tail_diff = pickle.load(infile)

    if((global_head_diff['diff_index'] - checkpoint_diff_index) <= (checkpoint_diff_index - global_tail_diff['diff_index'])):
        pointer_direction = "prev_diff"
        curr_model = AugumentedOnnxModel(global_head)
        current_diff = global_head_diff
    else:
        pointer_direction = "next_diff"
        curr_model = AugumentedOnnxModel(global_tail)
        current_diff = global_tail_diff

    # Iterate and update using apply_diff_file
    while(current_diff[pointer_direction] != diff_file):
        curr_model = apply_diff_file(curr_model, current_diff[pointer_direction])
        with open("{}.modeldiff".format(current_diff[pointer_direction]), "rb") as infile:
            current_diff = pickle.load(infile)




def print_human_readable_diff(diff_hash):
    with open(f"{diff_hash}.modeldiff.content", "rb") as infile:
        update_dict = pickle.load(infile)
    rprint(update_dict)
