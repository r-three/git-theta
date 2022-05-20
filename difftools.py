import onnx
import torch
import numpy as np
from onnx import numpy_helper as nh

import copy
import pickle
import hashlib
import os
import json
from rich import print as rprint
from typing import *
from typeguard import check_type

import constants
from CommitHeadInfo import CommitHeadInfo
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


class DiffTools:
    def __init__(self):
        # Fetch Commit Head Info
        self.commit_head_info = CommitHeadInfo()

    @staticmethod
    def setup(model_name=None, reset_folders=False):
        if (reset_folders):
            print(f"Resetting diff tracking in {os.getcwd()} to a clean slate.")
            # Get path to model head (onnx file)
            commit_head_info = CommitHeadInfo()
            print(f"Deleting model head -: {commit_head_info.global_head_model_path}")
            os.remove(commit_head_info.global_head_model_path)

            if (os.path.exists(constants.COMMITS_FOLDER)):
                for file in sorted(os.listdir(constants.COMMITS_FOLDER)):
                    print(f"Deleting {file}...")
                    os.remove(os.path.join(constants.COMMITS_FOLDER, file))
        else:
            print(f"Setting up diff tracking in {os.getcwd()}")

        os.makedirs(constants.COMMITS_FOLDER, exist_ok=True)
        # Append _head for model head name
        model_head_name = model_name.split(".")[0] + "_head" + model_name.split(".")[1]
        commit_head_pointer = {
            "current_head": "",
            "global_head_model": os.path.join(constants.MODELS_FOLDER, model_head_name),
            "global_tail_model": os.path.join(constants.MODELS_FOLDER, model_name),
            "global_head": "",
            "global_tail": ""
        }
        with open(os.path.join(constants.COMMITS_FOLDER, "commit_head_pointer.json"), 'w') as f:
            f.write(json.dumps(commit_head_pointer, indent=4, sort_keys=True))

        print("Finished setup successfully!")


    def create_diff_file(self, updates: Union[List[ModelUpdate], ModelUpdate]) -> str:
        if not isinstance(updates, List):
            updates = [updates]

        # Fetch Global Head's diff index value
        if(self.commit_head_info.global_head):
            print("Loading Global Head File -: {}".format(self.commit_head_info.global_head))
            with open("{}.modeldiff".format(self.commit_head_info.global_head), "rb") as infile:
                global_head_diff = pickle.load(infile)

        # Each entry in changes should be a 3-tuple (initializer, update_type, delta)
        update_dict = {
            "diff_index": global_head_diff['diff_index'] + 1 if self.commit_head_info.global_head else 1,
            "previous_diff": self.commit_head_info.global_head,
            "next_diff": "",
            "initializers": {
                update.initializer_name: (update.update_type, update.delta) for update in updates
            }
        }
        print("New diff chain link (index, prev, next) -:", update_dict["diff_index"], update_dict["previous_diff"], update_dict["next_diff"])

        hash = hashlib.md5(pickle.dumps(update_dict)).hexdigest()
        new_diff_name = os.path.join(constants.COMMITS_FOLDER, f"{hash}.modeldiff")
        print(new_diff_name)
        # Add name of new diff file to update_dict for easier access
        update_dict["current_diff"] = new_diff_name

        # Update next diff pointer of current global head & persist
        if(self.commit_head_info.global_head):
            print("Updating pointers in chain")
            global_head_diff['next_diff'] = new_diff_name
            with open(self.commit_head_info.global_head, "wb") as content_file:
                    pickle.dump(global_head_diff, content_file)
            self.commit_head_info.update("global_head", new_diff_name)
        else:
            print("New Chain, setting pointers")
            self.commit_head_info.update("global_head", new_diff_name)
            self.commit_head_info.update("global_tail", new_diff_name)

        # Persist new diff
        print("Persisting new diff")
        with open(new_diff_name, "wb") as content_file:
                pickle.dump(update_dict, content_file)

        # Apply new diff to model and persist it at head
        print("Persisting new model")
        model_head = self.apply_diff_file(AugumentedOnnxModel(self.commit_head_info.global_head_model_path), self.commit_head_info.global_head)
        model_head.save_model(self.commit_head_info.global_head_model_path)

        return new_diff_name


    def apply_diff_file(self, augumented_onnx_model: AugumentedOnnxModel, diffname, inplace=False):
        if(inplace):
            new_model = augumented_onnx_model
        else:
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


    def checkout(self, checkpoint_file):
        """
        Sequentially applies diffs in order till checkpoint is reached. Order of diffs determined by diff_index of diff_file
        :param diff_file:
        :return: ONNX model state at checkpoint defined by diff_file
        """
        checkpoint_file = os.path.join(constants.COMMITS_FOLDER, checkpoint_file)
        with open(checkpoint_file, "rb") as infile:
            checkpoint_diff = pickle.load(infile)
        checkpoint_diff_index = checkpoint_diff['diff_index']

        with open(self.commit_head_info.global_head, "rb") as infile:
            global_head_diff = pickle.load(infile)

        with open(self.commit_head_info.global_tail, "rb") as infile:
            global_tail_diff = pickle.load(infile)

        if((global_head_diff['diff_index'] - checkpoint_diff_index) <= (checkpoint_diff_index - global_tail_diff['diff_index'])):
            pointer_direction = "prev_diff"
            model = AugumentedOnnxModel(self.commit_head_info.global_head_model_path)
            # Chain starts from diff before global head diff
            with open(global_head_diff[pointer_direction], "rb") as infile:
                diff = pickle.load(infile)
        else:
            pointer_direction = "next_diff"
            model = AugumentedOnnxModel(self.commit_head_info.global_tail_model_path)
            diff = global_tail_diff

        # Iterate and update until prior to the checkpoint file using apply_diff_file
        while(diff["current_diff"] != checkpoint_file):
            model = self.apply_diff_file(model, diff["current_diff"])
            with open(diff[pointer_direction], "rb") as infile:
                diff = pickle.load(infile)

        # Apply checkpoint diff file
        model = self.apply_diff_file(model, diff["current_diff"])

        return model


    def print_human_readable_diff(self, diff_hash):
        with open(f"{diff_hash}.modeldiff.content", "rb") as infile:
            update_dict = pickle.load(infile)
        rprint(update_dict)
