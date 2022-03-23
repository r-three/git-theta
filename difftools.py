import onnx
import torch
import numpy as np
from onnx import numpy_helper as nh
import copy
import pickle
import hashlib

# Diff files are pairs of files: the human readable diff, with names of the initializers,
# and then a pickled dictionary of intializers
def create_diff_file(filename, initializer, update_type, value):
    hr_line = f"+{initializer}, {update_type}\n"
    update_dict = {initializer: value}
    hash = hashlib.md5(pickle.dumps(update_dict) + hr_line.encode()).hexdigest()

    with open(f"{filename}.modeldiff.index", "w") as index_file:
        index_file.write(hash + "\n---\n")
        index_file.write(hr_line)
    with open(f"{filename}.modeldiff.content", "wb") as content_file:
        pickle.dump(update_dict, content_file)


def apply_diff_file(model, filename):
    new_model = copy.deepcopy(model)

    with open(f"{filename}.modeldiff.index", "r") as index_file:
        lines = list(filter(len, index_file.read().split("\n")))

    def to_apply(x):
        return x[0] == "+"

    updates = filter(to_apply, lines[2:])
    for update in updates:
        initializer_name, update_type = update[1:].split(", ")
        initializer = new_model.get_weight_by_name(initializer_name)

        with open(f"{filename}.modeldiff.content", "rb") as content_file:
            update_dict = pickle.load(content_file)
        initializer.raw_data = update_dict[initializer_name].numpy().tobytes()

    return new_model
