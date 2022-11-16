"""Utilities for git theta"""


import operator as op
import os
import logging

from scipy import sparse
import numpy as np
from collections import defaultdict
from git_theta import checkpoints, file_io, constants

logging.basicConfig(
    level=logging.DEBUG, format="git-theta: [%(asctime)s] %(levelname)s - %(message)s"
)


def iterate_dict_leaves(d):
    """
    Generator that iterates through a dictionary and produces (leaf, keys) tuples where leaf is a dictionary leaf
    and keys is the sequence of keys used to access leaf. Dictionary is iterated in depth-first
    order with lexicographic ordering of keys.

    Example
    -------
    d = {'a': {'b': {'c': 10, 'd': 20, 'e': 30}}}
    iterate_dict_leaves(d) --> ((10, ['a','b','c']), (20, ['a','b','d']), (30, ['a','b','e']))

    Parameters
    ----------
    d : dict
        dictionary to iterate over

    Returns
    -------
    generator
        generates dict leaf, key path tuples
    """

    def _iterate_dict_leaves(d, prefix):
        for k, v in sorted(d.items(), key=op.itemgetter(0)):
            if isinstance(v, dict):
                yield from _iterate_dict_leaves(v, prefix + [k])
            else:
                yield (v, prefix + [k])

    return _iterate_dict_leaves(d, [])


def reconstruct_apply_update(updates_folder, update_file, update_type, base_value):
    """
    Reconstruct dense update from other update types and apply that to base_value
    Parameters
    ----------
    updates_folder
    update_file
    update_type
    base_value

    Returns
    -------
    updated_val
    """
    updated_val = base_value
    if update_type == constants.SPARSE_UPDATE:
        # Reshape since shape sometimes changes on load. e.g, actual shape (128,) on load becomes (1,128)
        update_val = (
            sparse.load_npz(os.path.join(updates_folder, update_file))
            .toarray()
            .reshape(base_value.shape)
        )
        np.copyto(updated_val, update_val, where=update_val != 0)

    return updated_val


def apply_param_updates(base_val, updates_folder):
    """
    Iterate through all update files and apply them to the dense base_val
    Parameters
    ----------
    base_val
    updates_folder

    Returns
    -------
    final_val
    """
    if not os.path.exists(updates_folder) or not os.listdir(updates_folder):
        logging.debug(f"No updates in {updates_folder}")
        return base_val

    final_val = base_val
    with open(
        os.path.join(updates_folder, constants.THETA_UPDATES_METADATA_FILE), "r"
    ) as f:
        update_apply_order = [line.strip("\n").split(",") for line in f.readlines()]

    logging.info(f"{len(update_apply_order)} updates found in {updates_folder}")
    for update_file, update_type in update_apply_order:
        logging.info(f"Applying update from {update_file}")
        final_val = reconstruct_apply_update(
            updates_folder, update_file, update_type, final_val
        )

    return final_val


def get_prev_param_dict(theta_model_dir):
    """
    Retrieves the model_dict for the latest committed model from values in .git_theta
    Parameters
    ----------
    theta_model_dir

    Returns
    -------
    model_dict
    """
    model_dict = defaultdict(dict)
    for leaf_dir, keys in checkpoints.iterate_dir_leaves(theta_model_dir):
        param_file = os.path.join(leaf_dir, constants.THETA_PARAMS_FOLDER)
        # logging.debug(f"Populating model parameter {'/'.join(keys)}")
        logging.info(f"Populating model parameter {'/'.join(keys)}")
        d = model_dict
        for k in keys[:-1]:
            d = d[k]
        base_val = file_io.load_tracked_file(param_file)
        final_val = apply_param_updates(
            base_val, os.path.join(leaf_dir, constants.THETA_UPDATES_FOLDER)
        )
        d[keys[-1]] = final_val

    return model_dict


def get_sparse_diff(prev_param_dict, param_dict):
    """
    Retrieve parameters which have changed since previous model state and sparsify them
    Parameters
    ----------
    prev_param_dict
    param_dict

    Returns
    -------
    sparse_diff_dict
    """
    sparse_diff_dict = defaultdict(dict)
    for param in prev_param_dict:
        if param not in param_dict:
            logging.debug(f"{param} not present in latest model state. Skipping.")
            continue

        if not np.allclose(prev_param_dict[param], param_dict[param]):
            diff_val = param_dict[param].copy()
            diff_val[np.isclose(prev_param_dict[param], param_dict[param])] = 0
            sparse_diff_dict[param] = sparse.csr_matrix(diff_val)
        else:
            logging.debug(f"{param} is unchanged. Skipping.")

    return sparse_diff_dict
