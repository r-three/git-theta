#!/usr/bin/env python3

import logging
from typing import Optional
from git_theta.updates import Update, get_update


class SparseUpdate(Update):
    """An update where only some parameters are touched."""

    @property
    def name(self):
        return "sparse"

    def calculate_update(self, repo, path, param_keys, param_metadata, param):
        # TODO: Calculate a real sparse matrix and return the tensors in a dictionary
        update = get_update(param_metadata.theta_metadata.update_type)()
        prev_param = update.apply(repo, path, param_keys, param_metadata)
        return {"update": param - prev_param}

    def apply(self, repo, path, param_keys, param_metadata):
        logging.debug(f"Reading Sparse update for {'/'.join(param_keys)}")
        sparse_update = self.read(param_metadata)["update"]
        last_metadata = self.get_last_version(repo, path, param_keys, param_metadata)
        update = get_update(last_metadata.theta_metadata.update_type)()
        return sparse_update + update.apply(repo, path, param_keys, last_metadata)
