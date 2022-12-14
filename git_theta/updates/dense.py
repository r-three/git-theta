#!/usr/bin/env python3

import logging
from typing import Optional
from git_theta.updates import Update


class DenseUpdate(Update):
    """An update where all parameters are changed."""

    @property
    def name(self):
        return "dense"

    def calculate_update(self, repo, path, param_keys, param_metadata, param):
        return {"update": param}

    def apply(self, repo, path, param_keys, param_metadata):
        logging.debug(f"Reading Dense update for {'/'.join(param_keys)}")
        return self.read(param_metadata)["update"]
