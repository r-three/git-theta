"""Utilties to summarize parameters."""

import hashlib
import numpy as np


def get_shape_str(p):
    """
    Parameters
    ----------
    p : list or scalar

    Returns
    -------
    str
        shape of parameter
    """
    return str(np.asarray(p).shape)


def get_dtype_str(p):
    """
    Parameters
    ----------
    p : list or scalar

    Returns
    -------
    str
        dtype of parameter
    """
    return np.asarray(p).dtype.str


def get_hash(p):
    """
    Parameters
    ----------
    p : list or scalar

    Returns
    -------
    str
        hash of parameter bytes
    """
    return hashlib.sha1(np.asarray(p).tobytes()).hexdigest()
