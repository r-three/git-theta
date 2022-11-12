import hashlib
import torch


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
    return str(torch.tensor(p).numpy().shape)


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
    return torch.tensor(p).numpy().dtype.str


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
    return hashlib.sha1(torch.tensor(p).numpy().tobytes()).hexdigest()
