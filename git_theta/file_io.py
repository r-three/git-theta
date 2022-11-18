import tensorstore as ts
import io
import json
import logging

from file_or_name import file_or_name


def read_tensorstore_from_memory(files):
    ctx = ts.Context()
    kvs = ts.KvStore.open("memory://", context=ctx).result()
    for name, contents in files.items():
        kvs[name] = contents

    store = ts.open({"driver": "zarr", "kvstore": "memory://"}, context=ctx).result()
    return store.read().result()


def load_tracked_file(f):
    """
    Load tracked file

    Parameters
    ----------
    f : str
        path to file tracked by git-theta filter

    Returns
    -------
    np.ndarray
        numpy array stored in tracked file

    """
    logging.debug(f"Loading tracked file {f}")
    ts_file = ts.open(
        {
            "driver": "zarr",
            "open": True,
            "kvstore": {
                "driver": "file",
                "path": f,
            },
        }
    ).result()
    return ts_file.read().result()


def write_tracked_file(f, param):
    """
    Dump param into tracked file

    Parameters
    ----------
    f : str
        path to output file
    param : np.ndarray
        param value to dump to file

    """
    logging.debug(f"Dumping param to {f}")
    ts_file = ts.open(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": f,
            },
            "metadata": {
                "shape": param.shape,
                "dtype": param.dtype.str,
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()
    return ts_file.write(param).result()


@file_or_name
def load_staged_file(f):
    """
    Load staged file

    Parameters
    ----------
    f : str or file-like object
        staged file to load

    Returns
    -------
    dict
        staged file contents
    """
    return json.load(f)


@file_or_name(f="w")
def write_staged_file(f, contents):
    """
    Write staged file

    Parameters
    ----------
    f : str or file-like object
        file to write staged contents to
    contents : dict
        dictionary to write to staged file
    """
    json.dump(contents, f, indent=4)
