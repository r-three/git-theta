import tensorstore as ts
import io
import json
import logging

from file_or_name import file_or_name


def load_tracked_file_from_memory(files):
    ctx = ts.Context()
    kvs = ts.KvStore.open("memory://", context=ctx).result()
    for name, contents in files.items():
        kvs[name] = contents

    store = ts.open({"driver": "zarr", "kvstore": "memory://"}, context=ctx).result()
    return store.read().result()


def write_tracked_file_to_memory(param):
    store = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "memory"},
            "metadata": {"shape": param.shape, "dtype": param.dtype.str},
            "create": True,
        },
    ).result()
    store.write(param).result()
    tensor_files = {
        k.decode("utf-8"): store.kvstore[k] for k in store.kvstore.list().result()
    }
    return tensor_files


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
