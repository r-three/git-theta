import tensorstore as ts
import io
import json
import logging

def load_tracked_file(f):
    """
    Load tracked file 

    Parameters
    ----------
    f : str
        path to file tracked by git-cml filter

    Returns
    -------
    np.ndarray
        numpy array stored in tracked file

    """
    logging.debug(f"Loading tracked file {f}")
    ts_file = ts.open({
        'driver': 'zarr',
        'open': True,
        'kvstore': {
            'driver': 'file',
            'path': f,
        },
    }).result()
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
    ts_file = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f,
        },
        'metadata': {
            'shape': param.shape,
            'dtype': param.dtype.str,
        },
        'create': True,
        'delete_existing': True
    }).result()
    return ts_file.write(param).result()


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
    if isinstance(f, io.IOBase):
        return json.load(f)
    else:
        with open(f, "r") as f:
            return json.load(f)


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
    if isinstance(f, io.IOBase):
        json.dump(contents, f, indent=4)
    else:
        with open(f, "w") as f:
            json.dump(contents, f, indent=4)
