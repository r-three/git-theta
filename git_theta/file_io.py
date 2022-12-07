import tensorstore as ts
import io
import json
import logging
import zipfile
import tarfile

from file_or_name import file_or_name


def untar_tracked_file_in_memory(file_bytes):
    file = io.BytesIO(file_bytes)
    untarred_file = {}
    with tarfile.open(fileobj=file, mode="r") as archive:
        for archive_file in archive.getnames():
            untarred_file[archive_file] = archive.extractfile(archive_file).read()

    return untarred_file


def load_tracked_file_from_memory(file):
    files = untar_tracked_file_in_memory(file)
    ctx = ts.Context()
    kvs = ts.KvStore.open("memory://", context=ctx).result()
    for name, contents in files.items():
        kvs[name] = contents

    store = ts.open({"driver": "zarr", "kvstore": "memory://"}, context=ctx).result()
    return store.read().result()


def tar_tracked_files_in_memory(files):
    tarred_file = io.BytesIO()
    with tarfile.open(fileobj=tarred_file, mode="w") as archive:
        for filename, file_bytes in files.items():
            tarinfo = tarfile.TarInfo(filename)
            tarinfo.size = len(file_bytes)
            archive.addfile(tarinfo, io.BytesIO(file_bytes))
    tarred_file.seek(0)
    return tarred_file.read()


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
    file = tar_tracked_files_in_memory(tensor_files)
    return file


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
