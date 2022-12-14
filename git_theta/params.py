"""Utilties to summarize parameters."""

import hashlib
import dataclasses
from collections import OrderedDict, defaultdict
import re
import tensorstore as ts
import io
import tarfile
import posixpath
import json

from git_theta import utils
from git_theta import git_utils
from file_or_name import file_or_name


class MetadataField:
    def to_serializable(self):
        d = dataclasses.asdict(self)
        for k, v in d.items():
            d[k] = v if v is None else str(v)
        return d


@dataclasses.dataclass(eq=True)
class LfsMetadata(MetadataField):
    version: str
    oid: str
    size: str

    @classmethod
    @property
    def name(cls):
        return "lfs_metadata"

    @property
    def lfs_pointer(self):
        return (
            f"version {self.version}\noid sha256:{self.oid}\nsize {self.size}".encode()
        )

    @classmethod
    def from_pointer(cls, pointer_file):
        match = re.match(
            "^version (?P<version>[^\s]*)\s*oid sha256:(?P<oid>[^\s]*)\s*size (?P<size>[0-9]*)$",
            pointer_file,
        )
        return cls(
            version=match.group("version"),
            oid=match.group("oid"),
            size=match.group("size"),
        )

    @classmethod
    def from_bytes(cls, b):
        return cls.from_pointer(git_utils.git_lfs_clean(b))


@dataclasses.dataclass(eq=True)
class TensorMetadata(MetadataField):
    shape: str
    dtype: str
    hash: str

    @classmethod
    @property
    def name(cls):
        return "tensor_metadata"

    @classmethod
    def from_tensor(cls, tensor):
        shape = str(tensor.shape)
        dtype = str(tensor.dtype)
        hash = hashlib.sha256(tensor.round(4).tobytes()).hexdigest()
        return cls(shape=shape, dtype=dtype, hash=hash)


@dataclasses.dataclass(eq=True)
class ThetaMetadata(MetadataField):
    update_type: str
    last_commit: str

    @classmethod
    @property
    def name(cls):
        return "theta_metadata"


class ParamMetadata(OrderedDict):
    def __init__(self, tensor_metadata, lfs_metadata, theta_metadata):
        self[TensorMetadata.name] = tensor_metadata
        self[LfsMetadata.name] = lfs_metadata
        self[ThetaMetadata.name] = theta_metadata

    @classmethod
    def from_metadata_dict(cls, d):
        tensor_metadata = TensorMetadata(**d[TensorMetadata.name])
        lfs_metadata = LfsMetadata(**d[LfsMetadata.name])
        theta_metadata = ThetaMetadata(**d[ThetaMetadata.name])
        return cls(tensor_metadata, lfs_metadata, theta_metadata)

    def to_serializable(self):
        for metadata_key, metadata in self.items():
            self[metadata_key] = metadata.to_serializable()
        return self

    @property
    def theta_metadata(self):
        return self.get(ThetaMetadata.name)

    @property
    def tensor_metadata(self):
        return self.get(TensorMetadata.name)

    @property
    def lfs_metadata(self):
        return self.get(LfsMetadata.name)


class Metadata(OrderedDict):
    @classmethod
    def from_metadata_dict(cls, d):
        flattened = utils.flatten(d, is_leaf=lambda v: LfsMetadata.name in v)
        for param_keys, param_metadata in flattened.items():
            flattened[param_keys] = ParamMetadata.from_metadata_dict(param_metadata)
        metadata = utils.unflatten(flattened)
        return cls(metadata)

    @classmethod
    @file_or_name
    def from_file(cls, file):
        metadata_dict = json.load(file)
        return cls.from_metadata_dict(metadata_dict)

    @file_or_name(file="w")
    def write(self, file):
        metadata_dict = self.to_serializable()
        json.dump(metadata_dict, file, indent=4)

    def flatten(self):
        return utils.flatten(self, is_leaf=lambda v: isinstance(v, ParamMetadata))

    def unflatten(self):
        return utils.unflatten(self)

    def diff(self, other):
        self_flattened = self.flatten()
        other_flattened = other.flatten()
        added = [
            self_flattened[k] for k in self_flattened.keys() - other_flattened.keys()
        ]
        removed = [
            other_flattened[k] for k in other_flattened.keys() - self_flattened.keys()
        ]
        modified = []
        for param_keys in set(self_flattened.keys()).intersection(
            other_flattened.keys()
        ):
            if (
                self_flattened[param_keys].tensor_metadata
                != other_flattened[param_keys].tensor_metadata
            ):
                modified.append(self_flattened[param_keys])

        return added, removed, modified

    def to_serializable(self):
        flattened = self.flatten()
        for param_keys, param_metadata in flattened.items():
            flattened[param_keys] = param_metadata.to_serializable()
        return flattened.unflatten()


class TensorSerializer:
    def serialize(self, tensor):
        raise NotImplementedError

    def deserialize(self, serialized_tensor):
        raise NotImplementedError


class TensorStoreSerializer(TensorSerializer):
    def serialize(self, param):
        store = ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "memory"},
                "metadata": {"shape": param.shape, "dtype": param.dtype.str},
                "create": True,
            },
        ).result()
        store.write(param).result()
        serialized_param = {
            k.decode("utf-8"): store.kvstore[k] for k in store.kvstore.list().result()
        }
        return serialized_param

    def deserialize(self, serialized_param):
        ctx = ts.Context()
        kvs = ts.KvStore.open("memory://", context=ctx).result()
        for name, contents in serialized_param.items():
            kvs[name] = contents

        store = ts.open(
            {"driver": "zarr", "kvstore": "memory://"}, context=ctx
        ).result()
        param = store.read().result()
        return param


class FileCombiner:
    def combine(self, files):
        raise NotImplementedError

    def split(self, file):
        raise NotImplementedError


class TarCombiner(FileCombiner):
    def combine(self, param_files):
        tarred_file = io.BytesIO()
        with tarfile.open(fileobj=tarred_file, mode="w") as archive:
            for param_name, param_files in param_files.items():
                for filename, file_bytes in param_files.items():
                    tarinfo = tarfile.TarInfo(posixpath.join(param_name, filename))
                    tarinfo.size = len(file_bytes)
                    archive.addfile(tarinfo, io.BytesIO(file_bytes))

        tarred_file.seek(0)
        return tarred_file.read()

    def split(self, tarred_file):
        file = io.BytesIO(tarred_file)
        param_files = defaultdict(dict)
        with tarfile.open(fileobj=file, mode="r") as archive:
            for archive_file in archive.getnames():
                param_name = posixpath.split(archive_file)[0]
                filename = posixpath.join(*posixpath.split(archive_file)[1:])
                param_files[param_name][filename] = archive.extractfile(
                    archive_file
                ).read()

        return param_files


class UpdateSerializer:
    def __init__(self, tensor_serializer, file_combiner):
        self.serializer = tensor_serializer
        self.combiner = file_combiner

    def serialize(self, update_params):
        serialized_params = {
            name: self.serializer.serialize(param)
            for name, param in update_params.items()
        }
        serialized_object = self.combiner.combine(serialized_params)
        return serialized_object

    def deserialize(self, serialized_object):
        serialized_params = self.combiner.split(serialized_object)
        update_params = {
            name: self.serializer.deserialize(serialized_param)
            for name, serialized_param in serialized_params.items()
        }
        return update_params


def get_update_serializer():
    # TODO: Right now this just returns a tensorstore/tar serializer but in the future we can implement other Serializers and/or support user plugins
    return UpdateSerializer(TensorStoreSerializer(), TarCombiner())
