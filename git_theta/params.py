"""Utilties to summarize parameters."""

import hashlib
import numpy as np
import dataclasses
from collections import OrderedDict
import re

from git_theta import utils
from git_theta import git_utils


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
        hash = hashlib.sha256(tensor.round(6).tobytes()).hexdigest()
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
