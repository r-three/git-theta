"""Classes representing checkpoint metadata files"""

import hashlib
import dataclasses
from collections import OrderedDict
import re
import json

from git_theta import git_utils, utils
from file_or_name import file_or_name


@dataclasses.dataclass(eq=True)
class MetadataField:
    def serialize(self):
        return dataclasses.asdict(self, dict_factory=OrderedDict)


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
            f"version {self.version}\noid sha256:{self.oid}\nsize {self.size}\n".encode(
                "utf-8"
            )
        )

    @classmethod
    def from_pointer(cls, pointer_contents):
        match = re.match(
            "^version (?P<version>[^\s]+)\noid sha256:(?P<oid>[0-9a-f]{64})\nsize (?P<size>[0-9]+)\n$",
            pointer_contents,
        )
        if match is None:
            raise ValueError(f"Failed to parse pointer file {pointer_contents}")
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


@dataclasses.dataclass(eq=True)
class ParamMetadata(MetadataField):
    tensor_metadata: TensorMetadata
    lfs_metadata: LfsMetadata
    theta_metadata: ThetaMetadata

    @classmethod
    def from_metadata_dict(cls, d):
        tensor_metadata = TensorMetadata(**d[TensorMetadata.name])
        lfs_metadata = LfsMetadata(**d[LfsMetadata.name])
        theta_metadata = ThetaMetadata(**d[ThetaMetadata.name])
        return cls(tensor_metadata, lfs_metadata, theta_metadata)


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

    @classmethod
    def from_commit(cls, repo, path, commit_hash):
        obj = git_utils.get_file_version(repo, path, commit_hash)
        if obj is None:
            return cls()
        else:
            return cls.from_file(obj.data_stream)

    @file_or_name(file="w")
    def write(self, file):
        metadata_dict = self.serialize()
        json.dump(metadata_dict, file, indent=4)

    def flatten(self):
        return utils.flatten(self, is_leaf=lambda v: isinstance(v, ParamMetadata))

    def unflatten(self):
        return utils.unflatten(self)

    def diff(self, other):
        self_flattened = self.flatten()
        other_flattened = other.flatten()
        added = Metadata(
            {
                k: self_flattened[k]
                for k in self_flattened.keys() - other_flattened.keys()
            }
        ).unflatten()
        removed = Metadata(
            {
                k: other_flattened[k]
                for k in other_flattened.keys() - self_flattened.keys()
            }
        ).unflatten()
        modified = Metadata()
        for param_keys in set(self_flattened.keys()).intersection(
            other_flattened.keys()
        ):
            if (
                self_flattened[param_keys].tensor_metadata
                != other_flattened[param_keys].tensor_metadata
            ):
                modified[param_keys] = self_flattened[param_keys]

        modified = modified.unflatten()
        return added, removed, modified

    def serialize(self):
        flattened = self.flatten()
        for param_keys, param_metadata in flattened.items():
            flattened[param_keys] = param_metadata.serialize()
        return flattened.unflatten()
