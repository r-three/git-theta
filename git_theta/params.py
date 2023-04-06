"""Classes for serializing model updates."""

import io
import posixpath
import tarfile
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import tensorstore as ts


class TensorSerializer(metaclass=ABCMeta):
    """Serialize/Deserialize tensors."""

    @abstractmethod
    async def serialize(self, tensor):
        """Convert a tensor to bytes."""

    @abstractmethod
    async def deserialize(self, serialized_tensor):
        """Convert bytes to a tensor object."""


class TensorStoreSerializer(TensorSerializer):
    async def serialize(self, tensor):
        store = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "memory"},
                "metadata": {"shape": tensor.shape, "dtype": tensor.dtype.str},
                "create": True,
            },
        )
        await store.write(tensor)
        serialized_param = {
            k.decode("utf-8"): store.kvstore[k] for k in await store.kvstore.list()
        }
        return serialized_param

    async def deserialize(self, serialized_tensor):
        ctx = ts.Context()
        kvs = await ts.KvStore.open("memory://", context=ctx)
        for name, contents in serialized_tensor.items():
            kvs[name] = contents

        store = await ts.open({"driver": "zarr", "kvstore": "memory://"}, context=ctx)
        param = await store.read()
        return param


class FileCombiner(metaclass=ABCMeta):
    """Combine and Split serialized tensors, enables single blob processing for multiple tensors."""

    @abstractmethod
    def combine(self, files):
        """Combine multiple byte steams into one."""

    @abstractmethod
    def split(self, file):
        """Split a combined byte stream into original bytes."""


class TarCombiner(FileCombiner):
    def combine(self, files):
        tarred_file = io.BytesIO()
        with tarfile.open(fileobj=tarred_file, mode="w") as archive:
            for param_name, param_file in files.items():
                for filename, file_bytes in param_file.items():
                    # N.b. posixpath is used to create the "virtual path" in the tar file to each underlying parameter file
                    # Ensures consistent reading/writing of virtual paths across platforms
                    tarinfo = tarfile.TarInfo(posixpath.join(param_name, filename))
                    tarinfo.size = len(file_bytes)
                    archive.addfile(tarinfo, io.BytesIO(file_bytes))

        tarred_file.seek(0)
        return tarred_file.read()

    def split(self, file):
        file = io.BytesIO(file)
        param_files = defaultdict(dict)
        with tarfile.open(fileobj=file, mode="r") as archive:
            for file_in_archive in archive.getnames():
                param_name, filename = posixpath.split(file_in_archive)
                param_files[param_name][filename] = archive.extractfile(
                    file_in_archive
                ).read()
        return param_files


class Serializer(metaclass=ABCMeta):
    """Serialize/Deserialize parameters, even when represented with multiple tensors."""

    @abstractmethod
    async def serialize(self, params):
        """Serialize parameter."""

    @abstractmethod
    async def deserialize(self, serialized):
        """Deserialize parameter."""


class UpdateSerializer(Serializer):
    def __init__(self, tensor_serializer, file_combiner):
        self.serializer = tensor_serializer
        self.combiner = file_combiner

    async def serialize(self, params):
        serialized_params = {
            name: await self.serializer.serialize(param)
            for name, param in params.items()
        }
        return self.combiner.combine(serialized_params)

    async def deserialize(self, serialized):
        serialized_params = self.combiner.split(serialized)
        update_params = {
            name: await self.serializer.deserialize(serialized_param)
            for name, serialized_param in serialized_params.items()
        }
        return update_params


def get_update_serializer():
    # TODO: Right now this just returns a tensorstore/tar serializer but in the future we can implement other Serializers and/or support user plugins
    return UpdateSerializer(TensorStoreSerializer(), TarCombiner())
