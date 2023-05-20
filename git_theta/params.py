"""Classes for serializing model updates."""

from abc import ABCMeta, abstractmethod

import msgpack
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


class MsgPackCombiner(FileCombiner):
    def combine(self, files):
        return msgpack.packb(files, use_bin_type=True)

    def split(self, file):
        return msgpack.unpackb(file, raw=False)


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
    # TODO: Right now this just returns a tensorstore/msgpack serializer but in
    # the future we can implement other Serializers and/or support user plugins
    return UpdateSerializer(TensorStoreSerializer(), MsgPackCombiner())
