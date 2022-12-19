"""Classes for serializing model updates."""

from collections import defaultdict
import tensorstore as ts
import io
import tarfile
import posixpath


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
                    # N.b. posixpath is used to create the "virtual path" in the tar file to each underlying parameter file
                    # Ensures consistent reading/writing of virtual paths across platforms
                    tarinfo = tarfile.TarInfo(posixpath.join(param_name, filename))
                    tarinfo.size = len(file_bytes)
                    archive.addfile(tarinfo, io.BytesIO(file_bytes))

        tarred_file.seek(0)
        return tarred_file.read()

    def split(self, tarred_file):
        file = io.BytesIO(tarred_file)
        param_files = defaultdict(dict)
        with tarfile.open(fileobj=file, mode="r") as archive:
            for file_in_archive in archive.getnames():
                param_name, filename = posixpath.split(file_in_archive)
                param_files[param_name][filename] = archive.extractfile(
                    file_in_archive
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
