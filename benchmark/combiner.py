#!/usr/bin/env python3

import asyncio
import random
import string
import timeit

import numpy as np
import scipy

import git_theta

PARAMS = 2
RUNS = 50


def random_string(min_=20, max_=50):
    return "".join(
        random.choice(string.ascii_letters) for _ in range(random.randint(min_, max_))
    )


def random_tensor():
    return np.random.rand(4096, 4096)


def human_readable_bytes(num_bytes: int) -> str:
    # Use 1000, i.e. Giga not Gibi bytes
    scale = 1000
    if num_bytes < scale:
        return f"{num_bytes}B"
    units = ["kB", "MB", "GB", "TB"]
    u = -1
    while num_bytes >= scale:
        u += 1
        num_bytes = num_bytes / scale
    return f"{num_bytes:.2f}{units[u]}"


tar = git_theta.params.TarCombiner()
msgpack = git_theta.params.MsgPackCombiner()
serializer = git_theta.params.TensorStoreSerializer()


data = {random_string(): random_tensor() for _ in range(PARAMS)}


async def serialize(d):
    return {k: await serializer.serialize(v) for k, v in data.items()}


serialized_data = asyncio.run(serialize(data))

tar_comb = tar.combine(serialized_data)
tar_size = len(tar_comb)
print(
    f"Size of Tar combined files: {tar_size} bytes ({human_readable_bytes(tar_size)})."
)

msg_comb = msgpack.combine(serialized_data)
msg_size = len(msg_comb)
print(
    f"Size of MsgPack combined files: {msg_size} bytes ({human_readable_bytes(msg_size)})."
)

print(
    f"Saved {tar_size - msg_size} bytes ({human_readable_bytes(tar_size - msg_size)})."
)

tar_time = timeit.repeat("tar.combine(serialized_data)", number=RUNS, globals=globals())
tar_mean = np.mean(tar_time)
tar_std = np.std(tar_time, ddof=1)
print(f"Tar Time: {tar_mean:.4f} \u00B1 {tar_std:.4f} over {RUNS} runs")

msg_time = timeit.repeat(
    "msgpack.combine(serialized_data)", number=RUNS, globals=globals()
)
msg_mean = np.mean(msg_time)
msg_std = np.std(msg_time, ddof=1)
print(f"MsgPack Time: {msg_mean:.4f} \u00B1 {msg_std:.4f} over {RUNS} runs")

tt, p = scipy.stats.ttest_ind_from_stats(
    tar_mean, tar_std, RUNS, msg_mean, msg_std, RUNS, equal_var=False
)
if p < 0.05:
    print(f"The difference in speed is statistically significant, p({p}<0.05).")
else:
    print(f"Difference in speed is NOT statistically significant, p({p}>0.05).")
