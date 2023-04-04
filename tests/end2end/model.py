"""Utility for creating models for testing."""

import argparse
import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


parser = argparse.ArgumentParser(description="Model building for Integration tests.")
parser.add_argument(
    "--action", choices=["init", "dense", "sparse", "low-rank"], required=True
)
parser.add_argument("--seed", default=1337, type=int)
parser.add_argument("--model-name", default="model.pt")
parser.add_argument("--previous")


class TestingModel(nn.Module):
    """A small model for testing, weird architecture but tries to cover several pytorch paradigms."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(30, 10)
        self.layers = nn.Sequential(
            TestingLayer(10, 8, 6),
            nn.ReLU(),
            TestingLayer(6, 4, 2),
            nn.Tanh(),
            nn.LogSoftmax(dim=-1),
        )

    def __call__(self, x):
        return self.layers(x)


class TestingLayer(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.hidden = nn.Linear(n_in, n_hidden)
        self.output = nn.Linear(n_hidden, n_out)
        self.highway_like = nn.Linear(n_in, n_out)

    def __call__(self, x):
        y = self.hidden(x)
        y = F.relu(y)
        y = self.output(y)
        y += self.highway_like(x)
        return y


def low_rank_update(t, rank):
    if t.ndim == 1:
        return np.random.uniform(size=t.shape, dtype=t.dtype)
    R = np.random.uniform(size=(*t.shape[:-1], rank), dtype=t.dtype)
    C = np.random.uniform(size=(rank, *t.shape[-1:]), dtype=t.dtype)
    return R @ C


def main(args):
    # Set seeds and enable determinism
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    file_name, ext = os.path.splitext(args.model_name)
    persistent_name = f"{file_name}-{args.action}-{args.seed}{ext}"
    if args.previous is None:
        args.previous = args.model_name

    if args.action == "init" or args.action == "dense":
        model = TestingModel().state_dict()
        torch.save(model, args.model_name)
        torch.save(model, persistent_name)
    elif args.action == "sparse":
        previous = torch.load(args.previous)
        model = TestingModel().state_dict()
        sparse = {name: value + previous[name] for name, value in model.items()}
        torch.save(sparse, args.model_name)
        torch.save(sparse, persistent_name)

    elif args.action == "low-rank":
        previous = torch.load(args.previous)
        low_rank = 2
        new_model = {
            name: value + low_rank_update(value, low_rank)
            for name, value in previous.items()
        }
        torch.save(new_model, args.model_name)
        torch.save(new_model, persistent_name)

    print(persistent_name)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
