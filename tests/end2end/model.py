"""Utility for creating models for testing."""

import argparse
import os
import random

import numpy as np
import scipy.sparse
import torch
from torch import nn
from torch.nn import functional as F

import git_theta

parser = argparse.ArgumentParser(description="Model building for Integration tests.")
parser.add_argument(
    "--action", choices=["init", "dense", "sparse", "low-rank", "ia3"], required=True
)
parser.add_argument("--seed", default=1337, type=int)
parser.add_argument("--model-name", default="model.pt")
parser.add_argument("--previous")


class TestingModel(nn.Module):
    """A small model for testing, weird architecture but tries to cover several pytorch paradigms."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(30, 10)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)
        self.layers = nn.Sequential(
            TestingLayer(10, 8, 6),
            nn.ReLU(),
            TestingLayer(6, 4, 2),
            nn.Tanh(),
            nn.LogSoftmax(dim=-1),
        )

    def __call__(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
    # TODO: Add a way to get numpy dtype from torch dtype easily.
    if t.ndim == 1:
        return np.random.uniform(size=t.shape).astype(np.float32)
    R = np.random.uniform(size=(*t.shape[:-1], rank)).astype(np.float32)
    C = np.random.uniform(size=(rank, *t.shape[-1:])).astype(np.float32)
    return {"A": R, "B": C}


def make_ia3_update(value):
    ia3 = np.random.randn(*value.shape).astype(np.float32)
    axes = (0, -1) if ia3.ndim > 3 else (-1,)
    ia3 = np.mean(ia3, axis=axes, keepdims=True)
    return {"ia3": ia3}


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
        update_handler = git_theta.updates.get_update_handler("sparse")
        previous = torch.load(args.previous)
        # Create a new version of the model and call it the "sparse" update
        sparse = TestingModel().state_dict()
        # Combine the sparse update and the old values to create the "new" model
        with_sparse = {name: value + previous[name] for name, value in sparse.items()}
        # Save the combined model to the persistent location for comparisons.
        torch.save(with_sparse, persistent_name)
        # Convert the sparse update into a sparse format.
        sparse_update = {}
        for name, value in sparse.items():
            update = update_handler.format_update(value.numpy())
            for k, v in update.items():
                sparse_update[f"{name}/{k}"] = torch.tensor(v)
        torch.save(sparse_update, "sparse-data.pt")

    elif args.action == "low-rank":
        previous = torch.load(args.previous)
        low_rank = 2
        lr_update = {
            name: low_rank_update(value, low_rank) for name, value in previous.items()
        }
        update_handler = git_theta.updates.get_update_handler("low-rank")
        # Just the low-rank data
        update_data = {}
        # The updated parameter values
        new_model = {}
        for name, update in lr_update.items():
            if isinstance(update, dict):
                # Formatter is simple, just assigned param1 and param2 to R and
                # C respectivly.
                update = update_handler.format_update(update["A"], update["B"])
                for k, v in update.items():
                    update_data[f"{name}/{k}"] = torch.tensor(v)
                new_model[name] = previous[name] + update["R"] @ update["C"]
            else:
                new_model[name] = previous[name] + update
                previous[name] = previous[name] + update
        # Checkpoint with the non-low-rank changes added
        torch.save(previous, args.model_name)
        # Checkpoint with the low-rank updates
        torch.save(update_data, "low-rank-data.pt")
        # Checkpoint with all changes added
        torch.save(new_model, persistent_name)

    elif args.action == "ia3":
        previous = torch.load(args.previous)
        ia3_update = {name: make_ia3_update(value) for name, value in previous.items()}
        update_handler = git_theta.updates.get_update_handler("ia3")
        # Just the ia3 data
        update_data = {}
        # The updated parameter values
        new_model = {}
        for name, update in ia3_update.items():
            if isinstance(update, dict):
                # Formatter is simple, just assigned param to ia3
                update = update_handler.format_update(update["ia3"])
                for k, v in update.items():
                    update_data[f"{name}/{k}"] = torch.tensor(v)
                new_model[name] = previous[name] * update["ia3"]
        # Save the new model
        torch.save(new_model, persistent_name)
        # save the ia3 data
        torch.save(update_data, "ia3-data.pt")

    print(persistent_name)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
