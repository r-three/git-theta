#!/usr/bin/env python3

import copy
import os

import torch

import git_theta

print("loading old model")
model = torch.load("og_model.pt")
print("making a copy of the model with a single value change")
updated_model = copy.deepcopy(model)
updated_model["layers.0.hidden.weight"] = torch.rand(
    *updated_model["layers.0.hidden.weight"].shape
)

print("committing the same model to different paths")
model_1_sha = git_theta.save(model, "model_1.pt", "commit first model")
model_2_sha = git_theta.save(model, "model_2.pt", "commit second model")
print("committing the changed model to the same path.")
model_1_1_sha = git_theta.save(
    updated_model, "model_1.pt", "committing changed model to the same path."
)

print("Making sure the models we not saved to disk.")
assert not os.path.exists("model_1.pt")
assert not os.path.exists("model_2.pt")

print("loading the model from git-theta directly.")
m1 = git_theta.load(model_1_sha, "model_1.pt")
m11 = git_theta.load(model_1_1_sha, "model_1.pt")
m2 = git_theta.load(model_2_sha, "model_2.pt")

print("saving the models to disk to inspect them later.")
torch.save(m1, "should_match_1.pt")
torch.save(m2, "should_match_2.pt")
torch.save(m11, "no_match.pt")
