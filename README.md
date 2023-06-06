# Git-Theta

<center><img src="https://user-images.githubusercontent.com/417568/229904559-d61d710c-7986-4a07-a405-d86b196f5046.png" width="50"></center>

Git-Theta is a Git extension for collaborative, continual, and communal development of machine learning models.

Version control systems like Git enable large distributed teams to collaborate on shared codebases by tracking changes over time and providing tools for merging changes from multiple sources.
Git-Theta is a Git extension that aims to provide similar functionality for machine learning model checkpoints by *efficiently* and *meaningfully* track a model's version history natively through Git.
Specifically, rather than treating the checkpoint as a blob of data (as done by other systems for tracking models with Git), Git-Theta
- atomically tracks each parameter "group" (e.g. a weight matrix or bias vector in a neural network)
- tracks dense or communication-efficient updates like [low-rank](https://arxiv.org/abs/2106.09685) or [sparse](https://arxiv.org/abs/2111.09839) changes to parameter groups
- allows models to be merged automatically or manually
- displays meaningful "diffs" by showing which parameter groups have changed
- supports checkpoint formats from most popular machine learning frameworks
- enables easy extension of update types, merging methods, and checkpoint formats through a plugin system

Git-Theta is currently under active development and should be used with caution.
For feature discussions and debugging help, please join the #git-theta stream in the [CCCML Zulip community](https://bit.ly/cccml-community).
If you use Git-Theta as part of a published research project, please cite [our paper](https://arxiv.org/TODO).

# Quick Start

## Installing Git LFS
Download and install Git LFS using the instructions from [the Git LFS website](https://git-lfs.github.com).

## Installing Git-Theta
1) Install the git-theta Python package:
```bash
pip install git-theta
```

By default, installing `git-theta` with `pip` will not install any of the supported machine learning frameworks (PyTorch, TensorFlow, etc.).
If you want to install the framework you intend to use when installing `git-theta`, you can specify it when installing (e.g. by running `pip install git-theta[pytorch]` for PyTorch).

2) Configure Git to use Git-Theta when tracking model checkpoints:
```bash
git theta install
```

## Tracking a model

Say you have a codebase for training a model along with the model's checkpoint:

```bash
my_codebase
├── model.pt
└── train.py
```

Git-Theta allows you to use Git to track the changes to your code ***and*** your model's parameters in tandem.
To use Git-Theta to track the model checkpoint, first run

```bash
git theta track model.pt
```

This will create or update the `.gitattributes` file that tells Git to use Git-Theta to handle the checkpoint file.
You can then add and commit the `.gitattributes` file:

```bash
git add .gitattributes
git commit
```

After tracking the model, you can regular Git commands (`add`, `commit`, `push`, `pull`, `checkout`, `status`, `diff`, etc.) as if the checkpoint file was any other file.
To add and commit the initial version of the checkpoint, simply run

```bash
git add model.pt
git commit
```

## Storing updates efficiently

Additionally, `git theta add` can be used instead of `git add` to provide optional extra information, including e.g.\ the checkpoint format with `--checkpoint-type`, the [`Update`](https://github.com/r-three/git-theta/tree/main/git_theta/updates) used to update parameters with `--update-type`, and the location of auxiliary information/data for the update with `--update-path`.
For example, if the model was updated using using [LoRA](https://arxiv.org/abs/2106.09685), the low-rank factors can be efficiently stored by Git-Theta by running:

```bash
# After training with LoRA and saving the factors to updates.pt...
git theta add model.pt --update-type low-rank --update-path updates.pt
git commit
```

## Merging updates

Git-Theta can also handle merging of models trained with differing updates.
For example, if an existing model is further trained on a new branch called `alternate-training`:

```bash
git checkout -b alternate-training
# After performing training...
git add model.pt
git commit
```

and is separately trained on the main branch:

```bash
git checkout main
# After some other training...
git add model.pt
git commit
```

We then can then merge the updates from the `alternate-training` branch via a standard `git merge`:

```bash
git merge alternate-training
```

Git-Theta supports various methods for automatically merging models, including parameter averaging. The merge tools shows us each parameter that is different between the two models and asks what merge operation to perform.

# Efficiently tracking updates

Git-Theta supports various workflows for efficiently tracking updates to a checkpoint.

## Parameter groups

Under the hood, Git-Theta tracks changes to a checkpoint at the *parameter group* level.
A parameter group is a semantically-grouped collection of parameters like a weight matrix or bias vector in a neural network.
Parameter groups are determined based on the structure of the checkpoint file itself as specified in the format-specific [Checkpoint class](https://github.com/r-three/git-theta/tree/main/git_theta/checkpoints).
In the simplest case where all of the parameters of a model are updated, Git-Theta will effectively store an entirely new copy of the checkpoint.
However, if only a subset of the model's parameter groups are updated, Git-Theta will only store the updates to the changed parameter groups, which saves space and communication costs.
Similarly, if a model is updated by adding new parameter groups, Git-Theta will only store the new parameter groups.

## Parameter-efficient updates

Beyond updating a subset of a model's parameter groups, Git-Theta also natively supports *parameter-efficient* updates.
Examples of parameter-efficient updates include updating a sparse subset of the model's parameters (as in [FISH Mask](https://arxiv.org/abs/2111.09839) or [Diff Pruning](https://arxiv.org/abs/2012.07463)) or applying a low-rank update (as in [LoRA](https://arxiv.org/abs/2106.09685)).
There are multiple workflows for efficiently tracking parameter-efficient updates with Git-Theta.

### Saving update information as new parameter groups

A simple way to track parameter-efficient updates is to store the information required to produce the update (e.g.\ the low-rank factors for LoRA or the indices and values for a sparse update) as new parameter groups in the checkpoint file itself.
In this case, model code handles creating and applying the update and the checkpoint is saved and loaded as usual.

**Pros:**
* Simple to implement.
* Original checkpoint and updates are bundled together and saving and loading is done as usual without special logic.

**Cons:**
* Checkpoint saving may result in unnecessary writes of unchanged parameters.
* If many subsequent parameter-efficient updates are made, the number of parameter groups stored in the checkpoint file could become onerous.

After saving update information in the checkpoint, the new checkpoint can be committed simply using `git add` and `git commit` as usual.

### Applying updates to existing parameter groups before saving

A second option is to apply the updates to the parameter groups before saving them.
Git-Theta will treat these updates in the same way it treats updating all parameters in a parameter group, so this approach sacrifices any savings to communication or storage costs that would have been achieved by using a parameter-efficient method.

**Pros:**
* Similar to saving update information as new parameter groups, this is simple to implement and only involves handling a single checkpoint file.
* The checkpoint can be used as-is without any special logic for re-applying the update.

**Cons:**
* Sacrifices any communication/storage savings from using a parameter-efficient update.
* Checkpoint saving may result in unnecessary writes of unchanged parameters.

After folding the updates into the parameter groups, the model can be saved, added, and committed as usual.

### Saving update information externally

Another option is to save parameter-efficient update information in a separate file from the original checkpoint.
This maintains storage and communication efficiency at the cost of requiring additional implementation overhead.

**Pros:**
* Only the parameter updates are saved, reducing storage requirements.
* Only updated parameters are saved during the training loop, removing wasteful writes.
* Makes it easy to work with multiple datasets via different file names or branches.

**Cons:**
* Implementation overhead. Training code needs to be able to segment out and save only the parameters that have changed. Inference code needs to know how to load both the original checkpoint and the update from the new checkpoint as well as how to merge them.
* The original checkpoint and parameter udpates are decoupled, running the risk that one could be changed without appropriately modifying the other.

Assuming we have already committed the original model, the auxiliary information checkpoint needs to be separately added and committed as normal.

### Using Git-Theta to incorporate external update information

To streamline the workflow of saving update information externally, Git-Theta has functionality for applying the update as part of the version control process.
This ties together the main model checkpoint and the update checkpoint to prevent them from diverging.
In addition, Git-Theta takes care of applying the update so that the model checkpoint can be used as-is after checkout.
Git-Theta assumes assumes that the update information checkpoint uses the same format as the original checkpoint and that the names of updates are prefixed by the name of the parameter group they are applied to.
For example, if a parameter group called `/layer1/weights` was updated with a low-rank update, then Git-Theta would look for parameters named `/layer1/weights/R` and `/layer1/weights/C` in the update information checkpoint based on the naming conventions in the [`LowRankUpdate`](https://github.com/r-three/git-theta/blob/main/git_theta/updates/low_rank.py) class.
The low-rank update can then be efficiently tracked and applied with Git-Theta via

```bash
git theta add /path/to/original/checkpoint.ckpt --update-type low-rank --update-path /path/to/updates.ckpt
git commit
```

Note that using this approach requires using `git theta add` instead of just `git add` to allow for additional command line arguments.
Updates that involve modifying existing parameters (rather than just completely replacing them) are referred to by Git-Theta as "incremental updates" and are handled via a plugin system (described [below](#incremental-updates)).

# Managing model development with Git-Theta

Git-Theta provides principled and rigorous way to keep track of different versions of a model based on the standard version control workflow.

## Tracking the progression of a model

Pre-trained models are increasingly being continually updated to make them applicable to new tasks and domains.
For example, a pre-trained language model might be [adapted to a new objective](https://arxiv.org/abs/2104.08691), [process text in a new domain](https://arxiv.org/abs/2004.10964), and [improve its instruction-following capabilities](https://arxiv.org/abs/2110.08207) before being fine-tuned on a target task.
Git-Theta allows the provenance of these steps to be straightforwardly tracked using Git's built-in functionality.
Apart from committing each model to keep track of a checkpoint's history, other Git functionality like [tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging) can be used to keep track of notable versions.
When checking out a particular version of a model, Git-Theta will only download what's required to reconstruct it and won't download any files that have already been cached.

## Tracking different versions of a model

Model development is not always straightforward - often we want to try out different versions of a base model, or we might create different versions that are applicable to different tasks.
Git-Theta supports this mode of development natively simply by using Git's branch feature - simply create a new branch (`git checkout -b`), modify the model, and add and commit it as usual.
This provides a straightforward workflow for trying out different ways to update a model.
If parameter groups are shared across checkpoints being tracked by Git-Theta (whether they are on the same or different branches), Git-Theta will only store a single copy of each parameter group.
Contributors can also develop their own updated versions of a model by forking the base repository.

## Merging models

If different versions of a model are created on different branches or repositories, Git-Theta will handle merging them.
When `git merge` is run and there is a merge conflict between two histories of a model, Git-Theta will automatically open its merge tool.
Git-Theta's merge tool currently supports basic resolution patterns like choosing the parameters from one of the models or merging parameter groups via averaging.
For more sophisticated merges, the environment variable `GIT_THETA_MANUAL_MERGE` can be set to true when performing the merge operation, i.e.

```bash
export GIT_THETA_MANUAL_MERGE=True
git merge ${other-branch}
```

and the merge tool will write out 4 copies of the model, one for each branch being merged and an additional one that represents the model at the most recent commit in the history of both branches.
The merge tool will also specify where to save the merged model.
After the merged model has been saved to the specified location, a merge commit can be created as usual.

# Sharp Edges

Git-Theta aims to support all standard Git workflows.
However, there are currently some situations that Git-Theta does not currently support.

## Git Rebase

Currently, `git rebase` is not supported when special update types are used.
Additionally, repeated merge-conflict resolution---often encountered in a rebase---can be onerous for large models.

## Octopus Merges

Currently, git-theta's merge utilities are optimized for (and only tested for) 3-way merges where two branches with a shared ancestor commit are merged together.
We are working on support for Octopus merges where multiple branches are all combined at once.

# Under the hood

This section describes how Git-Theta works in more detail.

## Git-Theta's filters

Git offers several points of customization where specialized, model-aware Git-Theta versions of various tools are run.
Git has a "working tree" where human-facing files live and a "staging area" where a copies of working tree files live before they are stored in Git.
When a file is moved from the working tree to the staging area, the "clean filter" is run.
When it is moved back the "smudge filter" is run.
Git-theta provides model-aware versions of these filters.

When a model checkpoint is **cleaned** (`git add`):

1. Git-Theta reads the checkpoint from the working tree using a plug-in system to support different deep-learning frameworks.
2. Git-Theta converts the checkpoint into a tree of parameter group names that map to parameter values.
3. Git-Theta records metadata for each parameter group, including a hash of the parameter values.
4. Git-Theta compares the metadata for the current parameter group with its previous value.  If the metadata doesn't match, the parameter is serialized and then saved using Git LFS. The Git LFS metadata is recorded in the metadata file.
5. The metadata is written to the staging area.

Thus, Git itself only tracks the model metadata; actual values are stored efficiently Git LFS. Additionally, by checking for matching metadata, only changed parameters are stored.

When a model checkpoint is **smudged** (`git checkout`):

1. The Git-Theta metadata file is retrieved from Git.
2. For each parameter, the [Update](https://github.com/r-three/git-theta/tree/main/git_theta/updates) plug-in system is used to get actual parameter values.
  a. For updates that change all parameter values, the Git LFS metadata is used to get the values directly.
  b. For parameter-efficient updates, Git LFS metadata is used to get update values, previous parameter values are retrieved from Git itself, and the update is applied.
4. The parameter values are written into the working tree using the checkpoint plug-in system to handle different deep learning frameworks.

When installing Git-Theta with `git theta install`, the following lines are added to the global `~/.gitconfig`:

```ini
[filter "theta"]
    clean = git-theta-filter clean %f
    smudge = git-theta-filter smudge %f
    required = true
[merge "theta"]
    name = Merge Models with Git-Theta
    driver = git-theta-merge %O %A %B %P
[diff "theta"]
    command = git-theta-diff
```

This configuration defines two [Git filter drivers](https://git-scm.com/docs/gitattributes#_filter) for Git-Theta and registers them under the name `theta`.
In addition, it defines merge and diff programs, also named `theta`.
When `git theta track path/to/model` is run, an entry is added to the `.gitattributes` file to configure Git to use Git-Theta. The new entry looks like

```ini
path/to/model filter=theta merge=theta diff=theta
```

This tells git that anytime a file that matches the pattern `path/to/model` is processed, use the filter/merge/diff driver named `theta`.

## Incremental updates

Git-Theta supports updates that are based on the previous version of the parameter values.
For example, if a few entries of a parameter group are updated, Git-Theta can avoid storing a new copy of the parameter group; instead, it can be computed on the fly during a smudge filter based on the sparse update and the previous value.
Such updates are implemented as subclasses of the `IncrementalUpdate` class.
`IncrementalUpdate`s include references to the commit that holds the last parameter value in their metadata.
Then, when the new value is needed, the `IncrementalUpdate` class will fetch the value of the previous parameter *from git* and apply the current update.
This yields a massive reduction in storage costs.
Additionaly, this can be done recursively, i.e. Git-Theta will continuous fetch previous values and apply `IncrementalUpdate`s until a self-contained update (such as a `Dense` update that replaces all parameter values with new ones) is hit.

## Locality-sensitive hashing

To avoid processing parameter groups that have not been changed, Git-Theta needs a way to determine whether a given parameter group's values have changed.
Directly testing for equality or comparing bitwise hashes might be overly strict due to numerical instability and noise that could arise from using incremental updates, different hardware, or different software stacks.
Instead, Git-Theta uses uses locality sensitive hashing (LSH) for parameter hashes.
Specifically, an LSH that approximates Euclidean distance and uses the random-pool approach to hash parameters of variable sizes.
Git-Theta's LSH uses 16 hash functions and is calibrated so that two parameter groups with a Euclidean distance less than $1e^{-8}$ will have the same hash with a probability of at least $0.99$.
Additionally, weights with a distance $\in [1e{-8}, 1e^{-6}]$ are double-checked with [`numpy.allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).

## Plug-ins

Git-theta makes heavy use of [python plug-ins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/) to enable users to add support for additional checkpoint formats as well as custom merge patterns and incremental updates.
Specifically, Git-Theta currently support plug-ins for the [`Checkpoint`](https://github.com/r-three/git-theta/blob/main/git_theta/checkpoints/base.py), [`Update`](https://github.com/r-three/git-theta/blob/main/git_theta/updates/base.py), and [`Merge`](https://github.com/r-three/git-theta/blob/main/git_theta/merges/base.py) classes.
Third-party users can register a plug-in by creating a small installable package that defines the plugin and registers it as an entry point under the name scope `git_theta.plugins.(checkpoints|updates|merges)`.
An example plugin for JSON formatted checkpoints can be found [here](https://github.com/r-three/git-theta/tree/main/plugins#git-theta-plug-ins).
Alternatively, plug-ins can be added directly to the `git-theta` package by adding new subclasses to the appropriate modules, then declaring it in the `entry_points` dict in `setup.py`.

# Development Setup

This project uses `black` for code formatting and `isort` for import statement ordering. Additionally, it includes CI that checks for compliance.
We include pre-commit hooks that will automatically run `black` and `isort` against any python files staged for commit.
 These hooks can be installed with:

```bash
$ pip install -r requirements-dev.txt
$ pre-commit install
```

When one of these tools must reformat your file, it will show as the pre-commit hook failing and your commit will be cancelled.
Reformatted source files will appear in your working directory ready to be re-added to staging (`git add`).
 Running `git commit -m ${msg}` again will result in the hooks passing and the commit actually happening. *Note:* As your initial commit was blocked, you will probably want to use the same message in the commit that actually goes through.

