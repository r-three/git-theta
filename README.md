# Git-Theta

A Git extension for collaborative, continual, and communal development of machine learning models.

# Example Usage
<!--We should motivate this better...-->
Large distributed teams are able to efficiently work on shared codebases due to version control systems like Git. In an effort to [build ML models like we build open-source software](https://colinraffel.com/blog/a-call-to-build-models-like-we-build-open-source-software.html), Git-Theta is a Git extension that allows you to *efficiently* and *meaningfully* track a model's version history natively through Git.

Say you have a codebase for training an ML model along with a model checkpoint that is continually updated as new training methods are developed and new data is collected:
```bash
my_codebase
├── model.pt
└── train.py
```
Git-Theta allows you to track the version history of your continually updated model alongside the code through Git.

If you haven't already, first initialize your codebase as a Git repository.
```bash
git init
```
In order to track the model checkpoint using Git-Theta, run the command
```bash
git theta track model.pt
```

This command adds the line `model.pt filter=theta` to the `.gitattributes` files in the top-level directory of the repository. This signals to Git that when running certain operations on `model.pt`, it should delegate to Git-Theta.

With the model tracked by Git-Theta, you can treat the model checkpoint exactly like you would any other file in a Git repository. All of the regular Git commands like `add`, `checkout`, `status`, `commit`, `push`, and `pull` will work out of the box.


## Under Construction
`git diff` for a meaningful summary of how two models are different (e.g., added/removed/modified parameter groups, were parameter group modifications dense/sparse/low-rank, etc.)

`git merge` to perform various possible automated merging strategies (e.g., parameter averaging, mix and match parameter groups, etc.) that can be evaluated.

# Why is this better than using Git or Git LFS?
Git on its own can certainly be used for versioning non-text files like model checkpoints. However, the main limiting factors are that
1. Git remotes like Github and Bitbucket have a maximum file size (~50MB)
2. Git is not designed to handle very large repositories

There are a number of existing solutions for storing large files with Git that circumvent the maximum file and repository size, such as Git LFS. The main issue with these solutions for versioning ML models is that they are unaware of the fact that they are tracking ML models. 

Imagine you have a checkpoint that you are fine-tuning by [training only a small percentage of the parameters](https://arxiv.org/abs/2111.09839), [training only a few of the layers](https://arxiv.org/abs/2106.10199), or by [adding new trainable modules](https://arxiv.org/abs/1902.00751). In these cases, most of the model remains the same and only a small fraction of the model gets modified. However, tools like Git LFS just see that the checkpoint file has changed, and will store the new version of the checkpoint file in its entirety. 

Git-Theta is aware of the structure of ML models and is designed for efficiently storing only the parts of a model that have changed from its previous version.

# Getting Started
## Git LFS installation
Download and install Git LFS using the instructions from [the Git LFS website](https://git-lfs.github.com)

## Setting up Git-Theta
First, clone the repository
```bash
git clone https://github.com/r-three/git-theta.git
```
Install the Git-Theta package by running:
```bash
cd git-theta
pip install .[all]
```
The final step of installation is running:
```bash
git theta install
```
This command adds the following lines to your global `~/.gitconfig`:
```
[filter "theta"]
        clean = git-theta-filter clean %f
        smudge = git-theta-filter smudge %f
        required = true
```
These define a [Git filter driver](https://git-scm.com/docs/gitattributes#_filter) that can be engaged in any repositories you work with to track machine learning models natively through Git.

### A Single Deep Learning Framework

If you plan to track model checkpoints created by a single deep learning
framework, for example only PyTorch or only Tensorflow, you can elect to only
ensure the framework you use will be installed, avoiding the long install times
and possible version requirements issues installing unused frameworks may bring.

For example, install Git-Theta with only PyTorch checkpoint support:

``` bash
cd git-theta
pip install .[pytorch]
```

If you already have your framework of choice installed (i.e. pip doesn't need
to ensure it is installed), you can just install Git-Theta with `pip install .`

# Development Setup

This project uses black for code formatting and includes CI checks for black compliance.
To configure pre-commit hooks, which will automatically run black against any files
staged for commit before allowing the commit to happen run the following:

``` sh
$ pip install -r requirements-dev.txt
$ pre-commit install
```

When black must reformat your file, it will show as the black pre-commit hook
failing. When this happens you will see that the source file has been reformatted
and is ready to be re-added to the index. Running `git commit` again should
result in all the hooks passing and the commit actually happening.

## Add support for new checkpoint types

We support new checkpoint types via plug-ins. Third-party users do this by
writing small installable packages that define and register a new checkpoint
type.

Alternatively, plug-ins can be added directly to the `git-theta` package by
adding the checkpoint handler to `checkpoints.py` and adding it to the
`entry_points` dict of `setup.py`.
