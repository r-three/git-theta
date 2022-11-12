# git-theta

git extension for collaborative, continual, and communal model development

# Design notes

`git-theta` adds functionality for keeping track of a model's parameter values.
Creation of the model's computational graph, actual usage of the parameters,
configuration, hyperparameter values, etc. is left up to code
(which can be versioned in tandem with the checkpoint via git as usual).
When a checkpoint is staged (added to the repository) via `git-theta`,
each parameter "group" (e.g. a weight matrix or a bias vector) is
given its own directory in `.git_theta/{checkpoint_file_name}/{parameter_name}`.
The files within the `.git_theta/{checkpoint_file_name}` directory are tracked by git.
`git-lfs` is used to store these files.
Parameter group files are stored via TensorStore.
When you add a checkpoint, a metadata file describing the contents of the
checkpoint (shape, dtype and hash for each parameter group)
is staged by `git` (and will be what is pushed to a remote),
but the local copy of your checkpoint remains unchanged.
During a checkout operation, the checkpoint is reconstituted locally based
on the contents of `.git_theta/{checkpoint_file_name}`.

For a given parameter group, `git-theta` will store either the group's parameter's
values or an update to the values in the event that it can be stored more
efficiently.
For example, if a sparse update is made to a parameter group, `git-theta` will
store the sparse update in a sparse format to save storage costs.
For a given parameter group, all updates made since the last time the group's
values were stored are stored in an `updates` subdirectory within
`.git_theta/{checkpoint_file_name}/{parameter_name}`.
Whenever a dense update is made to a parameter group, the new full set of
parameter values is stored and all prior updates are removed from the
`updates` subdirectory.
All parameters and updates are currently stored via `git-lfs`.

`git diff` will identify which parameter groups have changed and how.

`git merge` will assume that all merges to the checkpoint (i.e. to parameter
group files) result in merge conflicts and offer various possible automated
merging strategies that can be tried and vetted.


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
