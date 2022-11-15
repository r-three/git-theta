# Git Theta Plug-ins

Git theta support plugins for custom model checkpoint formats.

## Writing a Checkpoint Plug-in.

A checkpoint plug-in should subclass `git_theta.checkpoints.Checkpoint`.  The plugin
should implement the `_load` method which reads the checkpoint format into a dict
mapping parameter names to parameter weights. It should also implement `save` which
writes the original checkpoint format based on the dict of weights representation.

## Packaging a Plug-in

The plug-in should be wrapped in an installable package that declares itself as a plugin with
an entry point like this:

```python
setup(
    ...,
    install_requires=[
        "git_theta",
        ...,
    ]
    entry_points={
        "git_theta.plugins.checkpoints": [
            "my-cool-checkpoint = package.subpatch:MyCoolCheckpointClass",
        ],
    },
    ...,
)
```

## Using a Plugin

Having a plug-in installed will give `git_theta` access to the checkpoint class it provides. 
To use this class include the CLI argument `--type my-cool-checkpoint` to the 
`git-theta add ...` command.
