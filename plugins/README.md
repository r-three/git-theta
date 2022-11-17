# git-theta Plug-ins

git-theta support plugins for custom model checkpoint formats.

## Writing a Checkpoint Plug-in.

A checkpoint plug-in should subclass `git_theta.checkpoints.Checkpoint`.  The plugin
should implement the `load` method which reads the checkpoint format into a dict
mapping parameter names to parameter weights. It should also implement `save` which
writes the original checkpoint format based on the dict of weights representation.

## Packaging a Plug-in

The plug-in should be wrapped in an installable package that declares itself as a plugin using the
`"git_theta.plugins.checkpoint"` entry point. The following should appear in the
`setup.py` for the package.

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

Note: user plug-in packages can have any name as long as they register the
`"git_theta.plugins.checkpoints"` entry point.

## Using a Plugin

Having a plug-in installed will give `git_theta` access to the checkpoint class it provides.
To use this class include the CLI argument `--type my-cool-checkpoint` to the
`git-theta add ...` command.
