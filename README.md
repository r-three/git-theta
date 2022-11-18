# git-theta

A git extension for collaborative, continual, and communal development of machine learning models. 

# How to use this repository?
## Git LFS installation 
Download and install Git LFS using the instructions from [the Git LFS website](https://git-lfs.github.com)

## Installing the git-theta package
clone the repository  
```bash
git clone https://github.com/r-three/git-theta.git
```
Install the git-theta package by running:
```bash
cd git-theta
pip install .
```
## Initializing git theta
Initialize git theta by running:
```bash
git theta install
```

The following lines will be added to `~.gitconfig` after successful installation. 
```
[filter "lfs"]
        smudge = git-lfs smudge -- %f
        required = true
        clean = git-lfs clean -- %f
[filter "theta"]
        clean = git-theta-filter clean %f
        smudge = git-theta-filter smudge %f
        required = true
```

# Example Usage
<!--Create a folder with a text file and a model checkpoint. Initialize it as a git repository.-->
Imagine you have a codebase containing code for training a model and a trained model checkpoint.
```bash
my_ml_repo
├── model.pt
└── train.py
```
You may want to version control both your model and your training code. git-theta extends git to efficiently and meaningfully track ML models.

First, initialize your codebase as a git repository.
```bash
git init
```
In order to track the model checkpoint using git theta, run the command
```bash 
git theta track {path_to_model_checkpoint}
```

The above command adds the following lines to the `.gitattributes` files in the home directory.
```
".git_theta/model.pt/**" filter=lfs diff=lfs merge=lfs -text
model.pt filter=theta
```

Stage the model in git by running the command 
```bash
git theta add model.pt
```

This will store the parameters of the model using tensorstore inside a newly created `.git_theta/model.pt` directory. For example, consider a parameter name `decoder.block.0.layer.0.SelfAttention.k.weight` in the model checkpoint. The corresponding parameter values will be stored in `.git_theta/model.pt/decoder.block.0.layer.0.SelfAttention.k.weight`. 

At this step, you can run `git status` and see all the `.git_theta/model.pt/{parameter_name}` files in "Changes to be committed" along with the model checkpoint file and the `.gitattributes` file.

Since this is just a normal git repo, you can also add any other code/text files that you would like to version control using `git add`. You can then commit the changes and push to a git remote. 

The remote will contain the `.git_theta/model.pt` directory where the actual model parameters are stored. These parameteres are actually stored using Git LFS, and on certain git remotes (like Github and BitBucket) you should see them listed as LFS objects. The actual model checkpoint on the remove will simply contain some metadata related to the model parameters, such as the hash, shape and type of each of the parameter groups. 

## TBA
`git diff` on the model checkpoint will identify which parameter groups are modified or added or removed. 

`git merge` will assume that all merges to the checkpoint (i.e. to parameter group files) result in merge conflicts and offer various possible automated merging strategies that can be tried and vetted.

`git checkout` to a commit will construct a checkpoint based on the contents of `.git_theta/<model_checkpoint_name>` at that commit. 

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
