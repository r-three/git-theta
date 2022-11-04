# git-theta

git extension for collaborative, continual, and communal model development. 

# How to use this repository?
## LFS installation 
Download the LFS package from the [website](https://github.com/git-lfs/git-lfs/releases/tag/v3.2.0). For Linux users, download the amd64 version from the list of assests in the website. 

## Getting started
clone the repository  
```bash
git clone https://github.com/r-three/git-theta.git
```
Install the packages by running:
```bash
cd git-theta
pip install -e .
```
## Installing git theta
<!--Is repository same as codebase?-->
You can initialize the git theta in the home directory of the codebase to track code and models as follows:
```bash
git theta install
```

The following lines will be added to the `.gitconfig` file in the root directory of the user after the successful installation. 
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
First, initialize git in the home directory of the codebase
```bash
git init
```
In order to track the model checkpoint using git theta, run this command
```bash 
git theta track {path_to_model_checkpoint}
```

The above command adds the following lines to the `.gitattributes` files in the home directory.
```
".git_theta/{model_checkpoint_name}/**" filter=lfs diff=lfs merge=lfs -text
{path_to_model_checkpoint} filter=theta
```
It also creates a `.git_theta/{model_checkpoint_name}` folder  in the home directory of the codebase. 

Add the model to git by running the command 
```bash
git theta add {path_to_model_checkpoint}
```

This will store the parameters of the model in the tensorstore format inside the `.git_theta/{model_checkpoint_name}` folder. For example, consider a parameter name `decoder.block.0.layer.0.SelfAttention.k.weight` in the model checkpoint with name `pytorch_model.bin`, the corresponding parameters are stored as the following hierarchy `.git_theta/pytorch_model.bin/decoder.block.0.layer.0.SelfAttention.k.weight`. 

At this step, run `git status`, you should see all the `.git_theta/{model_checkpoint_name}/{parameter_name}` files in "Changes to be committed" along with the model checkpoint file and the `.gitattributes` file.

After adding the model checkpoint, add any other code/text files that are modified using `git add`. You can then commit the changes and push to remote. 

The remote will have the `.git_theta/{model_checkpoint_name}` folder in it where instead of the actual params, git remote shows the params are stored as LFS objects. A metadata file describing the contents of the params like shape, dtype, and hash are stored inside `.git_theta/{model_checkpoint_name}/{parameter_name}`on git remote. The actual model checkpoint will be stored as a file containing the hash, shape and type of each of the keys in the checkpoint. 

## TBA
`git diff` on the model checkpoint will identify which parameter groups are modified or added or removed. 

`git merge` will assume that all merges to the checkpoint (i.e. to parameter group files) result in merge conflicts and offer various possible automated merging strategies that can be tried and vetted.

`git checkout` to a commit will construct a checkpoint based on the contents of `.git_theta/<model_checkpoint_name>` at that commit. 
