Implemented a Hello World exampe using a design similar to the proposal in [#16](https://github.com/r-three/checkpoint-vcs/issues/16).

Below is a demo of the implemented proof of concept:

1. First initialize a new git repo
```
git init
```

2. Initialize the git filter driver for model files:
```
git-theta init
```
For this simple example, models are stored as json files. The command defines a filter that captures model files by adding
```
*.json filter=theta
```
to `.git/info/attributes` and defines a smudge and clean filter by adding
```
[filter "theta"]
        clean = git-theta-filter clean %f
        smudge = git-theta-filter smudge %f
        required = true
```
to `.git/config`. This means that when `foo.json` is being added to staging area, `git-theta-filter clean foo.json` is run and when it is checked out, `git-theta-filter smudge foo.json` is run.

3. Create a file `my_model.json` in the repo containing
```
{
    "layer1": {
        "w": [1,2,3,4],
        "b": [10]
    },
    "layer2": {
        "w": [-1,-2,-3,-4],
        "b": [-10]
    },
    "other_params": {
        "alpha": 0.1,
        "lr": 0.001
    }
}
```

4. Run `git-theta add my_model.json`.

`git-theta` is a python program that (1) loads `my_model.json`, (2) saves each individual parameter group to the filesystem under `.git_theta/my_model`, (3) runs git add on each parameter group file saved under `.git_theta`, (4) runs git add on `my_model.json`.

Note that when `my_model.json` is added to the staging area (step 4), it gets intercepted by the previously defined clean filter for *.json files. The clean filter runs `git-theta-filter clean my_model.json`. `git-theta-filter clean` is another python program that replaces the contents `my_model.json` with a dictionary containing `{'model_dir': '.git_theta/my_model', 'model_hash': <hash of my_model.json>}`

After all this, the staging area contains a snapshot of the model's parameter groups under `.git_theta/my_model` and a file called `my_model.json` that doesn't actually have the model parameters but instead some metadata about where to find the parameters at a later time. Although the staged version of `my_model.json` only contains metadata, the working copy still contains the model parameters.

The output of `git status` at this point is:

```
Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   .git_theta/my_model/layer1/b
	new file:   .git_theta/my_model/layer1/w
	new file:   .git_theta/my_model/layer2/b
	new file:   .git_theta/my_model/layer2/w
	new file:   .git_theta/my_model/other_params/alpha
	new file:   .git_theta/my_model/other_params/lr
	new file:   my_model.json
```

5. `git commit` to commit the model
6. Modify one parameter group in `my_model.json`
7. Run `git-theta add my_model.json` to stage the changes to the model.

At this point in time only the modified parameter group's file under `.git_theta` has been modified. The output of `git status` at this point is:

```
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
	modified:   .git_theta/my_model/layer1/w
	modified:   my_model.json
```

8. `git commit` to commit the change
9. Look at the output of `git log` to get the commit hashes:

```
commit 11b37c3aacd5d6f4ec23986d95e739ed44433d6c (HEAD -> master)
Author: Nikhil Kandpal <nkandpa2@gmail.com>
Date:   Wed Oct 5 00:31:02 2022 -0400

    modify layer1/w

commit d13dca536d2690cb758c3866acb2abd1e0f32790
Author: Nikhil Kandpal <nkandpa2@gmail.com>
Date:   Wed Oct 5 00:29:21 2022 -0400

    initial commit
```

10. Checkout the first commit and make a new branch to test whether we can re-create the initial model

`git checkout d13dca536d2690cb758c3866acb2abd1e0f32790 -b my_branch`

When this occurs, the smudge filter intercepts the model file and is called with `git-theta-filter smudge my_model.json`. `git-theta-filter smudge` is a python program that reads the metadata file and reconstructs the model checkpoint from the data in `.git_theta/my_model`.

11. Check the contents of `my_model.json`

```
cat my_model.json
{"other_params": {"lr": 0.001, "alpha": 0.1}, "layer1": {"b": [10], "w": [1, 2, 3, 4]}, "layer2": {"b": [-10], "w": [-1, -2, -3, -4]}}
```

Notice that the model parameters are what we started with (although in a different order since the model checkpoints are json) and `layer1/w` has its starting parameters.

12. Switch back to HEAD and check `my_model.json`

```
git switch -
cat my_model.json
{"other_params": {"lr": 0.001, "alpha": 0.1}, "layer1": {"b": [10], "w": [10, 20, 30, 40]}, "layer2": {"b": [-10], "w": [-1, -2, -3, -4]}}
```

At the HEAD commit, the parameters for `layer1/w` are the modified version once again.
