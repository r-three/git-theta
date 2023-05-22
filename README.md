# Git-Theta

<center><img src="https://user-images.githubusercontent.com/417568/229904559-d61d710c-7986-4a07-a405-d86b196f5046.png" width="50"></center>

`git-theta` is a Git extension for collaborative, continual, and communal development of machine learning models.

Version control systems, like Git, enable large distributed teams to effectivly work on shared codebases by tracking changes over time as well as providing powerful tools for merging changes from multiple sources and seeing what parts of a file have actually changed.

In an effort to [build ML models like we build open-source software](https://colinraffel.com/blog/a-call-to-build-models-like-we-build-open-source-software.html), Git-Theta is a Git extension that allows you to *efficiently* and *meaningfully* track a model's version history natively through Git.

Git tracks the history of a repository at each commit using snapshots. This is done efficiently by only saving the state for files that have changed since the last snapshot. This approach does not scale to the large files used for model checkpoints. Even if only a small number of parameters are changed, the whole model checkpoint will included in the snapshot.

Git-theta efficiently tracks model history by extending the idea of snapshots into the model checkpoint itself. Instead of treating the checkpoint as a binary blob---as other git large file extensions like git-lfs do---git-theta breaks the model down into its constituents, allowing for the snapshotting of each weight individually.

This approach enables *meaningful* diff information. For example, git-theta can show you which parameters have changed and by how much. Similarly, git-theta provides the ability to merge multiple models together by applying various merge operations only to the parameters that have changed.

# Quick Start

## Git LFS installation
Download and install Git LFS using the instructions from [the Git LFS website](https://git-lfs.github.com).

## Installing Git-Theta
1) intstall git-theta.
```bash
pip install git-theta
```

2) Configure Git to use git-theta when tracking model checkpoints.
```bash
git theta install
```

*Note:* A Single Deep Learning Framework

If you plan to track model checkpoints created by a single deep learning framework, e.g. only PyTorch or only Tensorflow, you can elect to ensure that only your framework of choice will be installed. For example, install Git-Theta with only PyTorch checkpoint support:

``` bash
cd git-theta
pip install .[pytorch]
```

Alternativly, if you have already installed your framework, i.e. pip doesn't need to make sure that it is installed, you can just install Git-Theta without any deep-learning frameworks using `pip install git-theta`.

## Version Controling a Model

Say you have a codebase for training an ML model along with a model checkpoint. i.e.

```bash
my_codebase
├── model.pt
└── train.py
```

Git-Theta allows you to use Git to track the version history of your code ***and*** model as you iteratively update them.

To use git-theta to version control the model:

1. Run the command:

```bash
git theta track model.pt
```

2. Stage the updated `.gitattributes` files. These first two steps are only done the first time a model is tracked.

```bash
git add .gitattributes
```

  * Optionally, commit the `.gitattributes` file now. It can also be committed as part of the models first commit. This file tells git to use git-theta and needs to be committed so that when others clone the repo, their git knows to use git-theta.
```bash
git commit -m .gitattributes
```

4. Add the model to staging
```bash
git add model.pt
```
6. Commit the model
```bash
git commit
```

After tracking the model, git-theta will run transparently, that is, you can regular Git commands (`add`, `commit`, `push`, `pull`, `checkout`, `status`, `diff`, etc.) as if it was any other file.

Additionally, `git theta add` can be used over `git add` to provide extra information. This includes the deep-learning checkpoint format with `--checkpoint-type`, what type of change happened with `--update-type` (which can save storage and bandwidth), and the location of updates that stored external to the model with `--update-path`.

# Example Usage

To showcase some of git-theta's features, we will walk through a mini example of how a collaborative workflow may work. We start by committing the pre-trained T0-3b[^10] checkpoint.

```bash
git theta track model.pt
git add model.pt
git commit -m "Initial T0-3b checkpoint."
```

Then we do few-shot training on [CB](https://www.tensorflow.org/datasets/catalog/super_glue#super_gluecb) using LoRA[^5] and commit the updated model.

```bash
git theta add model.pt --update-type low-rank --update-path /path/to/updates.pt
git commit -m "Few-shot LoRA training on CB."
```

On a branch, fine-tuning is used for few-shot training on [RTE](https://www.tensorflow.org/datasets/catalog/super_glue#super_gluerte).
```bash
git checkout -b RTE
...
git add model.pt
git commit -m "Few-shot fine-tuning on RTE."
```

Concurrently, on the main branch we do few-shot training on [ANLI-r1](https://www.tensorflow.org/datasets/catalog/anli#anlir1_default_config).
```bash
git checkout main
...
git add model.pt
git commit -m "Few-shot fine-tuning on ANLI-r1"
```

We then merge the RTE model back into the main branch using parameter averaging. The merge tools shows us each parameter that is different between the two models and asks what merge operation to perform.
```bash
git merge RTE
```

Finally we remove the T5 [^9] pre-training specific sentinel tokens from the vocabulary. This is done using an `Update` plug-in to remove the need to store the updated embedding table at all.
```bash
# This is just an example plugin
cd git-theta-remove-sentinels-plugin && pip install . && cd -;
git theta add model.pt --update-type remove-sentinels
git commit -m "Removing pre-training sentinels from vocabulary."
```

Below we can see the performance of various checkpoints in this process. Additionaly we can see comparisions of speed and storage requirements. As git-theta does more work than git-lfs it is unsuprising it takes longer but we are working on incresing the speed. Whenever non-dense updates are used, git-theta shows large gains in speed.

<center><img src="" width="400px"></center> TODO: Insert image from github-editor and remove this todo


# Git-Theta Cookbook

I'm a user and I want to:

## Work with Parameter Efficient Updates

There are many different approaches to parameter efficient training of large language models; some methods only make sparse updates to parameters [^1][^2], some only update a subset of the weights [^3], and others introduce new trainable modules---either as part of the model [^4][^5][^6] or as part of the input [^7]. As such, there are many different possible ways to implement and save these new models.

### Saving In-Model with New Parameter Groups

The simplest option is save the updates as new parameter groups in the model.

**Pros:**
* Simple to implement. Training code loads the original checkpoint and adds/trains the new parameters and reuses your frameworks save functionality; inference code just loads and runs the checkpoint.
* Create different models for different datasets by using different checkpoint names (or different branchs to leverage a git-theta assistend merge later).
* Train on multiple datasetes in series by continuing training from the last checkpoint.
* Original checkpoint and updates are bundled together.
* git-theta automatically deduplicates git and remote storage of unchanged parameters, i.e. only new values are stored, even across multiple files.

**Cons:**
* Unnessicary writes are preformed in your training loop when when unchanged parameters are saved during a new checkpoint.
* While multiple models what share the same base checkpoint are efficiently stored inside of git-theta, there will be duplicates in the working directory.

This is done with:

```shell
git add ${/path/to/model.ckpt}
git commit
```

### Saving In-Model as Pre-Applied Updates

A second option is to save the updates by applying them to the original parameters. This is the most common approach when finetuning subset of parameters.

**Pros:**
* Simple to implement. Training code loads the checkpoint as is, adds/trains the new parameters, and then folds them back into the original parameters before reusing your frameworks save functionality; inference code just loads and runs the checkpoint.
* Create different models for different datasets by using different checkpoint names (or different branchs to leverage a git-theta assistend merge later).
* Train on multiple datasetes in series by continuing training from the last checkpoint.
* Original checkpoint and updates are bundled together.
* git-theta automatically deduplicates git and remote storage of unchanged parameters, i.e. only new values are stored, even across multiple files.

**Cons:**
* Unnessicary writes are preformed in your training loop when when unchanged parameters are saved during a new checkpoint.
* Updates need to be folded into the original parameters before each save.
* While multiple models what share the same base checkpoint are efficiently stored inside of git-theta, there will be duplicates in the working directory.
* When using a parameter efficient method like LoRA, folding the parameter efficient update back into the original parameter will result in a dense update that matches the shape of the original parameter `[d_model, d_model]`, instead of of the much smaller `[d_model, low_rank]`.

This is done with:

```shell
git add ${/path/to/model.ckpt}
git commit
```

#### Special Update Types, Experimental

git-theta currently supports the back-calculation of parameter efficient updates based on the current parameter value and the previous value, provided the user specifies the type of update used. This results in more efficient storage but introduces the possibility of slight numerical noise. git-theta takes steps to mitegate this; however, this apporach is not recommended and may be removed soon.

This is done with:
```shell
git theta add ${/path/to/model.ckpt} --update-type ${my-fancy-update-type}
git commit
```

### Saving Externally

Another option is to save parameter efficient updates in a seperate file from the original checkpoint. This help storage efficiency at the cost of additionaly infrastructure overhead for you.

**Pros:**
* Only the parameter updates are saved, reducing storage requirements.
* Only parameters updates are saved during the training loop, removing wasteful writes.
* It is easy to work with multiple datasets via different file names or branches.
* Only simple update formats are required, special updates types like ia^3 vectors are stored as is.

**Cons:**
* Implementation overhead. Training code needs to be able to segment out the parameters that have changed and only save that piece. Inference code needs to know how to load both the original checkpoint and the new parameters as well as how to merge them.
* The original checkpoint and parameter udpates are decoupled, if one is changed the final results can be very different

Assuming we have already committed the original model, we just run:

```shell
git add ${/path/to/updates.ckpt}
git commit
```

#### External to Pre-Applied

To help avoid issues casused by write-skew (for example, the core model checkpoint is a different version than what the update was trained with), git-theta allows one to efficiently store parameter updates from an external file with the original checkpoint during a commit. This ties the original checkpoint and the update together as a single model.

In git-theta we assume that the update are stored in the same format as the original checkpoint and that the names of updates are prefixed by the name of the parameter group they are applied to.

Assuming we already committed the original model, we run:

```shell
git theta add ${/path/to/original/checkpoint.ckpt} --update-type ${my-update} --update-path /path/to/updates.ckpt
git commit
```

*Note:* This command is `git theta add` instead of just `git add` to allow for additional command line arguments.


## Track and Access Checkpoint History as I Adapt Models

In our usage example, we started from the T0-3b checkpoint [^10]. Using this as a opaque starting point hides the history of the model. T0-3b was not trained from scratch; instead, it started as a T5 1.1-3b [^9] model trained via Span Corruption on [c4](https://www.tensorflow.org/datasets/catalog/c4) and was then adapted to the prefix LM task in [^7]. Finally, it was further adapted for zero-shot inference on novel tasks, resulting in T0-3b.

``` sh
python train.py --data c4 --output "t5_1_1_xl.pt"
git theta track t5_1_1_xl.pt
git add t5_1_1_xl.pt
git commit -m "T5 1.1 initial training run"
git tag t5-1.1
```

Now we have the original of T5 1.1 checked in and the commit is reference via the tag `t5-1.1`. This makes it easy to look up later and create release artifacts.

``` sh
python train.py --data c4-prefix --continue --output "t5_1_1_xl.pt"
git add t5_1_1_xl.pt
git commit -m "T5 1.1 LM adaptation"
git tag t5-1.1-lm
```

Now we have the LM adapted version committed, but the original version is still accessible via it's tagged commit.

``` sh
python train.py --data p3 --continue --output "t5_1_1_xl.pt"
git add t5_1_1_xxl.pt
git commit -m "T0-xl"
git tag t0-3b
```

Each version of this pre-trained checkpoint is accessible via git and the model's history is explicitly tracked in git. Additionally, only required files are downloaded, so if you never want to use T5 1.1 LM, you'll never have to download it.

## Finetune on Multiple Datasets

That's great but I have a bunch of datasets and I don't want to train them one after the other, I want to start each one from the same starting point. We could save each into its own file, but we'll use git branches in case we want ot merge the models later.

I want to fine-tune BERT [^8] on [SST2](tensorflow.org/datasets/catalog/glue#gluesst2) and [MNLI](https://www.tensorflow.org/datasets/catalog/glue#gluemnli)

```bash
git checkout -b SST2
python train.py --data SST2 --model BERT.ckpt
git add BERT.ckpt
git commit -m 'BERT finetuned on SST2'
```

```bash
git checkout main  # Now BERT.ckpt is the original model!
git checkout -b MNLI

python train.py --data MNLI --model BERT.ckpt
git add BERT.ckpt
git commit -m 'BERT finetuned on MNLI'
```

Now we have multiple finetuned copies of BERT we can access by switching branches.

## Merge a Contributors Model

I've heard that MNLI is a good transfer task and starting from their can help a lot with training other tasks, so I want to bring in those changes. First we need to be on the MNLI branch, this is a local branch in our example, but it could also be from a contributor, i.e. a GitHub pull request. GitHub has further instructions, but the basices are:

``` sh
git checkout -b ${contrib}-MNLI
git pull git@github.com:${contrib}/${repo} MNLI
```

Now we are on a branch looking at their model! We can merge their model in order to test it.

``` sh
git merge --no-ff main
```

There will most likely be a merge conflict between the two models so the git-theta merge tool will open. We select our merge strategies via the prompts and end up with the merged model.

Now we can do things like run tests, evaluate on different datasets, and decide if we want to keep their model. If we don't, all we need to do is delete this branch and let them know why we won't be merging it. In this example there isn't much to test as we are overwriting the original checkpoint, but if their patch was something like a few steps of training to fix some specific behavior, we would want to verify it doesn't have detrimental effects.

If we do want to merge it:

``` sh
git checkout main
git merge --no-ff ${contrib}-MNLI
git push origin main
```

The last line pushes the merged model to the remote repo, making it accessible to all.

### Manual Merges

If the environment variable `GIT_THETA_MANUAL_MERGE` is set to true when performing the merge operation, i.e.
```bash
export GIT_THETA_MANUAL_MERGE=True
git merge ${other-branch}
```

Then the merge tool will write out 3 copies of the model, one for each branch being merged and an additionaly one which represents the model at the most recent commit in the history of both branches. The merge tool will also tell them where to save their merged model. One can then merge the models however they want, save it to the correct path, and continue the merge commit.

# Shape Edges

## Git Rebase

Currently, `git rebase` is not supported when special update types are used. Additionaly, repeated merge-conflict resolution---often encountered in a rebase---can be onerous for large models.

We are working on support for this workflow.

## Octopus Merges

Currently, git-theta's merge utilities are optimized for (and only tested for) 3-way merges where two branchs with a shared ancestor commit are merged together. We are working on support for Octopus merges where multiple branchs are all combined at once.

# What is Git-Theta?

Git-Theta is an externsion that is designed for effective and meaningful version control of machine learning models.

## Why not Git (or Git-LFS)?

While the main use-case for Git involves text files, its includes tools to versioning non-text files (such as our model checkpoints). These tools are limited as 1) Git is not designed to handle very large repositories and 2) Git remotes like Github and Bitbucket have a maximum file size (~50MB).

Existing solutions (e.g. Git-LFS) for large file storage circumvent these issues by saving large files to an external location. Instead of tracking the large file itself, Git is used to track its metadata.

However, these tools don't address the root issue, Git is designed around the idea of file-level snapshots, efficient storage by only including copies of changed files. This works well for small text-files, but model checkpoints are huge files. Thus the smallest change within the file results in a new copy of the whole checkpoint.

Git-Theta understands the internal structure of model checkpoints, that they are logically partitioned into parameter groups (weight matrices, bias vectors, etc.), and leverages that to extend snapshots to the parameter level. When a version controlled model is changed, git-theta only snapshots the parameter groups that are actually different, removing storage of duplicated values.

## How Git-Theta Works


### Git Extensions

Git offers several points of customization where specialized, model-aware git-theta versions of various tools are run.

<center>TODO use github-editor to insert flowchart image</center>

Git has a "working tree" where human facing files live and a "staging area" where a copies of working tree files live before they are stored in Git. When a file is moved from the working tree to the staging area, the "clean filter" is run. When it is moved back the "smudge filter" is run. Git-theta provides model-aware vesrions of these filters.

When a model checkpoint is **cleaned** (`git add`):

1. Git-Theta reads the checkpoint from the working tree using a plug-in system to support different deep-learning frameworks.
2. Git-Theta converts the checkpoint into a tree of parameter names that map to parameter values.
3. Git-Theta records metadata for each parameter group.
4. Git-Theta compares the metadata for the current parameter group with its previous value.
  a. If the metadata matches, no actions is taken and the previous metadata is used moving forward.
  b. If the metadata doesn't match, the parameter is serialized and then saved using git-lfs. The git-lfs metadata is recorded in the metadata file.
5. The metadata is written to the staging area.

Thus, git itself only tracks the model metadata, actual values are stored efficiently in git-lfs. Additionaly, by checking for matching metadata, only changed parameters are stored.

When a model checkpoint is **smuged** (`git checkout`):

1. The git-theta metadata file is retrieved from git.
2. For each parameter, the Update plug-in system is used to get actual parameter values.
  a. For simple updates the git-lfs metadata is used to get the values directly.
  b. For incremental updates, git-lfs metadata is used to get update values, previous parameter values are retrieved from git itself, and then the update is applied.
4. The real parameter values are written into the working tree, again using the checkpoint plug-in system to handle different deep-learning frameworks.

**Configuraiton:** The following command must be run to configure Git to use Git-Theta.

```bash
git theta install
```

This command adds the following lines to your global `~/.gitconfig`:

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

This configuration defines our two [Git filter drivers](https://git-scm.com/docs/gitattributes#_filter), clean and smuge, and registers them under the name `theta`. Similarly, it defines merge and diff programs, also named `theta`.

At the repository level, we still need to tell git which files are supposed to use the git-theta drivers. This is done with:

```bash
git theta track ${path/to/model}
```

This adds an entry to the `.gitattributes` file to configure git to use git-theta. The new entry looks like

```ini
path/to/model filter=theta merge=theta diff=theta
```

This tells git that anytime a file that matches the pattern `path/to/model` is processed, use the filter/merge/diff driver named `theta`.

### Updates via git

Git-Theta supports a special set of `IncrementalUpdates`. These are updates that are based on the previous version of the parameter values. For example, a spase update to the embedding table doesn't need to be stored as a new copy of the table, instead it can be computed on the fly during a sumdge filter based on the sparse update and the previous value.

`IncrementalUpdate`s include references to the commit that holds the last parameter value in their metadata. Then when the new value is needed, the `IncrementalUpdate` class will fetch the value of the previous parameter *from git* and apply the current update. This yields a massive reduction in storage costs. Additionaly this is done recursively, an `IncrementalUpdate` fetches the previous value using the same Update machinary. Thus previous incremental updates fetch their own previous values until a self-contained update (such as a Dense one) is hit.

### LSH Hashing

To avoid processing parameter groups that have not been changed, git-theta uses parameter hashes when comparing parameters for equality. If the hashes match, the parameters do and we don't need to save a new copy. However, incremental updates and computation on the fly opens the door for small amounts of numerical noise as different machines can use different versions/implementations of core mathematical libraries. This noise results in differences in bit-level hashes.

Therefore, git-theta uses Locality Sensative Hashing, LSH, for parameter hashes, Specifically, an LSH which approximates Euclidean distance. It also uses the random-pool approach to hash parameters of variable sizes.

Git-Theta's LSH uses 16 hash functions and is calibrated so that two parameter groups with a a Euclidean distance less than $1e^{-8}$ will have the same hash with a probability of at least $0.99$. Additionally, weights with a distance $\in [1e{-8}, 1e^{-6}]$ are double-checked with `np.allclose`.

### Merging

Model merging can be done interactivly. When a merge conflict is detected git calls our merge driver program and provides it the driver with 3 files, one representing the model state on each branch and one for the model as it was at the most recent commit shared by both branches. At this point each model is represented as git-theta metadata and only loaded when needed.

The dirver then loops through all the parameters in the model on each branch and see how they are changed. When there are differences, a menu of possible merge operations is presented. Each merge operation is a plug-in that registers what kind of parameter differences it can be used for. For example, it does not make sense to use parameter averaging when a parameter was deleted in one of the branches; therefore, the averaging merge plugin will not appear in the menu.

Once all changed parameters have been merged, the final result is written to disk and commited.

### Diffing

Diffing works similar to merging where only parameter groups that have changed are reported.

# Development Setup

This project uses `black` for code formatting and `isort` for import statement ordering. Additionaly, it includes CI that checks for compliance.

We include pre-commit hooks which will automatically run `black` and `isort` against any python files staged for commit. These hooks can be installed with:

``` sh
$ pip install -r requirements-dev.txt
$ pre-commit install
```

When one of these tools must reformat your file it will show as the pre-commit hook failing and your commit will **not** have happened. Reformatted source files will appear in your working dir and are ready to be re-added to staging (`git add`). Running `git commit -m ${msg}` again will result in the hooks passing and the commit actually happening. *Note:* As your initial commit was blocked, you will probably want to use the same message in the commit that actually goes through.

## Adding new plug-ins

Git-theta makes heavy use of [python plug-ins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/) to support the vast landscape of deep learning frameworks, as well as to provide an entry point for custom use cases.

Git-theta currently support plug-ins for the [`Checkpoint`](https://github.com/r-three/git-theta/blob/main/git_theta/checkpoints/base.py), [`Update`](https://github.com/r-three/git-theta/blob/main/git_theta/updates/base.py), and [`Merge`](https://github.com/r-three/git-theta/blob/main/git_theta/merges/base.py) classes.

Third-party users can register a plug-in by creating a small installable package that defines the plugin and registers it as an entry point under the name scope `git_theta.plugins.(checkpoints|updates|merges)`. An example plugin for JSON formatted checkpoints can be found [here](https://github.com/r-three/git-theta/tree/main/plugins#git-theta-plug-ins).

Alternatively, plug-ins can be added directly to the `git-theta` package by adding new subclasses to the approperiate modules, then declaring it in the the `entry_points` dict in `setup.py`.

# Citation

If you use git-theta in your work; 1) Super Exciting! 2) please cite:

```bibtex
@InProceedings{kandpal-etal-2023-git-theta
    title={Git-Theta: A Git Extension for Collaborative Development of Machine Learning Models},
    author={Kandpal, Nikhil and Lester, Brian and Muqeeth, Mohammed and Mascarenhas, Anisha and Evans, Monty and Baskaran, Vishal and Huang, Tenghao and Liu, Haokun and Raffel, Colin},
    journal={International Conference on Machine Learning, {ICML}},
    year={2023},
    month={july},
    url={},
}
```

# References

[^1]: Yi-Lin Sung, Varun Nair, and Colin Raffel. [“Training Neural Networks with Fixed Sparse Masks.”](https://arxiv.org/abs/2111.09839) NeurIPS 2021.
[^2]: Demi Guo, Alexander M. Rush, and Yoon Kim. [“Parameter-Efficient Transfer Learning with Diff Pruning.”](https://arxiv.org/abs/2012.07463) ACL 2020.
[^3]: Elad Ben-Zaken, Shauli Ravfogel and Yoav Goldberg. [“BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.”](https://arxiv.org/abs/2106.10199) ACL 2022.
[^4]: Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. [“Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning.”](https://arxiv.org/abs/2205.05638) NeurIPS 2022.
[^5]: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen. [“LoRA: Low-Rank Adaptation of Large Language Models.”](https://arxiv.org/abs/2106.09685) ICLR 2022.
[^6]: Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. [“Parameter-Efficient Transfer Learning for NLP.”](https://arxiv.org/abs/1902.00751) ICML 2019.
[^7]: Brian Lester, Rami Al-Rafu, and Noah Constant. ["The power of
scale for parameter-efficient prompt tuning"](https://aclanthology.org/2021.emnlp-main.243) EMNLP 2021.
[^8]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and, Kristine Toutanova. ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://www.aclweb.org/anthology/N19-1423) NAACL 2019.
[^9]: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. ["Exploring the limits of transfer learning with a unified text-to-text transformer"](http://jmlr.org/papers/v21/20-074.html) JMLR 2020.
[^10]: Victor Sanh *et al.* (39 more) ["Multitask Prompted Training Enables Zero-Shot Task Generalization"](http://arxiv.org/abs/2110.08207) ICLR 2022.
