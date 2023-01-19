## How Is Git-Theta Different To All The Other VCS-For-ML systems?

There are many VCS-For-ML systems already in existence. Some examples:

- [Data Version Control (from iterative.ai)](https://dvc.org/)

- [MLFlow](https://mlflow.org/)

- [Weights & Biases](https://docs.wandb.ai/guides/data-and-model-versioning/model-versioning)

- [HuggingFace](https://huggingface.co/docs/hub/repositories-getting-started (definitely just git-lfs under the hood))

- [Neptune.AI](https://neptune.ai/blog/version-control-for-ml-models)

- [Datatron](https://datatron.com/how-it-works/)

These are built to manage the chaos of ML model development: tracking experiments, hyper-parameters, and datasets.

They enable **reproducibility**: if Alice tracks the corresponding code, training data and compute environment for each experiment she runs, and can communicate the history of each to Bob, then Bob can exactly reproduce each of her resultant models.
They typically use *git-lfs* to store datasets and model checkpoints, and provide useful caching and remote storage options to enable fast switching between models / datasets.

Here are a few quotes directly from their docs:

- *"DVC is built to make ML models shareable and reproducible."*
- *"MLflow is an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry."*
- *"With automatic saving and versioning, each experiment you run stores the most recently trained model artifact to W&B. You can scroll through all these model versions, annotating and renaming as necessary while maintaining the development history. Know exactly which experiment code and configuration generated which weights and architecture. You and your team can download and restore any of your model checkpoints—across projects, hardware, and dev environments."*
- *"Models, Spaces, and datasets are hosted on the Hugging Face Hub as Git repositories, which means that version control and collaboration are core elements of the Hub. In a nutshell, a repository (also known as a repo) is a place where code and assets can be stored to back up your work, share it with the community, and work in a team."*

They *do not* attempt to compute diffs between model checkpoints, merge models, or indeed engage with models in any way other than as binary blobs stemming from a managed, versioned training process: this is precisely the void Git-Theta tries to fill.

If you know of or are involved in any prior work with similar goals, please let us know!
