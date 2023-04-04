# Git-Theta End-2-End tests.

This directory contains a collection of end-2-end tests for git-theta.

## Running the Tests

Each subdirectory represents a test that actually interacts with git. The `runner.sh` script is responsible for running them and reports if they passed for failed. All tests can be run with `./runner.sh`. Additionally, the `.github/workflows/end2endtest.yml` configures these tests to run via GitHub Actions.

## Anatomy of a Test

Each test is a directory and includes a `test.sh` script. Execution of this script runs the test and pass/fail is determined by its exit code (`0` means pass),

Each test also includes a `clean.sh` script. This is run before and after the test to clean up any generated artifacts. This script should never have a non-zero exit code and should be idempotent (we can run it multiple times).

The command `make-test.sh "${test name}"` can be used to create the skeleton of a new test.

### `test.sh`, the (Vegan-)?Meat and Potatoes

The first steps of a `test.sh` file include sourcing the `../utils.sh` so it can use some of our shared functions. Then it should run `set -e` to ensure that errors in part of the test cause the whole test to fail. It should also call `test_init`, which will create a git repo (with a `main` branch) for it and ensure that `set -e` was used.

The next step is often to create and modify some model. The provided `../model.py` file can be used as a helper to create new models and updates with special forms. This script creates two copies of the model, one that lives in the path that is version controlled and one that includes information about how it was created. This path is returned on stdout, and can be captured in `test.sh`. This checkpoint is not version controlled, uses the deep-learning framework native checkpoint format, and should be removed by the clean script.

The provided `verify.py` can be used to help check that version controlled models match their original values.

`../utils.sh` provides a `commit` function that will create a git commit and return the hash for that commit, for example `SHA=$(commit "commit msg")` will make a commit and save the hash to `${SHA}`. It can be called with just `commit "commit msg"` if you do not want to track the hash. This should help for tests that travel through git history.

> **Warning**
> The `set -e` option causes a whole bash script to exit if one of the subcommands in it has error/exits with a non `0` return code. Without this setting, tests that have failing steps will often appear as passing so we require this to be active. If your test has subcommands that are expected to fail and your test accounts for that correctly, you can suppress this `-e` behavior by adding `${subcommand} || true` to your failing commands (or call `set +e` after the `test_init` call). Both of **these settings can cause tests to fail silently**, so only use them if you are sure you understand what they are doing.
