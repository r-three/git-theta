#!/usr/bin/env bash
# checkout test: Test that we are able to checkout a git-theta model

source ../utils.sh

set -e

test_init

MODEL_SCRIPT=model.py
MODEL_NAME=model.pt

echo "Making model ${MODEL_NAME}"
INIT_MODEL=`python ../${MODEL_SCRIPT} --action init --seed 1337 --model-name=${MODEL_NAME}`

echo "Installing git-theta and tracking ${MODEL_NAME}"
git theta install
git theta track ${MODEL_NAME}

echo "Adding ${MODEL_NAME} to git repo."
git add ${MODEL_NAME}
echo "Committing ${MODEL_NAME} to git repo."
SHA=$(commit "first commit")
echo "Initial model commit was at ${SHA}"

echo "Making Dense update to ${MODEL_NAME}"
DENSE_MODEL=`python ../${MODEL_SCRIPT} --action dense --seed 42 --model-name=${MODEL_NAME}`

echo "Adding Dense update to git repo."
git add ${MODEL_NAME}
echo "Committing dense update to repo."
commit "updated model"

echo "Checking out initial model at ${SHA}"
git checkout ${SHA}
echo "Comparing checked out model (${MODEL_NAME}) to original save (${INIT_MODEL})."
python ../verify.py --old-model "${MODEL_NAME}" --new-model "${INIT_MODEL}"

echo "Checking out the dense update."
git checkout main
echo "Comparing checked out model (${MODEL_NAME}) to original save (${DENSE_MODEL})."
python ../verify.py --old-model "${MODEL_NAME}" --new-model "${DENSE_MODEL}"

green_echo "git checkout test passed!"
