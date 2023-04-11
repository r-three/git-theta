#!/usr/bin/env bash
# low-rank test: Test committing and checking out low-rank updates with
#                side-loaded information.

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

echo "Making a low-rank update to ${MODEL_NAME}"
LOW_RANK_MODEL=`python ../${MODEL_SCRIPT} --action low-rank --seed 42 --model-name=${MODEL_NAME}`
git theta add ${MODEL_NAME} --update-type="low-rank" --update-data="low-rank-data.pt"
LOW_RANK_SHA=$(commit "low-rank")

echo "Checking out initial model at ${SHA}"
git checkout ${SHA}
echo "Comparing checked out model (${MODEL_NAME}) to original save (${INIT_MODEL})"
python ../verify.py --new-model "${MODEL_NAME}" --old-model "${INIT_MODEL}"

echo "Checking out the low-rank update."
git checkout main
echo "Comparing checked out model (${MODEL_NAME}) to original save (${SPARSE_MODEL})"
python ../verify.py --new-model "${MODEL_NAME}" --old-model "${LOW_RANK_MODEL}"

echo "Verify that 'git status' doesn't have a diff."
git diff-index --quiet HEAD --
if [[ "$?" != 0 ]]; then
    exit 1
fi

green_echo "low-rank update test passed!"
