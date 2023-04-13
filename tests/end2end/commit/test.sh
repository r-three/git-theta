#!/usr/bin/env bash
# Test that we are able to checkout a git-theta model

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
commit "first commit"

echo "Comparing model (${MODEL_NAME}) to original save (${INIT_MODEL})."
python ../verify.py --old-model "${MODEL_NAME}" --new-model "${INIT_MODEL}"

echo "Making sure ${MODEL_NAME} is committed."
FILES=$(git ls-files)
if [[ ! ${FILES} =~ ${MODEL_NAME} ]]; then
    red_echo "${MODEL_NAME} not found in 'git ls-files'."
    exit 1
fi

green_echo "git commit test passed!"
