#!/usr/bin/env bash
# smudge test: This tests that a metadata file (produced by the clean filter)
#   can be used directly with the smudge filter to re-create the checkpoint,
#   regardless of which commit is currently checked out.
#
#   This behavior makes some parts of merging easier so we want to ensure that
#   it holds. Additionally, this property makes working with checkpoints/git
#   in git theta easier.

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

git show ${SHA}:${MODEL_NAME} > init-metadata.json

echo "Making Sparse Update to ${MODEL_NAME}"
SPARSE_MODEL=`python ../${MODEL_SCRIPT} --action sparse --seed 1234 --model-name=${MODEL_NAME}`
echo "Adding Sparse Update to the git repo."
git add ${MODEL_NAME}
echo "Commiting sparser update to repo."
SPARSE_SHA=$(commit "sparse update")

git show ${SPARSE_SHA}:${MODEL_NAME} > sparse-metadata.json

echo "Making Dense update to ${MODEL_NAME}"
DENSE_MODEL=`python ../${MODEL_SCRIPT} --action dense --seed 42 --model-name=${MODEL_NAME}`

echo "Adding Dense update to git repo."
git add ${MODEL_NAME}
echo "Committing dense update to repo."
commit "updated model"

git show HEAD:${MODEL_NAME} > dense-metadata.json


echo "Verifying smudges at the dense update commit."
echo "Verifying dense model"
git-theta-filter smudge ${MODEL_NAME} < dense-metadata.json > dense-model.pt
python ../verify.py --old-model ${DENSE_MODEL} --new-model dense-model.pt
echo "Verifying sparse model"
git-theta-filter smudge ${MODEL_NAME} < sparse-metadata.json > sparse-model.pt
python ../verify.py --old-model ${SPARSE_MODEL} --new-model sparse-model.pt
echo "Verifying initial model"
git-theta-filter smudge ${MODEL_NAME} < init-metadata.json > init-model.pt
python ../verify.py --old-model ${INIT_MODEL} --new-model init-model.pt

git checkout ${SPARSE_SHA}
echo "Verifying smudges at the sparse update commit."
echo "Verifying dense model"
git-theta-filter smudge ${MODEL_NAME} < dense-metadata.json > dense-model.pt
python ../verify.py --old-model ${DENSE_MODEL} --new-model dense-model.pt
echo "Verifying sparse model"
git-theta-filter smudge ${MODEL_NAME} < sparse-metadata.json > sparse-model.pt
python ../verify.py --old-model ${SPARSE_MODEL} --new-model sparse-model.pt
echo "Verifying initial model"
git-theta-filter smudge ${MODEL_NAME} < init-metadata.json > init-model.pt
python ../verify.py --old-model ${INIT_MODEL} --new-model init-model.pt

git checkout ${SHA}
echo "Verifying smudges at the initial commit."
echo "Verifying dense model"
git-theta-filter smudge ${MODEL_NAME} < dense-metadata.json > dense-model.pt
python ../verify.py --old-model ${DENSE_MODEL} --new-model dense-model.pt
echo "Verifying sparse model"
git-theta-filter smudge ${MODEL_NAME} < sparse-metadata.json > sparse-model.pt
python ../verify.py --old-model ${SPARSE_MODEL} --new-model sparse-model.pt
echo "Verifying initial model"
git-theta-filter smudge ${MODEL_NAME} < init-metadata.json > init-model.pt
python ../verify.py --old-model ${INIT_MODEL} --new-model init-model.pt
