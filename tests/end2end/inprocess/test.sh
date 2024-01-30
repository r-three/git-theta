#!/usr/bin/env bash
# inprocess test: TODO Add description of test.

source ../utils.sh

set -e

test_init

MODEL_SCRIPT=model.py
MODEL_NAME=og_model.pt

echo "Making model ${MODEL_NAME}"
INIT_MODEL=`python ../${MODEL_SCRIPT} --action init --seed 1337 --model-name=${MODEL_NAME}`

python test.py
if [[ "$?" != 0 ]]; then
    exit 1
fi
echo "Verifying that the same model saved in different paths match"
python ../verify.py --old-model should_match_1.pt --new-model should_match_2.pt
echo "Verifying that the changed model, which was the same path, but committed later, is different."
R=$(python ../verify.py --old-model should_match_1.pt --new-model no_match.pt 2> /dev/null || true)
if [[ "$R" == 0 ]]; then
   exit 1
fi

green_echo "in-process test passed!"
