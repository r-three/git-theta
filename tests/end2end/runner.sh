#!/usr/bin/env bash

source ./utils.sh

# Create an associated array mapping test name to exit code.
declare -A TESTS

# Setup Git Identity for tests
git config --global user.email "git-theta-tester@example.com"
git config --global user.name "Git Theta Tester"

for test in ./*/
do
    testname="${test%*/}"
    testname="${testname#./}"
    yellow_echo "Running Test: ${testname}"
    echo "============================================================"
    pushd ${testname}
    ./test.sh
    TESTS[${testname}]=${?}
    if [[ -f ./clean.sh ]]; then
        ./clean.sh
    else
        ../clean.sh
    fi
    popd
done

FAILED=0
echo "Test Summary:"
for test in "${!TESTS[@]}"
do
    if [[ "${TESTS[$test]}" == 0 ]]; then
        green_echo "${test}"
    else
        red_echo "${test}"
    fi
    FAILED+="${TESTS[$test]}"
done

exit "${FAILED}"
