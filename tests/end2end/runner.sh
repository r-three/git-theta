#!/usr/bin/env bash

source ./utils.sh

# Create an associated array mapping test name to exit code.
declare -A TESTS

for test in ./*/
do
    testname="${test%*/}"
    testname="${testname#./}"
    yellow_echo "Running Test: ${testname}"
    echo "============================================================"
    pushd ${testname}
    if [[ -f ./clean.sh ]]; then
        ./clean.sh
    else
        ../clean.sh
    fi
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
    # Passes tests have return values of 0. Summing all passed test results in
    # a 0 return value for the whole running. If one of the tests has a non-zero
    # return value, the runner will have a non-zero value (return value beyond
    # non-zero is not meaningful).
    FAILED+="${TESTS[$test]}"
done

exit "${FAILED}"
