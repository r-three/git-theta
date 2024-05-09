#!/usr/bin/env bash

source ./utils.sh

# Create an associated array mapping test name to exit code.
declare -A TESTS
# Map testname to the number of times the test was run.
declare -A RUNS

TRIALS=3

# Each sub-directory in the current directory is a test.
for test in ./*/
do
    # Remove things like dir / from the test name
    testname="${test%*/}"
    testname="${testname#./}"
    yellow_echo "Running Test: ${testname}"
    echo "================================================================="
    # Move into the test dir
    pushd ${testname}
    # Run the test up to ${TRIALS} times, stopping when it has a return of 0
    i=0
    # This is just a non-zero value, just to get us into the loop, it doesn't
    # matter what the value is as it will be overwritten by the test return code.
    return_code=-1
    while [[ "${i}" < "${TRIALS}" && ${return_code} != 0 ]]; do
        # If there is a local clean script run that, otherwise run the global one.
        # This is in the loop to ensure it is cleaned before each attempt.
        if [[ -f ./clean.sh ]]; then
            ./clean.sh
        else
            ../clean.sh
        fi
        if [[ "${i}" > 0 ]]; then
            red_echo "${testname} failed, running trial $((i + 1))"
        fi
        ./test.sh
        return_code="${?}"
        i=$((i + 1))
    done
    # Save the tests return value
    TESTS[${testname}]="${return_code}"
    # Save the number of times a test had to be run.
    RUNS[${testname}]="${i}"
    # Cleanup after the test, again, look for a local clean and use if found.
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
        green_echo "${test}" "n"
        # Check if we had to re-run tests.
        if [[ "${RUNS[$test]}" != 1 ]]; then
            yellow_echo " (Had to run test '${test}' ${RUNS[$test]} times to pass)."
        else
            echo
        fi
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
