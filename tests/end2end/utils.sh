#!/usr/bin/env bash


GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NORMAL="\033[0m"

function color_echo {
    local text="${1}"
    local color="${2}"
    local newline="${3}"
    # If we provide a 3rd argument, don't include the newline, this lets us
    # do two colors on one line easily.
    if [[ -z "${newline}" ]]; then
      echo -e "${color}${text}${NORMAL}"
    else
      echo -en "${color}${text}${NORMAL}"
    fi
}

function green_echo {
    local text="${1}"
    local newline="${2}"
    color_echo "${text}" "${GREEN}" "${newline}"
}

function red_echo {
    local text="${1}"
    local newline="${2}"
    color_echo "${text}" "${RED}" "${newline}"
}

function yellow_echo {
    local text="${1}"
    local newline="${2}"
    color_echo "${text}" "${YELLOW}" "${newline}"
}

function make_repo {
    echo "Making Git Repo."
    git init 2> /dev/null
    git branch -m main
    # Set the git user/email for the generated test repo
    git config --local user.email "git-theta-tester@example.com"
    git config --local user.name "Git Theta Tester"
}

function test_init {
    make_repo
    if [[ ! ${-} =~ e ]]; then
        red_echo "It seems that 'set -e' was not done in this test. This makes it easy for a test to fail but appear to pass. Please add 'set -e' and do specific error handling if a part of your test is allowed to fail."
        exit 1
    fi
}

function commit {
    local msg="${1}"
    git commit -m "${msg}" > /dev/null
    echo $(git rev-parse HEAD)
}
