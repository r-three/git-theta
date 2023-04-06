#!/usr/bin/env bash
# Make a test skeleton.

TEST_NAME=${1}

if [[ -z ${TEST_NAME} ]]; then
    echo "usage: make-test.sh 'test-name'"
    exit 1
fi

echo "Making skeleton for a test named ${TEST_NAME}"
mkdir ${TEST_NAME}
pushd ${TEST_NAME} > /dev/null

cat <<EOF > test.sh
#!/usr/bin/env bash
# ${TEST_NAME} test: TODO Add description of test.

source ../utils.sh

set -e

test_init
EOF

chmod +x test.sh
