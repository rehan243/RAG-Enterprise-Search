#!/bin/bash

# check for required tools
command -v flake8 >/dev/null 2>&1 || { echo >&2 "flake8 is not installed. Aborting."; exit 1; }
command -v pytest >/dev/null 2>&1 || { echo >&2 "pytest is not installed. Aborting."; exit 1; }

# function for linting
lint() {
    echo "running flake8 for linting"
    flake8 src/
    if [ $? -ne 0 ]; then
        echo "linting failed, fix issues before proceeding"
        exit 1
    fi
    echo "linting passed"
}

# function for testing
test() {
    echo "running pytest for tests"
    pytest tests/
    if [ $? -ne 0 ]; then
        echo "tests failed, fix issues before proceeding"
        exit 1
    fi
    echo "all tests passed"
}

# main function
main() {
    lint
    test
    echo "development checks completed successfully"
}

# run main function
main