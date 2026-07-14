#!/bin/bash

# this script is for running development tasks like linting and testing

set -e  # exit on error

# function to run linter
run_linter() {
    echo "running linter..."
    flake8 src/  # check the src directory for style issues
    echo "linting complete"
}

# function to run tests
run_tests() {
    echo "running tests..."
    pytest tests/  # execute all tests in the tests directory
    echo "all tests passed"
}

# function to build docker image
build_docker() {
    echo "building docker image..."
    docker build -t rag-enterprise-search .  # build the image with the current context
    echo "docker image built"
}

# parse command line arguments
case "$1" in
    lint)
        run_linter
        ;;
    test)
        run_tests
        ;;
    docker)
        build_docker
        ;;
    all)
        run_linter
        run_tests
        build_docker
        ;;
    *)
        echo "usage: $0 {lint|test|docker|all}"  # show usage info if no valid option is provided
        exit 1
        ;;
esac