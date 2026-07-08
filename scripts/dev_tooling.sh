#!/bin/bash

# this script is for development tooling
set -e  # exit immediately if a command exits with a non-zero status

# function to run linters
run_linters() {
    echo "running linters"
    flake8 src/  # check for style guide enforcement
    black --check src/  # check for code formatting
}

# function to run tests
run_tests() {
    echo "running tests"
    pytest tests/  # execute the test suite
}

# function to build and run docker container
run_docker() {
    echo "building and running docker container"
    docker-compose up --build  # build and start the services defined in docker-compose.yml
}

# function to display help
show_help() {
    echo "usage: ./dev_tooling.sh [command]"
    echo "commands:"
    echo "  lint     run linters"
    echo "  test     run tests"
    echo "  docker   build and run docker container"
    echo "  help     display this help message"
}

# check if no argument is given
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# main logic to handle commands
case $1 in
    lint)
        run_linters
        ;;
    test)
        run_tests
        ;;
    docker)
        run_docker
        ;;
    help)
        show_help
        ;;
    *)
        echo "unknown command: $1"
        show_help
        exit 1
        ;;
esac

# TODO: consider adding more commands in the future