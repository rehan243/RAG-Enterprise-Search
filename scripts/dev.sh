#!/bin/bash

# this script is for local development tasks

# function to run linter
run_linter() {
    echo "running linter..."
    flake8 src/ tests/ # check code style
    if [ $? -ne 0 ]; then
        echo "linting failed, fix issues above"
        exit 1
    fi
    echo "linting passed!"
}

# function to run tests
run_tests() {
    echo "running tests..."
    pytest tests/ # run tests with pytest
    if [ $? -ne 0 ]; then
        echo "tests failed, check the output above"
        exit 1
    fi
    echo "all tests passed!"
}

# function to build docker image
build_docker() {
    echo "building docker image..."
    docker build -t rag-enterprise-search . # build the image
    if [ $? -ne 0 ]; then
        echo "docker build failed, fix issues above"
        exit 1
    fi
    echo "docker image built successfully!"
}

# function to run the application locally
run_local() {
    echo "starting local app..."
    python src/app.py # replace with your app entry point
}

# main function to handle script arguments
main() {
    case $1 in
        lint)
            run_linter
            ;;
        test)
            run_tests
            ;;
        docker)
            build_docker
            ;;
        run)
            run_local
            ;;
        *)
            echo "usage: $0 {lint|test|docker|run}"
            exit 1
            ;;
    esac
}

# execute main with all script args
main "$@"