#!/bin/sh -xe

# This file can be executed on drone or locally by using:
# $ docker build . -t test && docker run test

# Cutting tests
tests=$(circleci tests glob "**/*_test.py" | circleci tests split) ||:

# Verbose view
pytest --durations=10 -v $tests

# To compare the packages
pip list
