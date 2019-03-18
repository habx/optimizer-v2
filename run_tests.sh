#!/bin/sh -xe

# This file can be executed on drone or locally by using:
# $ docker build . -t test && docker run test

pip install -q --upgrade pip
pip install -q -r requirements.txt

# Cutting tests
tests=$(circleci tests glob "**/*_test.py") ||:

# Verbose view
pytest -v $tests

# To compare the packages
pip list

pylint libs || true
