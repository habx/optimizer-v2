#!/bin/sh -xe

# This file can be executed on drone or locally by using:
# $ docker build . -t test && docker run test

# Cutting tests
tests=$(circleci tests glob "**/*_test.py" | circleci tests split) ||:

# requirements installation is done in CI
#pip install -q --upgrade pip
#pip install -q -r requirements.txt

# Verbose view
pytest --durations=10 -v $tests

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 pytest --durations=10 -v $tests

# To compare the packages
pip list

pylint libs || true
