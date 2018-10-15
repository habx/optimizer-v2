#!/bin/sh -xe

# This file can be executed on drone or locally by using:
# $ docker build . -t test && docker run test

pip install -q --upgrade pip
pip install -q -r requirements.txt

# Verbose view
pytest -v --duration 1

pylint libs || true
