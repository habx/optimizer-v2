#!/bin/sh -x
pip install -q --upgrade pip
pip install -q -r requirements.txt
# Synthetic view
pytest
# Verbose view
pytest -v
pylint libs || true
