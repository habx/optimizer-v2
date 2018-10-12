#!/bin/sh -x
pip install -q --upgrade pip
pip install -q -r requirements.txt
pytest
pylint libs || true
