#!/bin/sh
virtualenv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
