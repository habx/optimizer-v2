import os

# OPT-119: Making sure the optimizer.py file still works

rc = os.system('PYTHONPATH=$(pwd) libs/optimizer.py')

assert rc == 0
