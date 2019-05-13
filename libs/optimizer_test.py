import os

# OPT-119: Making sure the optimizer.py still works
rc = os.system('PYTHONPATH=$(pwd) libs/optimizer.py')

assert rc == 0
