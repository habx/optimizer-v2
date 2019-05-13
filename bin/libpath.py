import os
import sys


# There has to be a better way to do this.
# OPT-119: Fix around this bad implementation
def add_local_libs():
    path = os.path.realpath( os.path.join( os.path.dirname( os.path.realpath(__file__) ),os.pardir))

    if not path in sys.path:
        sys.path.append(path)


add_local_libs()

