import os

# This is only an issue on my computer
if 'MPLBACKEND' not in os.environ:
    os.environ['MPLBACKEND'] = 'svg'
