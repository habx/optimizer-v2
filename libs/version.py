import os

VERSION = '0.0.0'
version_file = os.path.realpath(
    os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        os.pardir,
        'version.txt'
    )
)
if os.path.exists(version_file):
    with open(version_file, 'r') as fp:
        VERSION = fp.read()
