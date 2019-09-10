import os

# BD (2019-05-21): Ces tests n'ont pas l'air de marcher
# FC (2019-09-05): They should


def test_optimizer_lib_as_cli():
    # OPT-119: Making sure the optimizer.py still works
    assert not os.system('PYTHONPATH=$(pwd) libs/optimizer.py')


def test_optimizer_cli():
    assert not os.system('PYTHONPATH=$(pwd) bin/cli.py'
                         ' -b resources/blueprints/016.json'
                         ' -s resources/specifications/016_setup0.json'
                         ' -p resources/params/timeout_15s.json')
