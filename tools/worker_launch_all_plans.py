#!/usr/bin/env python3

import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s | %(lineno)-5d | %(levelname).4s | %(message)s",
)

for i in range(1, 1000):
    blueprint_path = 'resources/blueprints/%03d.json' % i
    setup_path = 'resources/specifications/%03d_setup0.json' % i
    if not os.path.exists(blueprint_path):
        break
    cmd = "bin/worker.py -b {blueprint_path} -s {setup_path}".format(
        blueprint_path=blueprint_path,
        setup_path=setup_path,
    )
    logging.info("Executing:\n%s", cmd)
    os.system(cmd)

