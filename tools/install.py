#!/usr/bin/env python3
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s | %(lineno)-5d | %(levelname).4s | %(message)s",
)

bin_dir = os.path.realpath(
    os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        os.pardir
    )
)

links = {
    'bin/cli.py': 'opt-cli',
    'bin/worker.py': 'opt-worker',
}

for src, dst in links.items():
    src_abs_path = os.path.join(bin_dir, src)
    dst_abs_path = os.path.join('/usr/local/bin', dst)
    if not os.path.exists(dst_abs_path):
        logging.info("Creating %s from %s", dst_abs_path, src_abs_path)
        os.symlink(src_abs_path, dst_abs_path)
