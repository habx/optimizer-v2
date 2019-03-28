#!/usr/bin/env python3
# coding=utf-8
"""
Command line interpreter to run Optimizer
Please note you can install it simply by performing:
ln -s $(pwd)/bin/cli.py /usr/local/bin/optimizer
And then use it freely:
$ cd resources
$ optimizer \
    -l blueprints/paris-mon18_A601.json \
    -s specifications/paris-mon18_A601_setup0.json \
    -o /tmp
"""

import argparse
import logging
import json
import os
import libpath
from libs.utils.executor import Executor


def _exists_path(parser, path, file=None):
    if not os.path.exists(path):
        return parser.error("Path %s does not exist!" % path)

    if file is not None:
        if file and not os.path.isfile(path):
            return parser.error("Not a file: %s" % path)
        if not file and not os.path.isdir(path):
            return parser.error("Not a dir: %s" % path)

    return path


def _cli():
    # arg parser
    parser = argparse.ArgumentParser(description="Optimizer V2 CLI (%s)" % Executor.VERSION)
    parser.add_argument("-l", dest="lot", required=True, metavar="FILE",
                        type=lambda x: _exists_path(parser, x, True),
                        help="the input lot file path")
    parser.add_argument("-s", dest="setup", required=True, metavar="FILE",
                        type=lambda x: _exists_path(parser, x, True),
                        help="the input setup file path")
    parser.add_argument("-p", dest="params", required=False, metavar="FILE",
                        type=lambda x: _exists_path(parser, x, True),
                        help="the input params file path")
    parser.add_argument("-o", dest="output", required=True,
                        type=lambda x: _exists_path(parser, x, False),
                        help="the output solutions dir")
    parser.add_argument("-g", dest="grid", required=False,
                        help="grid type", default="optimal_grid")
    parser.add_argument("-u", dest="shuffle", required=False,
                        help="shuffle type", default="square_shape_shuffle_rooms")
    parser.add_argument("-P", "--plot",
                        help="plot outputs",
                        action="store_true")
    args = parser.parse_args()
    lot_path = args.lot
    setup_path = args.setup
    params_path = args.params
    output_dir = args.output

    # run
    logging.getLogger().setLevel(logging.INFO)
    executor = Executor()

    logging.info('Running (%s, %s) --> %s', lot_path, setup_path, output_dir)

    with open(lot_path, 'r') as lot_fp:
        lot = json.load(lot_fp)
    with open(setup_path, 'r') as setup_fp:
        setup = json.load(setup_fp)
    if params_path:
        with open(params_path, 'r') as params_fp:
            params = json.load(params_fp)
    else:
        params = {}

    if args.plot:
        params['do_plot'] = True

    response = executor.run(lot, setup, params)

    meta = {
        "times": response.elapsed_times
    }

    with open(os.path.join(output_dir, "meta.json"), 'w') as meta_fp:
        json.dump(meta, meta_fp, indent=2, sort_keys=True)

    for i, solution in enumerate(response.solutions):
        solution_path = os.path.join(output_dir, "solution_%d.json" % i)
        with open(solution_path, 'w') as solution_fp:
            json.dump(solution, solution_fp, indent=2, sort_keys=True)


if __name__ == "__main__":
    _cli()
