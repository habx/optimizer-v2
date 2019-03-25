#!/usr/bin/env python3
# coding=utf-8
"""
Command line interpreter to run Optimizer
Please note you can install it simply by performing:
ln -s $(pwd)/bin/cli.py /usr/local/bin/optimizer
And then use it freely:
"""

import argparse
import logging
import json
import os
import libpath
from libs.utils.executor import Executor


def _exists_file(parser, arg):
    if not os.path.exists(arg):
        return parser.error("The file %s does not exist!" % arg)
    return arg


def _cli():
    # arg parser
    parser = argparse.ArgumentParser(description="Optimizer V2")
    parser.add_argument("-l", dest="lot", required=True, metavar="FILE",
                        type=lambda x: _exists_file(parser, x),
                        help="the input lot file path")
    parser.add_argument("-s", dest="setup", required=True, metavar="FILE",
                        type=lambda x: _exists_file(parser, x),
                        help="the input setup file path")
    parser.add_argument("-o", dest="output", required=True,
                        help="the input solution dir")
    parser.add_argument("-g", dest="grid", required=False,
                        help="grid type", default="optimal_grid")
    parser.add_argument("-u", dest="shuffle", required=False,
                        help="shuffle type", default="square_shape_shuffle_rooms")
    parser.add_argument("-p", "--plot",
                        help="plot outputs",
                        action="store_true")
    args = parser.parse_args()
    lot_path = args.lot
    setup_path = args.setup
    output_dir = args.output
    grid_type = args.grid
    shuffle_type = args.shuffle
    do_plot = bool(args.plot)

    # run
    logging.getLogger().setLevel(logging.INFO)
    executor = Executor()
    executor.set_execution_parameters(grid_type=grid_type,
                                      shuffle_type=shuffle_type,
                                      do_plot=do_plot)
    logging.info('Running (%s, %s)', lot_path, setup_path)
    response = executor.run_from_file_paths(lot_path, setup_path)
    for i, solution in enumerate(response.solutions):
        solution_path = os.path.join(output_dir, "output{}.json".format(i))
        json.dump(solution, solution_path, indent=2, sort_keys=True)

# I don't think it makes any sense here: https://stackoverflow.com/a/419185
if __name__ == "__main__":
    _cli()
