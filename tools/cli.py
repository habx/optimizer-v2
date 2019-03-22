# coding=utf-8
"""
Command line interpreter to run Optimizer
"""

import os


def exists_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

if __name__ == '__main__':
    import argparse
    import logging
    from libs.utils.executor import Executor

    # arg parser
    parser = argparse.ArgumentParser(description="Optimizer V2")
    parser.add_argument("-l", dest="lot", required=True, metavar="FILE",
                        type=lambda x: exists_file(parser, x),
                        help="the input lot file path")
    parser.add_argument("-s", dest="setup", required=True, metavar="FILE",
                        type=lambda x: exists_file(parser, x),
                        help="the input setup file path")
    parser.add_argument("-g", dest="grid", required=False,
                        help="grid type", default="optimal_grid")
    parser.add_argument("-u", dest="shuffle", required=False,
                        help="shuffle type", default="square_shape_shuffle_rooms")
    args = parser.parse_args()
    lot_path = args.lot
    setup_path = args.setup
    grid_type = args.grid
    shuffle_type = args.shuffle

    # run
    logging.getLogger().setLevel(logging.INFO)
    executor = Executor()
    executor.set_execution_parameters(grid_type=grid_type, shuffle_type=shuffle_type)
    response = executor.run_from_file_paths(lot_path, setup_path)
    logging.info("Time: %i", int(response.elapsed_time))
    logging.info("Nb solutions: %i", len(response.solutions))
