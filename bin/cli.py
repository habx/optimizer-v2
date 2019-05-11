#!/usr/bin/env python3
# coding=utf-8
"""
Command line interpreter to run Optimizer
Please note you can install it simply by performing:
ln -s $(pwd)/bin/cli.py /usr/local/bin/optimizer
And then use it freely:
$ cd resources
$ optimizer \
    -l blueprints/011.json \
    -s specifications/011_setup0.json \
    -o /tmp
"""

import argparse
import logging
import json
import os
import tempfile

import uuid

from libs.executor.executor import Executor, TaskDefinition


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
    example_text = """
Example usage:
==============

bin/cli.py -b resources/blueprints/001.json -s resources/specifications/001_setup0.json
bin/cli.py -b resources/blueprints/001.json -s resources/specifications/001_setup0.json \
           -p resources/params/timeout.json
"""

    parser = argparse.ArgumentParser(
        description="Optimizer V2 CLI (%s)" % Executor.VERSION,
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-b", dest="blueprint", required=True, metavar="FILE",
                        type=lambda x: _exists_path(parser, x, True),
                        help="the input blueprint file path")
    parser.add_argument("-s", dest="setup", required=True, metavar="FILE",
                        type=lambda x: _exists_path(parser, x, True),
                        help="the input setup file path")
    parser.add_argument("-p", dest="params", required=False, metavar="FILE",
                        type=lambda x: _exists_path(parser, x, True),
                        help="the input params file path")
    parser.add_argument("-o", dest="output", required=False,
                        help="the output solutions dir")
    parser.add_argument("-g", dest="grid", required=False,
                        help="grid type")
    parser.add_argument("-u", dest="shuffle", required=False,
                        help="shuffle type")
    parser.add_argument("-P", "--plot", dest="plot",
                        help="plot outputs",
                        action="store_true")
    parser.add_argument("-t", "--task-id", dest="task_id", help="specify a task ID", required=False)
    args = parser.parse_args()
    blueprint_path: str = args.blueprint
    setup_path: str = args.setup
    params_path: str = args.params
    output_dir: str = args.output
    do_plot: bool = args.plot
    task_id: str = args.task_id

    if not output_dir:
        output_dir = tempfile.mkdtemp('opt-cli')
        logging.info("Using \"%s\" as output dir", output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # run
    logging.getLogger().setLevel(logging.INFO)
    executor = Executor()

    logging.info('Running (%s, %s) --> %s', blueprint_path, setup_path, output_dir)

    td = TaskDefinition()
    td.task_id = task_id

    if td.task_id:
        td.task_id = str(uuid.uuid4())

    with open(blueprint_path, 'r') as blueprint_fp:
        td.blueprint = json.load(blueprint_fp)
    with open(setup_path, 'r') as setup_fp:
        td.setup = json.load(setup_fp)
    if params_path:
        with open(params_path, 'r') as params_fp:
            td.params = json.load(params_fp)
    else:
        td.params = {}

    if do_plot:
        td.params['do_plot'] = True

    if output_dir:
        td.local_context.output_dir = output_dir

    td.check()

    response = executor.run(td)

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
