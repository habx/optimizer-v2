#!/usr/bin/env python3
import argparse
import json
import logging
import logging.handlers
import os
import socket
import uuid
import sentry_sdk

# OPT-119 & OPT-120: Dirty path handling
import libpath

from libs.executor.executor import Executor
from libs.worker.config import Config
from libs.worker.core import TaskDefinition, TaskProcessor
from libs.worker.dev import local_dev_hack
from libs.worker.mq import Exchanger

import config
import habx_logger

logging.root = habx_logger.HabxLogger(config.from_file())
#logging.setLevel(logging.INFO)

# OPT-120: Only to make sure libpath won't be removed
libpath.add_local_libs()


def _process_messages(args: argparse.Namespace, config: Config, exchanger: Exchanger):
    """Make it run. Once called it never stops."""

    logging.info(f'Optimizer V2 Worker ({Executor.VERSION})')
    # We need to both consume and produce
    exchanger.prepare(consumer=True, producer=True)

    processor = TaskProcessor(config, args.target)
    processor.prepare()

    while True:
        msg = exchanger.get_request()
        if not msg:  # No message received (queue is empty)
            continue

        # OPT-99: We shall NOT modify the source data
        td = TaskDefinition.from_json(msg.content.get('data'))

        # Declaring as task_id
        td.task_id = msg.content.get('taskId')

        if not td.task_id:  # Drop it at some point
            td.task_id = msg.content.get('requestId')

        result = processor.process_task(td)

        exchanger.send_result(result)

        # Always acknowledging messages
        exchanger.acknowledge_msg(msg)


def _send_message(args: argparse.Namespace, exchanger: Exchanger):
    """Core sending message function"""

    # We only need a producer, not a consumer
    exchanger.prepare(consumer=False, producer=True)

    # Reading the input files
    with open(args.blueprint) as blueprint_fp:
        lot = json.load(blueprint_fp)
    with open(args.setup) as setup_fp:
        setup = json.load(setup_fp)
    if args.params:
        with open(args.params) as params_fp:
            params = json.load(params_fp)
    else:
        params = {}

    if args.params_crash:
        params['crash'] = True

    if args.target:
        params['target_worker'] = args.target

    # Preparing a request
    request = {
        'type': 'optimizer-processing-request',
        'from': 'worker-optimizer:sender',
        'taskId': str(uuid.uuid4()),
        'data': {
            'lot': lot,
            'setup': setup,
            'params': params,
            'context': {
                'sender-version': Executor.VERSION,
            },
        },
    }

    # Sending it
    exchanger.send_request(request)


def _cli():
    """CLI orchestrating function"""
    local_dev_hack()

    # We're using AWS_REGION at habx and boto3 expects AWS_DEFAULT_REGION
    if 'AWS_DEFAULT_REGION' not in os.environ and 'AWS_REGION' in os.environ:
        os.environ['AWS_DEFAULT_REGION'] = os.environ['AWS_REGION']

    config = Config()
    exchanger = Exchanger(config)

    example_text = """
Example usage:
==============

# Will process any task coming its way
bin/worker.py 

# Will send a task towards processing workers
bin/worker.py -b resources/blueprints/001.json -s resources/specifications/001_setup0.json
bin/worker.py -b resources/blueprints/001.json -s resources/specifications/001_setup0.json \
              -p resources/params/timeout_2s.json
    """

    parser = argparse.ArgumentParser(
        description="Optimizer V2 Worker v" + Executor.VERSION,
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-b", "--blueprint", dest="blueprint", metavar="FILE",
                        help="Blueprint input file")
    parser.add_argument("-s", "--setup", dest="setup", metavar="FILE", help="Setup input file")
    parser.add_argument("-p", "--params", dest="params", metavar="FILE", help="Params input file")
    parser.add_argument("--params-crash", dest="params_crash", action="store_true",
                        help="Add a crash param")
    parser.add_argument("-t", "--target", dest="target", metavar="WORKER_NAME",
                        help="Target worker name")
    parser.add_argument('--myself', dest='myself', action="store_true",
                        help="Use this hostname as target worker")
    args = parser.parse_args()

    if args.myself:
        args.target = socket.gethostname()

    if args.blueprint or args.setup:  # if only one is passed, we will crash and this is perfect
        _send_message(args, exchanger)
    else:
        _process_messages(args, config, exchanger)


_cli()
