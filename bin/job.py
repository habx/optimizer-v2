#!/usr/bin/env python3
import argparse
import logging
import os
import copy

import uuid
import requests

# OPT-119 & OPT-120: Dirty path handling
import libpath

from libs.executor.executor import Executor
from libs.worker.config import Config
from libs.worker.core import TaskDefinition, TaskProcessor
from libs.worker.mq import Exchanger

import config as cf
import habx_logger

logging.root = habx_logger.HabxLogger(cf.from_file())
logging.getLogger().setLevel(logging.INFO)

# OPT-120: Only to make sure libpath won't be removed
libpath.add_local_libs()


def fetch_task_definition(context: dict) -> TaskDefinition:
    endpoints = {
        'local': 'http://localhost:3000/job',
        'dev': 'https://www.habx-dev.fr/api/optimizer-v2/job',
        'staging': 'https://www.habx-staging.fr/api/optimizer-v2/job',
        'prod': 'https://www.habx.fr/api/optimizer-v2/job',
    }

    endpoint = endpoints.get(os.getenv('HABX_ENV', 'local'))

    response = requests.get(endpoint, params=context, headers={
        'x-habx-token': os.getenv('HABX_TOKEN', 'ymSC4QkHwxEnAeyBu9UqWzbs')
    })

    if not response.ok:
        log_context = copy.deepcopy(context)
        log_context['httpStatusCode'] = response.status_code
        log_context['httpContent'] = response.content.decode("utf-8")
        logging.warning(
            'Invalid response: %d / %s', response.status_code, response.content.decode("utf-8"),
            extra=log_context,
        )

    job_input = response.json().get('job')

    td = TaskDefinition.from_json(job_input)
    return td


def process_task(config: Config, td: TaskDefinition):
    processor = TaskProcessor(config)
    processor.prepare()

    exchanger = Exchanger(config)
    exchanger.prepare(consumer=False, producer=True)

    td.local_context.prepare_mq(exchanger, td)
    result = processor.process_task(td)
    exchanger.send_result(result)


def _cli():
    example_text = """
Example usage:
==============

BLUEPRINT_ID=1000 SETUP_ID=2000 bin/job.py
"""

    parser = argparse.ArgumentParser(
        description="Optimizer V2 Job v" + Executor.VERSION,
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-r", "--result-id", dest="result_id", default=os.getenv('RESULT_ID'),
        metavar="ID", help="Result ID",

    )

    # OPT-106: Allowing to specify a taskId
    parser.add_argument(
        "-t", "--task-id", dest="task_id",  default=os.getenv('TASK_ID'),
        metavar="ID", help="Task ID",
    )
    args = parser.parse_args()

    job_fetching_params = {
        'resultId': args.result_id,
    }

    td = fetch_task_definition(job_fetching_params)

    if args.task_id is not None:
        td.task_id = args.task_id

    # If no taskId is specified, we should specify one
    if not td.task_id:
        td.task_id = str(uuid.uuid4())

    config = Config()

    process_task(config, td)


_cli()
