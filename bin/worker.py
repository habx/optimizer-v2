#!/usr/bin/env python3
import argparse
import json
import logging
import logging.handlers
import os
import glob
import socket
import sys
import traceback
import uuid
import tempfile
import time
import sentry_sdk
import copy
from typing import Optional, List

import boto3

import libpath

from libs.utils.executor import Executor
from libs.worker.config import Config
from libs.worker.core import TaskDefinition
from libs.worker.dev import local_dev_hack
from libs.worker.mq import Exchanger, Message

# Initializing sentry at the earliest stage to detect any issue that might happen later
sentry_sdk.init("https://55bd31f3c51841e5b2233de2a02a9004@sentry.io/1438222", {
    'environment': os.getenv('HABX_ENV', 'local'),
    'release': Executor.VERSION,
})


class MessageProcessor:
    """Message processing class"""

    def __init__(self, config: Config, exchanger: Exchanger, my_name: str = None):
        self.exchanger = exchanger
        self.config = config
        self.executor = Executor()
        self.my_name = my_name
        self.output_dir = None
        self.log_handler = None
        self.s3_client = boto3.client('s3')

    def start(self):
        """Start the message processor"""
        self.output_dir = tempfile.mkdtemp('worker-optimizer')

    def run(self):
        """Make it run. Once called it never stops."""
        while True:
            msg = self.exchanger.get_request()
            if not msg:  # No message received (queue is empty)
                continue

            # OPT-99: We shall NOT modify the source data
            td = TaskDefinition.from_json(msg.content.get('data'))

            result = self._process_task(td)

            result['requestId'] = msg.content.get('requestId')

            # End of processing code
            self._process_message_after(msg)

            # Always acknowledging messages
            self.exchanger.acknowledge_msg(msg)

    def _process_task(self, td: TaskDefinition) -> dict:
        self._process_message_before()

        logging.info("Processing %s", td)

        # We calculate the overall time just in case we face a crash
        before_time = time.time()

        try:
            result = self._process_task_core(td)
        except Exception as e:
            logging.exception("Problem handing message")
            result = {
                'type': 'optimizer-processing-result',
                'data': {
                    'status': 'error',
                    'error': traceback.format_exception(*sys.exc_info()),
                    'times': {
                        'totalReal': (time.time() - before_time)
                    },
                },
            }

            # Timeout is a special kind of error, we still want the stacktrace like any other
            # error.
            if isinstance(e, TimeoutError):
                result['data']['status'] = 'timeout'

        if result:
            # OPT-74: The fields coming from the request are always added to the result

            # If we don't have a data sub-structure, we create one
            data = result.get('data')
            if not data:
                data = {'status': 'unknown'}
                result['data'] = data
            data['version'] = Executor.VERSION

            # OPT-99: All the feedback shall only be done from the source data except for the
            #         context which is allowed to be modified by the processing.
            data['lot'] = td.blueprint
            data['setup'] = td.setup
            data['params'] = td.params
            data['context'] = td.context
            self.exchanger.send_result(result)

            if data.get('status') != 'ok':
                # If we had an issue, we save the output
                for k in ['lot', 'setup', 'params', 'context', 'solutions', 'version']:
                    if k in data:
                        with open(os.path.join(self.output_dir, '%s.json' % k), 'w') as f:
                            json.dump(data[k], f)

        return result

    def _save_output_files(self, request_id: str):
        files = self._output_files()

        if files:
            logging.info("Uploading some files on S3...")

        for src_file in files:
            # OPT-89: Storing files in a "tasks" directory
            dst_file = "tasks/{request_id}/{file}".format(
                request_id=request_id,
                file=src_file[len(self.output_dir)+1:]
            )
            logging.info(
                "Uploading \"%s\" to s3://%s/%s",
                src_file,
                self.config.s3_repository,
                dst_file
            )
            self.s3_client.upload_file(
                src_file,
                self.config.s3_repository,
                dst_file,
                ExtraArgs={
                    'ACL': 'public-read'
                }
            )

        if files:
            logging.info("Upload done...")

    def _output_files(self) -> List[str]:
        return glob.glob(os.path.join(self.output_dir, '*'))

    def _cleanup_output_dir(self):
        for f in self._output_files():
            logging.info("Deleting file \"%s\"", f)
            os.remove(f)

    def _process_message_before(self):
        self._cleanup_output_dir()

    def _process_message_after(self, msg: Message):
        self._save_output_files(msg.content.get('requestId'))

    def _process_task_core(self, td: TaskDefinition) -> Optional[dict]:
        """
        Actual message processing (without any error handling on purpose)
        :param msg: Message to process
        :return: Message to return
        """
        logging.info("Processing message: %s", td)

        # If we're having a personal identify, we only accept message to ourself
        target_worker = td.params.get('target_worker')
        if (self.my_name is not None and target_worker != self.my_name) or (
                self.my_name is None and target_worker):
            logging.info(
                "   ... message is not for me: target=\"%s\", myself=\"%s\"",
                target_worker,
                self.my_name,
            )
            return None

        td.local_params = {
            'output_dir': self.output_dir,
        }

        # Processing it
        executor_result = self.executor.run(td.blueprint, td.setup, td.params, td.local_params)
        result = {
            'type': 'optimizer-processing-result',
            'data': {
                'status': 'ok',
                'solutions': executor_result.solutions,
                'times': executor_result.elapsed_times,
            },
        }

        return result


def _process_messages(args: argparse.Namespace, config: Config, exchanger: Exchanger):
    """Core processing message method"""
    logging.info("Optimizer V2 Worker (%s)", Executor.VERSION)

    processing = MessageProcessor(config, exchanger, args.target)
    processing.start()
    processing.run()


def _send_message(args: argparse.Namespace, exchanger: Exchanger):
    """Core sending message function"""
    # Reading the input files
    with open(args.lot) as lot_fp:
        lot = json.load(lot_fp)
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
        'requestId': str(uuid.uuid4()),
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s | %(lineno)-5d | %(levelname).4s | %(message)s",
    )

    local_dev_hack()

    # We're using AWS_REGION at habx and boto3 expects AWS_DEFAULT_REGION
    if 'AWS_DEFAULT_REGION' not in os.environ and 'AWS_REGION' in os.environ:
        os.environ['AWS_DEFAULT_REGION'] = os.environ['AWS_REGION']

    config = Config()
    exchanger = Exchanger(config)

    parser = argparse.ArgumentParser(description="Optimizer V2 Worker v" + Executor.VERSION)
    parser.add_argument("-l", "--lot", dest="lot", metavar="FILE", help="Lot input file")
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

    if args.lot or args.setup:  # if only one is passed, we will crash and this is perfect
        exchanger.prepare(consumer=False)
        _send_message(args, exchanger)
    else:
        exchanger.prepare()
        _process_messages(args, config, exchanger)


_cli()
