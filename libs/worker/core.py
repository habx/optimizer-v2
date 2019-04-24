import copy
import glob
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from typing import List, Optional

import boto3

from libs.utils.executor import Executor
from libs.worker.config import Config


class TaskDefinition:
    """Definition of the task we're about to process"""

    def __init__(self):
        # All these parameters are fetched from the API
        self.blueprint: dict = None
        self.setup: dict = None
        self.params: dict = None
        self.context: dict = {}
        self.local_params: dict = {}

    def copy_for_processing(self) -> 'TaskDefinition':
        new = TaskDefinition()
        new.blueprint = copy.deepcopy(self.blueprint)
        new.setup = copy.deepcopy(self.setup)
        new.params = copy.deepcopy(self.params)
        new.local_params = self.local_params
        new.context = self.context
        return new

    def check(self):
        assert self.blueprint is not None
        assert self.setup is not None
        assert self.params is not None

    def __str__(self):
        return "Blueprint: {blueprint}, Setup: {setup}, Params: {params}, " \
               "LocalParams: {local_params}, Context: {context}".format(
            blueprint=self.blueprint,
            setup=self.setup,
            params=self.params,
            local_params=self.local_params,
            context=self.context,
        )

    @staticmethod
    def from_json(data: dict) -> 'TaskDefinition':
        td = TaskDefinition()

        # Preferring blueprint to lot (as it describes more precisely what we are actually
        # processing.
        td.blueprint = data.get('blueprint')
        if not td.blueprint:
            td.blueprint = data.get('lot')
        td.setup = data.get('setup')
        td.params = data.get('params')
        td.context = data.get('context')
        td.check()
        return td


class TaskProcessor:
    """Message processing class"""

    def __init__(self, config: Config, my_name: str = None):
        # self.exchanger = exchanger
        self.config = config
        self.executor = Executor()
        self.my_name = my_name
        self.output_dir = None
        self.log_handler = None
        self.s3_client = boto3.client('s3')

    def prepare(self):
        """Start the message processor"""
        self.output_dir = tempfile.mkdtemp('worker-optimizer')

    def process_task(self, td: TaskDefinition) -> dict:
        """This is the core processing method. It receives a TaskDefinition and produces a message
        for the SNS topic to answer. This is because different types of inputs can be used (SQS or
        K8S job) but only one kind of feedback can be sent."""
        logging.info("Processing %s", td)

        self._process_task_before()

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

            if data.get('status') != 'ok':
                # If we had an issue, we save the output
                for k in ['lot', 'setup', 'params', 'context', 'solutions', 'version']:
                    if k in data:
                        with open(os.path.join(self.output_dir, '%s.json' % k), 'w') as f:
                            json.dump(data[k], f)

        self._process_task_after(td)

        return result

    def _save_output_files(self, task_id: str):
        files = self._output_files()

        if files:
            logging.info("Uploading some files on S3...")

        for src_file in files:
            # OPT-89: Storing files in a "tasks" directory
            dst_file = "tasks/{task_id}/{file}".format(
                task_id=task_id,
                file=src_file[len(self.output_dir) + 1:]
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
                ExtraArgs={'ACL': 'public-read'}
            )

        if files:
            logging.info("Upload done...")

    def _output_files(self) -> List[str]:
        return glob.glob(os.path.join(self.output_dir, '*'))

    def _cleanup_output_dir(self):
        for f in self._output_files():
            logging.info("Deleting file \"%s\"", f)
            os.remove(f)

    def _process_task_before(self):
        self._cleanup_output_dir()

    def _process_task_after(self, td: TaskDefinition):
        request_id: str = td.context.get('taskId') if td.context else None
        if request_id:
            self._save_output_files(request_id)
        else:
            logging.warning("You didn't specify a context.taskId, no upload was performed. "
                            "Are you sure you want that ?")

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
