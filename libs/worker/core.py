import glob
import logging
import os
import socket
import sys
import tempfile
import time
import traceback
from typing import List, Optional

from libs.executor.executor import Executor, TaskDefinition
from libs.worker.config import Config
from libs.worker.mqproto import MQProto


class TaskProcessor:
    """Message processing class"""

    def __init__(self, config: Config, my_name: str = None):
        # self.exchanger = exchanger
        self.config = config
        self.executor = Executor()
        self.my_name = my_name
        self.output_dir = None
        self.log_handler = None

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
        before_time_real = time.time()
        before_time_cpu = time.process_time()

        try:
            result = self._process_task_core(td)
        except Exception as e:
            result = {
                'type': 'optimizer-processing-result',
                'data': {
                    'status': 'error',
                    'error': traceback.format_exception(*sys.exc_info()),
                    'times': {
                        'totalReal': (time.time() - before_time_real),
                        'total': (time.process_time() - before_time_cpu)
                    },
                },
            }

            # Timeout is a special kind of error, we still want the stacktrace like any other
            # error.
            if isinstance(e, TimeoutError):
                result['data']['status'] = 'timeout'
            else:
                logging.exception("Problem handing message")

        self._process_task_after()

        return result

    def _output_files(self) -> List[str]:
        return glob.glob(os.path.join(self.output_dir, '*'))

    def _cleanup_output_dir(self):
        for f in self._output_files():
            logging.info("Deleting file \"%s\"", f)
            os.remove(f)

    def _process_task_before(self):
        self._cleanup_output_dir()

    def _process_task_after(self):
        pass

    def _process_task_core(self, td: TaskDefinition) -> Optional[dict]:
        """
        Actual message processing (without any error handling on purpose)
        :param td: Task definition we're processing
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

        td.local_context.output_dir = self.output_dir

        # Processing it
        response = self.executor.run(td)
        return MQProto.format_full_response(response, td, 'ok')
