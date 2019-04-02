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
import cProfile
import tracemalloc
from typing import Optional, List

import boto3

import libpath

from libs.utils.executor import Executor


class Config:
    """Config management"""
    TOPIC_REQUESTS_BASE = 'optimizer-processing-requests'
    TOPIC_RESULTS_BASE = 'optimizer-processing-results'

    def __init__(
            self,
            topic_requests_base=TOPIC_REQUESTS_BASE,
            topic_results_base=TOPIC_RESULTS_BASE
    ):
        name = os.getenv('WORKER_NAME', 'worker-optimizer')
        env = os.getenv('HABX_ENV', 'dev')
        env_ns = os.getenv('HABX_ENV_NS')
        env_low_priority = os.getenv('LOW_PRIORITY', 'false') == 'true'

        if env_low_priority:
            topic_requests_base += '-lowpriority'

        self.requests_topic_name = '{env}-{topic}'.format(
            env=env,
            topic=topic_requests_base,
        )
        self.requests_queue_name = '{env}-{name}-{topic}'.format(
            env=env,
            name=name,
            topic=topic_requests_base,
        )

        if env_ns:
            self.requests_queue_name += '-' + env_ns

        self.results_topic_name = '{env}-{topic}'.format(
            env=env,
            topic=topic_results_base,
        )

        self.s3_repository = 'habx-{env}-worker-optimizer-v2'.format(env=env)


class Message:
    """Message wrapper"""

    def __init__(self, content, handle):
        self.content = content
        self.handle = handle


class Exchanger:
    """Message exchange management"""

    def __init__(self, config: Config):
        self._sqs_client = boto3.client('sqs')
        self._sns_client = boto3.client('sns')
        self.config: Config = config
        self._consuming_queue_url: str = None
        self._publishing_topic_arn: str = None

    def _get_or_create_topic(self, topic_name: str) -> str:
        """Create a topic and return its ARN"""
        logging.info("Getting topic \"%s\" ...", topic_name)
        topic = self._sns_client.create_topic(
            Name=topic_name,
        )
        return topic.get('TopicArn')

    def _get_or_create_queue(self,
                             queue_name: str,
                             topic_name: str = None,
                             visibility_timeout: int = 3600 * 10
                             ) -> str:
        """Create a queue and return its URL"""
        logging.info("Getting queue \"%s\" ...", queue_name)
        try:
            queue = self._sqs_client.create_queue(
                QueueName=queue_name,
            )
            queue_url = queue.get('QueueUrl')
            queue_attrs = self._sqs_client.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=['QueueArn', 'VisibilityTimeout'],
            )
            queue_arn = queue_attrs['Attributes']['QueueArn']

            # Allowing to chang the visibility timeout
            if int(queue_attrs['Attributes']['VisibilityTimeout']) != visibility_timeout:
                logging.warning(
                    "Changing the visibility timeout of %s to %d",
                    queue_arn,
                    visibility_timeout
                )
                self._sqs_client.set_queue_attributes(
                    QueueUrl=queue_url,
                    Attributes={
                        'VisibilityTimeout': str(visibility_timeout),
                    },
                )

            # Allowing to register to a topic
            if topic_name:
                topic_arn = self._get_or_create_topic(topic_name)
                self._sns_client.subscribe(
                    TopicArn=topic_arn,
                    Protocol='sqs',
                    Endpoint=queue_arn
                )
                policy = """
{
  "Version":"2012-10-17",
  "Id": "%s",
  "Statement":[
    {
      "Effect":"Allow",
      "Principal" : {"AWS" : "*"},
      "Action":"SQS:SendMessage",
      "Resource": "%s",
      "Condition":{
        "ArnEquals":{
          "aws:SourceArn": "%s"
        }
      }
    }
  ]
}""" % (str(uuid.uuid4()), queue_arn, topic_arn)
                self._sqs_client.set_queue_attributes(
                    QueueUrl=queue_url,
                    Attributes={
                        'Policy': policy
                    }
                )
            return queue.get('QueueUrl')
        except Exception:
            logging.exception("Couldn't create queue: %s", queue_name)
            raise

    def start(self) -> None:
        """Start the worker by performing some setup operations"""

        # Creating the consuming (requests) queue and registering it to a topic
        self._consuming_queue_url = self._get_or_create_queue(
            self.config.requests_queue_name,
            self.config.requests_topic_name
        )

        # Creating the publishing (results) topic
        self._publishing_topic_arn = self._get_or_create_topic(
            self.config.results_topic_name
        )

    def get_request(self):
        """Fetch a request as a message"""
        logging.info("Waiting for a message to process...")
        sqs_response = self._sqs_client.receive_message(
            QueueUrl=self._consuming_queue_url,
            AttributeNames=['All'],
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        messages = sqs_response.get('Messages')
        if not messages:
            logging.info("   ...no result")
            return
        # Fetching the first (and sole) SQS message
        sqs_message = messages[0]

        # Getting the SNS content (what a message)
        sns_payload = json.loads(sqs_message['Body'])

        # Getting the actual content
        habx_message = json.loads(sns_payload['Message'])

        msg = Message(habx_message, sqs_message.get('ReceiptHandle'))
        logging.info("   ...got one")
        return msg

    def acknowledge_msg(self, msg: Message):
        """Acknowledge a message to make it disappear from the processing queue"""
        self._sqs_client.delete_message(
            QueueUrl=self._consuming_queue_url,
            ReceiptHandle=msg.handle,
        )

    def send_result(self, result: dict):
        """Send a result"""
        j = json.dumps(result)
        logging.info("Sending the processing result: %s", j)
        self._sns_client.publish(
            TopicArn=self._publishing_topic_arn,
            Message=j,
        )

    def send_request(self, request):
        """This is a helper method. It should only be used for testing"""
        j = json.dumps(request)
        requests_topic_arn = self._get_or_create_topic(self.config.requests_topic_name)
        logging.warning(
            "[TEST ONLY] Sending a processing request: topic_arn = %s ; content = %s",
            requests_topic_arn,
            j,
        )
        self._sns_client.publish(
            TopicArn=requests_topic_arn,
            Message=json.dumps(request),
        )


class MessageProcessor:
    """Message processing"""

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
        self.exchanger.start()
        self.output_dir = tempfile.mkdtemp('worker-optimizer')

    def run(self):
        """Make it run. Once called it never stops."""
        while True:
            msg = self.exchanger.get_request()
            if not msg:  # No message received (queue is empty)
                continue

            self._process_message_before(msg)

            before_time = time.time()

            try:
                result = self._process_message(msg)
            except Exception:
                logging.exception("Problem handing message: %s", msg.content)
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

            self._process_message_after(msg)

            if result:
                # OPT-74: The fields coming from the request are always added to the result
                result['requestId'] = msg.content.get('requestId')

                # If we don't have a data sub-structure, we create one
                data = result.get('data')
                if not data:
                    data = {'status': 'unknown'}
                    result['data'] = data
                data['version'] = Executor.VERSION
                src_data = msg.content.get('data')
                data['lot'] = src_data.get('lot')
                data['setup'] = src_data.get('setup')
                data['params'] = src_data.get('params')
                data['context'] = src_data.get('context')
                self.exchanger.send_result(result)

            # Always acknowledging messages
            self.exchanger.acknowledge_msg(msg)

    def _save_output_dir(self, request_id: str):
        files = self._output_files()

        if files:
            logging.info("Uploading some files on S3...")

        for src_file in files:
            dst_file = "{request_id}/{file}".format(
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

    def _logging_to_file_before(self):
        logger = logging.getLogger("")
        # Removing the previous handler
        if self.log_handler:
            self.log_handler.close()
            logger.removeHandler(self.log_handler)
            self.log_handler = None

    def _logging_to_file_after(self):
        logger = logging.getLogger("")
        log_file = os.path.join(self.output_dir, 'output.log')
        logging.info("Writing logs to %s", log_file)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)-15s | %(filename)15.15s:%(lineno)-4d | %(levelname).4s | %(message)s"
        )
        handler.setFormatter(formatter)

        # Adding the new handler
        self.log_handler = handler
        logger.addHandler(handler)

    def _process_message_before(self, msg: Message):
        self._cleanup_output_dir()
        self._logging_to_file_before()

        params = msg.content.get('data', {}).get('params')

        # CPU Profiling start
        if params.get('cpu_profile'):
            self.cpu_prof = cProfile.Profile()
            self.cpu_prof.enable()

        # Memory analysis start
        if params.get('tracemalloc'):
            tracemalloc.start()

    def _process_message_after(self, msg: Message):
        params = msg.content.get('data', {}).get('params')

        # CPU profiling stop
        if params.get('cpu_profile'):
            self.cpu_prof.disable()
            self.cpu_prof.dump_stats(os.path.join(self.output_dir, "profile.prof"))
            self.cpu_prof = None

        if params.get('tracemalloc'):
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            with open(os.path.join(self.output_dir, 'mem_stats.txt'), 'w') as f:
                for stat in top_stats[:40]:
                    f.write("%s\n" % stat)

        self._logging_to_file_after()

        self._save_output_dir(msg.content.get('requestId'))

    def _process_message(self, msg: Message) -> Optional[dict]:
        """
        Actual message processing (without any error handling on purpose)
        :param msg: Message to process
        :return: Message to return
        """
        logging.info("Processing message: %s", msg.content)

        # Parsing the message
        data = msg.content['data']

        # These are mandatory parameters and as such have to exist
        lot = data['lot']
        setup = data['setup']
        params = data['params']

        self._process_message_before(msg)

        # If we're having a personal identify, we only accept message to ourself
        target_worker = params.get('target_worker')
        if (self.my_name is not None and target_worker != self.my_name) or (
                self.my_name is None and target_worker):
            logging.info(
                "   ... message is not for me: target=\"%s\", myself=\"%s\"",
                target_worker,
                self.my_name,
            )
            return None

        # We can get an explicit crash request
        if params.get('crash', False):
            raise Exception('You asked me to crash !')

        # Processing it
        executor_result = self.executor.run(lot, setup, params)
        result = {
            'type': 'optimizer-processing-result',
            'data': {
                'status': 'ok',
                'solutions': executor_result.solutions,
                'times': executor_result.elapsed_times,
            },
        }

        self._process_message_after(msg)

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


def _local_dev_hack():
    if not os.path.exists(os.path.expanduser('~/.aws/credentials')) \
            and not os.getenv('AWS_ACCESS_KEY_ID') \
            and not os.getenv('HABX_ENV'):
        logging.warning(
            "[LOCAL DEV ONLY] Injecting some AWS credentials in env vars [/LOCAL DEV ONLY]"
        )
        os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAJLYSRVQ2B5NDHC3A'
        os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
        os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'
        os.environ['HABX_ENV_NS'] = socket.gethostname()


def _cli():
    """CLI orchestrating function"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s | %(lineno)-5d | %(levelname).4s | %(message)s",
    )

    _local_dev_hack()

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
        _send_message(args, exchanger)
    else:
        _process_messages(args, config, exchanger)


_cli()
