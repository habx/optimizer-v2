#!/usr/bin/env python3
import argparse
import json
import logging
import os
import uuid
import traceback
import sys
import socket
from typing import Optional

import libpath
import boto3

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
            self.requests_queue_name += '-'+env_ns

        self.results_topic_name = '{env}-{topic}'.format(
            env=env,
            topic=topic_results_base,
        )


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
                             visibility_timeout: int = 3600*10
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

    def __init__(self, exchanger: Exchanger, myself: str = None):
        self.exchanger = exchanger
        self.executor = Executor()
        self.myself = myself

    def start(self):
        """Start the message processor"""
        self.exchanger.start()

    def run(self):
        """Make it run. Once called it never stops."""
        while True:
            msg = self.exchanger.get_request()
            if not msg:
                continue
            try:
                result_ok = self._process_message(msg)
                if result_ok:
                    self.exchanger.send_result(result_ok)
                self.exchanger.acknowledge_msg(msg)
            except Exception:
                logging.exception("Problem handing message: %s", msg.content)
                error_result = {
                    'type': 'optimizer-processing-result',
                    'data': {
                        'requestId': msg.content.get('requestId'),
                        'status': 'error',
                        'error': traceback.format_exception(*sys.exc_info()),
                    },
                }
                self.exchanger.send_result(error_result)

    def _process_message(self, msg: Message) -> Optional[dict]:
        """Actual message processing (without any error handling on purpose)"""
        logging.info("Processing message: %s", msg.content)
        request_id = msg.content['requestId']
        # Parsing the message
        data = msg.content['data']

        # These are mandatory parameters and as such have to exist
        lot = data['lot']
        setup = data['setup']
        params = data['params']
        context = data['context']

        # Only process message to ourselves if requested
        target_worker = params.get('target_worker')
        if target_worker and target_worker != self.myself:
            logging.info(
                "Message is not for me: target=\"%s\", myself=\"%s\"",
                target_worker,
                self.myself
            )
            return None

        # We can get an explicit crash request
        if params.get('crash', False):
            raise Exception('You asked me to crash !')

        # Processing it
        executor_result = self.executor.run(lot, setup, params)
        result = {
            'type': 'optimizer-processing-result',
            'requestId': request_id,
            'data': {
                'status': 'ok',
                'solutions': executor_result.solutions,
                'times': executor_result.elapsed_times,
                'version': Executor.VERSION,
                'lot': lot,
                'setup': setup,
                'params': params,
                'context': context,
            },
        }
        return result


def _process_messages(args: argparse.Namespace, exchanger: Exchanger):
    """Core processing message method"""
    logging.info("Optimizer V2 Worker (%s)", Executor.VERSION)

    myself = None
    if args.myself:
        myself = socket.gethostname()

    processing = MessageProcessor(exchanger, myself)
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

    if args.myself:
        params['target_worker'] = socket.gethostname()

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
    import socket
    if not os.path.exists(os.path.join(os.getenv('HOME'), '.aws/credentials'))\
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

    config = Config()
    exchanger = Exchanger(config)

    parser = argparse.ArgumentParser(description="Optimizer V2 Worker v"+Executor.VERSION)
    parser.add_argument("-l", dest="lot", metavar="FILE", help="Lot input file")
    parser.add_argument("-s", dest="setup", metavar="FILE", help="Setup input file")
    parser.add_argument("-p", dest="params", metavar="FILE", help="Params input file")
    parser.add_argument("--params-crash", dest="params_crash", action='store_true', help='Add a crash param')
    parser.add_argument('--myself', dest='myself', action='store_true', help='Only deal with myself')
    args = parser.parse_args()

    if args.lot or args.setup:  # if only one is passed, we will crash and this is perfect
        _send_message(args, exchanger)
    else:
        _process_messages(args, exchanger)


_cli()
