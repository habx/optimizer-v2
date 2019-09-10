import json
import logging
import uuid

import boto3

from libs.worker.config import Config


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

    def prepare(self, consumer=True, producer=True) -> None:
        """Start the worker by performing some setup operations"""

        # Creating the consuming (requests) queue and registering it to a topic
        if consumer:
            self._consuming_queue_url = self._get_or_create_queue(
                self.config.requests_queue_name,
                self.config.requests_topic_name
            )

        # Creating the publishing (results) topic
        if producer:
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
        logging.info(
            "Sending the processing result",
            extra={
                'component': 'mq',
                'action': 'mq.send_result',
                'mqMsg': j,
            }
        )
        self._sns_client.publish(
            TopicArn=self._publishing_topic_arn,
            Message=j,
        )

    def send_request(self, request: dict):
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
