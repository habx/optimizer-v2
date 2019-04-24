import logging
import os
import socket


def local_dev_hack():
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
