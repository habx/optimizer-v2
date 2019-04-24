

def local_dev_hack():
    import os
    if not os.path.exists(os.path.expanduser('~/.aws/credentials')) \
            and not os.getenv('AWS_ACCESS_KEY_ID') \
            and not os.getenv('HABX_ENV'):
        import logging
        import socket
        import re
        logging.warning(
            "[LOCAL DEV ONLY] Injecting some AWS credentials in env vars [/LOCAL DEV ONLY]"
        )
        os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA2RESSX3JNVGOL4S5'
        os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
        os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'
        if 'HABX_ENV_NS' not in os.environ:
            ns = socket.gethostname()
            ns = re.sub('[^A-Za-z0-9]+', '', ns)
            os.environ['HABX_ENV_NS'] = ns[:20]

