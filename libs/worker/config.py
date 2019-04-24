import os


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

        # OPT-92: The env_ns shall only be used if it's not the same as the env
        if env_ns and env_ns != env:
            self.requests_queue_name += '-' + env_ns

        self.results_topic_name = '{env}-{topic}'.format(
            env=env,
            topic=topic_results_base,
        )

        self.s3_repository = 'habx-{env}-optimizer-v2'.format(env=env)
