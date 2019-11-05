import json
from typing import Dict
import requests
import os
import time
import logging


class FeatureFlag:
    """Feature flag internal structure"""
    def __init__(self, value: bool, expiration: float):
        self.value = value
        self.expiration = expiration


class Features:
    """
    Feature flags management class

    See https://www.habx.fr/ftr/ for more details
    """
    FLAGS: Dict[str, FeatureFlag] = {}
    ENV = os.getenv('HABX_ENV', 'dev')

    @classmethod
    def _get(cls, name) -> bool:
        ff = cls.FLAGS.get(name)
        if not ff or ff.expiration < time.time():
            logging.info(
                "Fetching a feature flag",
                extra={
                    'component': 'features',
                    'action': 'features.fetching',
                    'featureFlag': name,
                }
            )
            ff = cls._fetch(name)
            cls.FLAGS[name] = ff
        return ff.value

    @classmethod
    def _fetch(cls, name) -> FeatureFlag:
        r = requests.get(f'https://www.habx.fr/api/features/flag/{cls.ENV}/{name}')
        if r.status_code != 200:
            logging.warning(
                'Features: Request issue: %s', r.reason,
                extra={
                    'component': 'features',
                    'action': 'features.request_issue',
                    'featureFlag': name,
                    'statusCode': r.status_code,
                    'reason': r.reason,
                    'content': r.content,
                }
            )
            return FeatureFlag(False, time.time()+300)

        j = json.loads(r.content)
        j = j.get(name)

        if not j:
            return FeatureFlag(False, time.time()+300)

        ff = FeatureFlag(j.get('value', False), time.time()+j.get('ttl', 300))

        logging.info(
            "Feature flag fetched !",
            extra={
                'component': 'features',
                'action': 'features.fetched',
                'featureFlag': name,
                'value': ff.value,
                'ttl': ff.expiration,
            }
        )

        return ff

    @classmethod
    def do_door(cls) -> bool:
        return cls._get('optimizer-v2.do_door')

    @classmethod
    def disable_error_reporting(cls) -> bool:
        return cls._get('optimizer-v2.disable_error_reporting')

