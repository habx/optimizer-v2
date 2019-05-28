import os
import json

DEFAULT_CONF_PATH = 'config/config.default.json'

def from_file():
    env = os.environ.get('HABX_ENV')

    if env is None:
        raise Exception('Logger - HABX_ENV is missing.')

    # Try to load default conf
    default_conf = {}
    if os.path.isfile(DEFAULT_CONF_PATH):
        with open(DEFAULT_CONF_PATH, 'r') as f:
            default_conf = json.load(f)

    # Try to load env conf
    env_conf = {}
    env_conf_path = f'config/config.{env.lower()}.json'
    if os.path.isfile(env_conf_path):
        with open(env_conf_path, 'r') as f:
            env_conf = json.load(f)

    # merge both
    return {**default_conf, **env_conf}