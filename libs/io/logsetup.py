# coding=utf-8
"""Logging setup helper"""
import logging.handlers
import os
from output import DEFAULT_LOGS_OUTPUT_FOLDER


def init(reset=False, override_default=True, log_file_name='optimizer.log'):
    """Logging setup"""
    log_path = os.path.join(DEFAULT_LOGS_OUTPUT_FOLDER, log_file_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("default")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.handlers.RotatingFileHandler(log_path,
                                                        maxBytes=100 * 1024,
                                                        backupCount=20)

    log_format = "%(asctime)-15s | %(filename).8s:%(lineno)-5d | %(levelname).4s | %(message)s"
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    if reset:
        file_handler.doRollover()
    logger.addHandler(file_handler)

    client_handler = logging.StreamHandler()
    client_handler.setLevel(logging.INFO)
    client_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(client_handler)

    if override_default:
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[file_handler, client_handler],
        )
        logging.info("Started !")

    return logger
