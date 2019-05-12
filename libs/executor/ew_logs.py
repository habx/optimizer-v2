import logging
import os

from libs.executor.defs import ExecWrapper, TaskDefinition
import libs.optimizer as opt


class LoggingToFile(ExecWrapper):
    """
    Saving logs to files
    """

    FILENAME = 'output.log'

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.log_handler: logging.FileHandler = None

    def _before(self, td: TaskDefinition):
        logger = logging.getLogger('')
        log_file = os.path.join(self.output_dir, self.FILENAME)
        logging.info("Writing logs to %s", log_file)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)-15s | %(filename)15.15s:%(lineno)-4d | %(levelname).4s | %(message)s"
        )
        handler.setFormatter(formatter)
        handler.setLevel(logger.level)

        # Adding the new handler
        self.log_handler = handler
        logger.addHandler(handler)

    def _after(self, td: TaskDefinition, resp: opt.Response):
        if self.log_handler:
            self.log_handler.close()
            logging.getLogger('').removeHandler(self.log_handler)
            self.log_handler = None
            td.local_context.add_file(
                self.FILENAME,
                ftype='logging',
                title='Logs',
                mime='text/plain'
            )

    @staticmethod
    def instantiate(td: TaskDefinition):
        if not td.params.get('skip_file_logging', False):
            return __class__(td.local_context.output_dir)
        return None


class LoggingLevel(ExecWrapper):
    """
    Changing the logging level when a "logging_level" is specified
    """
    LOGGING_LEVEL_CONV = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(self, level: int):
        super().__init__()
        self.logging_level = level
        self.previous_level: int = 0

    def _before(self, td: TaskDefinition):
        logger = logging.getLogger('')
        self.previous_level = logger.level
        logger.setLevel(self.logging_level)

    def _after(self, td: TaskDefinition, resp: opt.Response):
        logging.getLogger('').setLevel(self.previous_level)

    @staticmethod
    def instantiate(td: TaskDefinition):
        if td.params.get('logging_level'):
            return __class__(LoggingLevel.LOGGING_LEVEL_CONV[td.params['logging_level']])
        return None
