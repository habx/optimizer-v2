import cProfile
import logging
import os
import signal  # for Timeout
import tracemalloc
from typing import List

import libs.optimizer as opt

"""
The (new) Executor allows to chain execution wrappers directly in the execution stack so that each
ExecWrapper can, if he wants to, apply treatments before and after the processing.
"""


class ExecWrapper:
    """Base class that defines how an ExecWrapper works."""

    def __init__(self):
        self.next: ExecWrapper = None

    def run(self, lot: dict, setup: dict, params: dict = None, local_params: dict = None) \
            -> opt.Response:
        self._before()
        try:
            return self._exec(lot, setup, params, local_params)
        finally:
            self._after()

    def _before(self):
        pass

    def _after(self):
        pass

    def _exec(self, lot: dict, setup: dict, params: dict = None, local_params: dict = None) \
            -> opt.Response:
        return self.next.run(lot, setup, params, local_params) if self.next else None

    @staticmethod
    def instantiate(params: dict, local_params: dict):
        return None


class OptimizerRun(ExecWrapper):
    """
    Running optimizer unless "skipOptimizer" is specified
    """
    OPTIMIZER = opt.Optimizer()

    def _exec(self, lot: dict, setup: dict, params: dict = None, local_params: dict = None) \
            -> opt.Response:
        return self.OPTIMIZER.run(lot, setup, params, local_params)

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        if params.get("skipOptimizer", False):
            return None
        return OptimizerRun()


class Crasher(ExecWrapper):
    """
    Crashing the execution if a "crash" parameter is specified
    """

    def _exec(self, lot: dict, setup: dict, params: dict = None, local_params: dict = None):
        raise Exception("Crashing !")

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        if params.get("crash", False):
            return Crasher()
        return None


class Timeout(ExecWrapper):
    """
    Enabling a timeout timer if a "timeout" is specified
    """

    def throw_timeout(self, signum, frame):
        raise TimeoutError("Processing timeout reached")

    def __init__(self, timeout: int):
        super().__init__()
        self.timeout = timeout

    def _before(self):
        signal.signal(signal.SIGALRM, self.throw_timeout)
        signal.alarm(self.timeout)

    def _after(self):
        signal.alarm(0)

    def _exec(self, lot: dict, setup: dict, params: dict = None, local_params: dict = None):
        super()._exec(lot, setup, params, local_params)

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        timeout = int(params.get('timeout', '0'))
        return Timeout(timeout) if timeout > 0 else None


class CpuProfile(ExecWrapper):
    """
    Enabling CPU profiling if the "cpuProfile" parameter is specified
    """

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def _before(self):
        self.cpu_prof = cProfile.Profile()
        self.cpu_prof.enable()

    def _after(self):
        self.cpu_prof.disable()
        self.cpu_prof.dump_stats(os.path.join(self.output_dir, "cpu_profile.prof"))
        self.cpu_prof = None

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        if params.get('cpuProfile', False):
            return CpuProfile(local_params['outputDir'])
        return None


class TraceMalloc(ExecWrapper):
    """
    Enabling malloc monitoring if "traceMalloc" is specified
    """

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def _before(self):
        tracemalloc.start()

    def _after(self):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        with open(os.path.join(self.output_dir, 'mem_stats.txt'), 'w') as f:
            for stat in top_stats[:40]:
                f.write("%s\n" % stat)

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        if params.get('tracemalloc', False):
            return TraceMalloc(local_params['outputDir'])
        return None


class LoggingToFile(ExecWrapper):
    """
    Saving logs to files
    """

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.log_handler: logging.FileHandler = None

    def _before(self):
        logger = logging.getLogger('')
        log_file = os.path.join(self.output_dir, 'output.log')
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

    def _after(self):
        if self.log_handler:
            self.log_handler.close()
            logging.getLogger('').removeHandler(self.log_handler)
            self.log_handler = None

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        if not params.get('skipFileLogging', False):
            return LoggingToFile(local_params['outputDir'])
        return None


class LoggingLevel(ExecWrapper):
    """
    Changing the logging level when a "loggingLevel" is specified
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

    def _before(self):
        logger = logging.getLogger('')
        self.previous_level = logger.level
        logger.setLevel(self.logging_level)

    def _after(self):
        logging.getLogger('').setLevel(self.previous_level)

    @staticmethod
    def instantiate(params: dict, local_params: dict):
        if params.get('loggingLevel'):
            return LoggingLevel(LoggingLevel.LOGGING_LEVEL_CONV[params['loggingLevel']])
        return None


class Executor:
    VERSION = opt.OPTIMIZER_VERSION

    EXEC_BUILDERS: List[ExecWrapper] = [OptimizerRun, Crasher, Timeout, CpuProfile, TraceMalloc,
                                        LoggingToFile, LoggingLevel, ]

    def run(self, lot: dict, setup: dict, params: dict = None,
            local_params: dict = None) -> opt.Response:
        if local_params is None:
            local_params = {}
        first_exec = self.create_exec_wrappers(params, local_params)
        return first_exec.run(lot, setup, params, local_params)

    def create_exec_wrappers(self, params: dict = None, local_params: dict = None) -> ExecWrapper:
        """
        Prepare the chain of ExecWrappers
        :param params: Params to use
        :param local_params Local parameters to use
        :return: Chain of ExecWrappers
        """

        # We create instances of ExecWrappers
        execs: List[ExecWrapper] = []
        for eb in self.EXEC_BUILDERS:
            e = eb.instantiate(params, local_params)
            if e:
                execs.append(e)

        # We chain them
        previous = None
        for e in execs:
            e.next = previous
            previous = e

        return execs[len(execs) - 1]
