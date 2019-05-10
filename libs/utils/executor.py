import cProfile  # for CProfile
import copy
import logging  # for LoggingLevel and LoggingToFile
import os
import pstats  # for CProfile
import signal  # for Timeout
import tracemalloc  # for TraceMalloc
from typing import List, Optional

import pprofile  # for PProfile
import pyinstrument

import libs.io.plot as plt
import libs.optimizer as opt

"""
The (new) Executor allows to chain execution wrappers directly in the execution stack so that each
ExecWrapper can, if he wants to, apply treatments before and after the processing.
"""


class TaskDefinition:
    """Definition of the task we're about to process"""

    def __init__(self):
        # All these parameters are fetched from the API
        self.blueprint: dict = None
        self.setup: dict = None
        self.params: dict = None
        self.context: dict = {}
        self.local_params: dict = {}
        self.task_id: str = None

    def copy_for_processing(self) -> 'TaskDefinition':
        """
        Create a copy of the parameters to avoid instance modification in the optimizer code.

        Please note the context and local_params are left uncopied on purpose.

        :return: New instance duplicated from the first one.
        """
        new = TaskDefinition()
        new.blueprint = copy.deepcopy(self.blueprint)
        new.setup = copy.deepcopy(self.setup)
        new.params = copy.deepcopy(self.params)
        new.local_params = self.local_params
        new.context = self.context
        return new

    def check(self):
        """Check the input is correct"""
        if not self.blueprint:
            raise ValueError('blueprint is invalid')
        if not self.setup:
            raise ValueError('setup is invalid')
        if self.params is None:
            raise ValueError('params is invalid')

    def __str__(self):
        return \
            "Blueprint: {blueprint}, Setup: {setup}, Params: {params}, " \
            "LocalParams: {local_params}, Context: {context}".format(
                blueprint=self.blueprint,
                setup=self.setup,
                params=self.params,
                local_params=self.local_params,
                context=self.context,
            )

    @staticmethod
    def from_json(data: dict) -> 'TaskDefinition':
        """
        Create a task from a given JSON input
        :param data: JSON input
        :return: A TaskDefinition
        """
        td = TaskDefinition()

        # Preferring blueprint to lot (as it describes more precisely what we are actually
        # processing.
        td.blueprint = data.get('blueprint')
        if not td.blueprint:
            td.blueprint = data.get('lot')
        td.setup = data.get('setup')
        td.params = data.get('params')
        td.context = data.get('context')
        td.check()
        return td


class ExecWrapper:
    """Base class that defines how an ExecWrapper works."""

    def __init__(self):
        self.next: Optional[ExecWrapper] = None

    def run(self, td: TaskDefinition) -> opt.Response:
        """
        Execution method
        :param td: Task definition
        :return: The optimizer response
        """
        self._before()
        try:
            return self._exec(td)
        finally:
            self._after()

    def _before(self):
        pass

    def _after(self):
        pass

    def _exec(self, td: TaskDefinition) -> opt.Response:
        return self.next.run(td) if self.next else None

    @staticmethod
    def instantiate(params: dict, local_params: dict):
        return None


class OptimizerRun(ExecWrapper):
    """
    Running optimizer unless "skipOptimizer" is specified
    """
    OPTIMIZER = opt.Optimizer()

    def _exec(self, td: TaskDefinition) -> opt.Response:
        output_path = td.local_params.get('output_dir')
        if output_path:
            plt.output_path = output_path
        return self.OPTIMIZER.run(td.blueprint, td.setup, td.params, td.local_params)

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        if params.get('skip_optimizer', False):
            return None
        return OptimizerRun()


class Crasher(ExecWrapper):
    """
    Crashing the execution if a "crash" parameter is specified
    """

    def _exec(self, td: TaskDefinition):
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

    def _exec(self, td: TaskDefinition):
        super()._exec(td)

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        timeout = int(params.get('timeout', '0'))
        return __class__(timeout) if timeout > 0 else None


class PProfile(ExecWrapper):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def _exec(self, td: TaskDefinition):
        prof = pprofile.Profile()
        with prof:
            res = super()._exec(td)
        prof.dump_stats(os.path.join(self.output_dir, 'pprofile_stats.out'))
        return res

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        if params.get('pprofile', False):
            return __class__(local_params['output_dir'])
        return None


class PyInstrument(ExecWrapper):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.profiler = pyinstrument.Profiler()

    def _before(self):
        self.profiler.start()

    def _after(self):
        self.profiler.stop()
        with open(os.path.join(self.output_dir, 'pyinstrument.html'), 'w') as fp:
            fp.write(self.profiler.output_html())
        with open(os.path.join(self.output_dir, 'pyinstrument.txt'), 'w') as fp:
            fp.write(self.profiler.output_text())

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        if params.get('pyinstrument', False):
            return __class__(local_params['output_dir'])
        return None


class CProfile(ExecWrapper):
    """
    Enabling CPU profiling if the "c_profile" parameter is specified
    """

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def _before(self):
        self.cpu_prof = cProfile.Profile()
        self.cpu_prof.enable()

    def _after(self):
        self.cpu_prof.disable()
        self.cpu_prof.dump_stats(os.path.join(self.output_dir, "cProfile.prof"))
        with open(os.path.join(self.output_dir, 'cProfile.txt'), 'w') as fp:
            stats = pstats.Stats(self.cpu_prof, stream=fp)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats()
        self.cpu_prof = None

    @staticmethod
    def instantiate(params: dict, local_params: dict = None):
        if params.get('c_profile', False):
            return __class__(local_params['output_dir'])
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
            return __class__(local_params['output_dir'])
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
        if not params.get('skip_file_logging', False):
            return __class__(local_params['output_dir'])
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

    def _before(self):
        logger = logging.getLogger('')
        self.previous_level = logger.level
        logger.setLevel(self.logging_level)

    def _after(self):
        logging.getLogger('').setLevel(self.previous_level)

    @staticmethod
    def instantiate(params: dict, local_params: dict):
        if params.get('logging_level'):
            return __class__(LoggingLevel.LOGGING_LEVEL_CONV[params['logging_level']])
        return None


class Executor:
    VERSION = opt.OPTIMIZER_VERSION

    EXEC_BUILDERS: List[ExecWrapper] = [OptimizerRun, Crasher, Timeout, CProfile, PProfile,
                                        PyInstrument, TraceMalloc, LoggingToFile, LoggingLevel, ]

    def run(self, td: TaskDefinition) -> opt.Response:
        first_exec = self.create_exec_wrappers(td.params, td.local_params)
        return first_exec.run(td)

    def create_exec_wrappers(self, params: dict = None, local_params: dict = None) -> ExecWrapper:
        """
        Prepare the chain of ExecWrappers
        :param params: Params to use
        :param local_params Local parameters to use
        :return: First element of the chain of ExecWrappers
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
