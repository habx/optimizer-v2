import logging
import os
import signal

import psutil

from libs.executor.defs import ExecWrapper, TaskDefinition
import libs.optimizer as opt
import libs.io.plot as plt
from typing import List


class OptimizerRun(ExecWrapper):
    """
    Running optimizer unless "skipOptimizer" is specified
    """
    OPTIMIZER = opt.Optimizer()

    def _exec(self, td: TaskDefinition) -> opt.Response:
        # APP-4810: Creating a duplicate instance before processing
        td = td.copy_for_processing()

        output_path = td.local_context.output_dir
        if output_path:
            plt.output_path = output_path
        return self.OPTIMIZER.run(td.blueprint, td.setup, td.params, td.local_context)

    @staticmethod
    def instantiate(td: TaskDefinition):
        if td.params.get('skip_optimizer', False):
            return None
        return OptimizerRun()


class Timeout(ExecWrapper):
    """
    Enabling a timeout timer if a "timeout" is specified
    """

    def throw_timeout(self, signum, frame):
        raise TimeoutError("Processing timeout reached")

    def __init__(self, timeout: int, cleanup_sub_processes=True):
        super().__init__()
        self.timeout = timeout
        self.cleanup_sub_processes = cleanup_sub_processes
        self.safe_processes: List[int] = []

    def _before(self, td: TaskDefinition):
        # self.identify_safe_processes()
        signal.signal(signal.SIGALRM, self.throw_timeout)
        signal.alarm(self.timeout)

    def _after(self, td: TaskDefinition, resp: opt.Response):
        signal.alarm(0)
        if self.cleanup_sub_processes:
            self.kill_sub_processes()

    # We don't seem to actually need it
    def identify_safe_processes(self):
        sp = []
        for p in psutil.Process(os.getpid()).children(recursive=True):
            sp.append(p.pid)
        self.safe_processes = sp

    def kill_sub_processes(self):
        children = psutil.Process(os.getpid()).children(recursive=True)

        for child in children:
            if child.pid in self.safe_processes:
                continue
            logging.warning(
                "Killing sub-process",
                extra={
                    'action': 'ew_timeout.sub_kill',
                    'component': 'ew_timeout',
                    'pid': child.pid,
                }
            )
            child.terminate()
            try:
                child.wait(5)
            except psutil.TimeoutExpired:
                try:
                    child.kill()
                    child.wait(1)
                except psutil.NoSuchProcess:
                    pass

    @staticmethod
    def instantiate(td: TaskDefinition):
        # OPT-4791: Adding a 1h timeout by default
        timeout = int(td.params.get('timeout', '3600'))
        return __class__(timeout) if timeout > 0 else None
