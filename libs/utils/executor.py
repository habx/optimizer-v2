import logging

import signal

import libs.optimizer as opt
from typing import List

"""
The (new) Executor allows to chain execution wrappers directly in the execution stack so that each
ExecWrapper can, if he wants to, apply treatments before and after the processing.
"""


class ExecWrapper:
    """Base class that defines how an ExecWrapper works."""
    def __init__(self):
        self.next: ExecWrapper = None

    def run(self, lot: dict, setup: dict, params: dict = None) -> opt.Response:
        self._before()
        try:
            return self._exec(lot, setup, params)
        finally:
            self._after()

    def _before(self):
        pass

    def _after(self):
        pass

    def _exec(self, lot: dict, setup: dict, params_dict: dict = None) -> opt.Response:
        return self.next.run(lot, setup, params_dict) if self.next else None

    @staticmethod
    def instantiate(params: dict):
        return None


class OptimizerRun(ExecWrapper):
    OPTIMIZER = opt.Optimizer()

    def _exec(self, lot: dict, setup: dict, params_dict: dict = None) -> opt.Response:
        return self.OPTIMIZER.run(lot, setup, params_dict)

    @staticmethod
    def instantiate(params: dict):
        if params.get("skip_optimizer", False):
            return None
        return OptimizerRun()


class Sample(ExecWrapper):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def _before(self):
        logging.info("[BEFORE OPTIMIZER] params: %s", self.params)

    def _after(self):
        logging.info("[AFTER OPTIMIZER]")

    @staticmethod
    def instantiate(params: dict):
        if params.get("skip_optimizer", False):
            return None
        return Sample(params)


class Crasher(ExecWrapper):
    @staticmethod
    def instantiate(params: dict):
        if params.get("crash", False):
            return Crasher()
        return None

    def _exec(self, lot: dict, setup: dict, params_dict: dict = None):
        raise Exception("Crashing !")


class Timeout(ExecWrapper):
    class TimeoutException(Exception):
        pass

    def throw_timeout(self, signum, frame):
        raise self.TimeoutException()

    def __init__(self, timeout: int):
        super().__init__()
        self.timeout = timeout

    def _before(self):
        signal.signal(signal.SIGALRM, self.throw_timeout)
        signal.alarm(self.timeout)

    def _after(self):
        signal.alarm(0)
        pass

    def _exec(self, lot: dict, setup: dict, params: dict = None):
        # try:
        super()._exec(lot, setup, params)
        # Intercepting the TimeoutException makes things more confusing:
        # except self.TimeoutException:
        #    return None

    @staticmethod
    def instantiate(params: dict):
        timeout = int(params.get('timeout', '0'))
        return Timeout(timeout) if timeout > 0 else None


class Executor:
    VERSION = opt.OPTIMIZER_VERSION

    EXEC_BUILDERS: List[ExecWrapper] = [OptimizerRun, Crasher, Sample, Timeout]

    def __init__(self):
        pass

    def run(self, lot: dict, setup: dict, params_dict: dict = None) -> opt.Response:
        first_exec = self.create_exec_wrappers(params_dict)
        return first_exec.run(lot, setup, params_dict)

    def create_exec_wrappers(self, params: dict = None) -> ExecWrapper:
        """
        Prepare the chain of ExecWrappers
        :param params: Params to use
        :return: Chain of ExecWrappers
        """

        # We create instances of ExecWrappers
        execs: List[ExecWrapper] = []
        for eb in self.EXEC_BUILDERS:
            e = eb.instantiate(params)
            if e:
                execs.append(e)

        # We chain them
        previous = None
        for e in execs:
            e.next = previous
            previous = e

        return execs[len(execs)-1]
