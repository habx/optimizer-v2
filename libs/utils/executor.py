import logging

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
        self._before(params)
        try:
            return self._exec(lot, setup, params)
        finally:
            self._after()

    def _before(self, params: dict):
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
    def _before(self, params: dict):
        logging.info("[BEFORE OPTIMIZER] params: %s", params)

    def _after(self):
        logging.info("[AFTER OPTIMIZER]")

    @staticmethod
    def instantiate(params: dict):
        if params.get("skip_optimizer", False):
            return None
        return Sample()


class Crasher(ExecWrapper):
    @staticmethod
    def instantiate(params: dict):
        if params.get("crash", False):
            return Crasher()
        return None

    def _exec(self, lot: dict, setup: dict, params_dict: dict = None):
        raise Exception("Crashing !")


class Executor:
    VERSION = opt.OPTIMIZER_VERSION

    EXEC_BUILDERS: List[ExecWrapper] = [OptimizerRun, Sample, Crasher]

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
