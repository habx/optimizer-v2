from typing import List

import libs.optimizer as opt
from libs.executor.defs import ExecWrapper, TaskDefinition
from libs.executor.ew_optimizer import OptimizerRun, Crasher, Timeout
from libs.executor.ew_instruments import CProfile, PProfile, PyInstrument, TraceMalloc, MultiRuns
from libs.executor.ew_logs import LoggingToFile, LoggingLevel
from libs.executor.ew_upload import S3Upload, SaveFilesOnError

"""
The (new) Executor allows to chain execution wrappers directly in the execution stack so that each
ExecWrapper can, if he wants to, apply treatments before and after the processing.
"""


class Executor:
    VERSION = opt.OPTIMIZER_VERSION

    EXEC_BUILDERS: List[ExecWrapper] = [
        # Core execution:
        OptimizerRun, Crasher, Timeout,

        # Instrumentation:
        CProfile, PProfile, PyInstrument, TraceMalloc, MultiRuns,

        # Logging management:
        LoggingToFile, LoggingLevel,

        # Result upload (mostly for workers and jobs):
        SaveFilesOnError, S3Upload,
    ]

    def run(self, td: TaskDefinition) -> opt.Response:
        first_exec = self.create_exec_wrappers(td)
        return first_exec.run(td)

    def create_exec_wrappers(self, td: TaskDefinition) -> ExecWrapper:
        """
        Prepare the chain of ExecWrappers
        :param td: Task definition
        :return: First element of the chain of ExecWrappers
        """

        # We create instances of ExecWrappers
        execs: List[ExecWrapper] = []
        for eb in self.EXEC_BUILDERS:
            e = eb.instantiate(td)
            if e:
                execs.append(e)

        # We chain them
        previous = None
        for e in execs:
            e.next = previous
            previous = e

        return execs[len(execs) - 1]
