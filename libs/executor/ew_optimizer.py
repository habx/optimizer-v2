import signal
from typing import Optional

from libs.executor.defs import ExecWrapper, TaskDefinition
import libs.optimizer as opt
import libs.io.plot as plt


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
    def instantiate(td: TaskDefinition) -> Optional['ExecWrapper']:
        if td.params.get('skip_optimizer', False):
            return None
        return OptimizerRun()


class Crasher(ExecWrapper):
    """
    Crashing the execution if a "crash" parameter is specified
    """

    def _exec(self, td: TaskDefinition):
        raise Exception("Crashing !")

    @staticmethod
    def instantiate(td: TaskDefinition) -> Optional['ExecWrapper']:
        if td.params.get("crash", False):
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

    def _before(self, td: TaskDefinition):
        signal.signal(signal.SIGALRM, self.throw_timeout)
        signal.alarm(self.timeout)

    def _after(self, td: TaskDefinition, resp: opt.Response):
        signal.alarm(0)

    @staticmethod
    def instantiate(td: TaskDefinition) -> Optional['ExecWrapper']:
        # OPT-4791: Adding a 1h timeout by default
        timeout = int(td.params.get('timeout', '3600'))
        return __class__(timeout) if timeout > 0 else None


class DelayedMQ(ExecWrapper):
    """
    Allows to send MQ messages after the processing has been done
    """

    def _after(self, td: TaskDefinition, resp: opt.Response):
        for msg in td.local_context.mq_requests_msg_list:
            td.local_context.mq.send_request(msg)
        td.local_context.mq_requests_msg_list.clear()

    @staticmethod
    def instantiate(td: TaskDefinition) -> Optional['ExecWrapper']:
        return __class__()
