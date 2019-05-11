import signal

from libs.executor.defs import ExecWrapper, TaskDefinition
import libs.optimizer as opt
import libs.io.plot as plt


class OptimizerRun(ExecWrapper):
    """
    Running optimizer unless "skipOptimizer" is specified
    """
    OPTIMIZER = opt.Optimizer()

    def _exec(self, td: TaskDefinition) -> opt.Response:
        output_path = td.local_context.output_dir
        if output_path:
            plt.output_path = output_path
        return self.OPTIMIZER.run(td.blueprint, td.setup, td.params, td.local_context)

    @staticmethod
    def instantiate(td: TaskDefinition):
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
    def instantiate(td: TaskDefinition):
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

    def _before(self):
        signal.signal(signal.SIGALRM, self.throw_timeout)
        signal.alarm(self.timeout)

    def _after(self):
        signal.alarm(0)

    def _exec(self, td: TaskDefinition):
        super()._exec(td)

    @staticmethod
    def instantiate(td: TaskDefinition):
        timeout = int(td.params.get('timeout', '0'))
        return __class__(timeout) if timeout > 0 else None
