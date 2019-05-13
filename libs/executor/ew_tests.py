from libs.executor.defs import ExecWrapper, TaskDefinition
import libs.optimizer as opt


class ExecTest(ExecWrapper):
    """Only useful for tests to mock things"""

    def __init__(self, response: opt.Response = None):
        super().__init__()

        if not response:
            response = opt.Response
            response.solutions = []
            response.elapsed_times = {'total': 2}

        self.response = response

    def _exec(self, td: TaskDefinition) -> opt.Response:
        return self.response

