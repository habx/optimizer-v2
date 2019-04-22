import copy

from libs.utils.executor import Executor


class TaskDefinition:
    """Definition of the task we're about to process"""
    def __init__(self):
        # All these parameters are fetched from the API
        self.blueprint: dict = None
        self.setup: dict = None
        self.params: dict = None
        self.context: dict = {}
        self.local_params: dict = {}

    def copy_for_processing(self) -> 'TaskDefinition':
        new = TaskDefinition()
        new.blueprint = copy.deepcopy(self.blueprint)
        new.setup = copy.deepcopy(self.setup)
        new.params = copy.deepcopy(self.params)
        new.local_params = self.local_params
        new.context = self.context
        return new

    def check(self):
        assert self.blueprint is not None
        assert self.setup is not None
        assert self.params is not None

    def __str__(self):
        return "Blueprint: {blueprint}, Setup: {setup}, Params: {params}, " \
               "LocalParams: {local_params}, Context: {context}".format(
                    blueprint=self.blueprint,
                    setup=self.setup,
                    params=self.params,
                    local_params=self.local_params,
                    context=self.context,
                )

    @staticmethod
    def from_json(data: dict) -> 'TaskDefinition':
        td = TaskDefinition()
        td.blueprint = data.get('lot')  # We should have named it blueprint
        td.setup = data.get('setup')
        td.params = data.get('params')
        td.context = data.get('context')
        td.check()
        return td


class Processor:
    def __init__(self, executor: Executor):
        self.executor = executor

    def process_message(self, td: TaskDefinition):
        td = td.copy_for_processing()

        executor_result = self.executor.run(td.blueprint, td.setup, td.params, td.local_params)
        result = {
            'type': 'optimizer-processing-result',
            'data': {
                'status': 'ok',
                'solutions': executor_result.solutions,
                'times': executor_result.elapsed_times,
            },
        }

        return result
