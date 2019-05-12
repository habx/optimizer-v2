import copy
from typing import Optional

import libs.optimizer as opt


class TaskDefinition:
    """Definition of the task we're about to process"""

    def __init__(self):
        self.blueprint: dict = None  # Blueprint to be processed (immutable)
        self.setup: dict = None  # Setup to be processed (immutable)
        self.params: dict = None  # Parameters controlling the processing behavior (immutable)
        self.context: dict = {}  # Why / Who / When / Where it was started
        self.task_id: str = None  # Task ID used for storage
        self.local_context: opt.LocalContext = opt.LocalContext()  # Local execution context

    def copy_for_processing(self) -> 'TaskDefinition':
        """
        Create a copy of the parameters to avoid instance modification in the optimizer code.

        Please note the context and local_params are left as-is on purpose.

        :return: New instance duplicated from the first one.
        """
        new = TaskDefinition()
        new.blueprint = copy.deepcopy(self.blueprint)
        new.setup = copy.deepcopy(self.setup)
        new.params = copy.deepcopy(self.params)
        new.local_context = self.local_context
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
            "LocalContext: {local_context}, Context: {context}".format(
                blueprint=self.blueprint,
                setup=self.setup,
                params=self.params,
                local_context=self.local_context,
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
        Execution method. Should not be overridden.
        :param td: Task definition
        :return: The optimizer response
        """
        self._before(td)

        resp: opt.Response = None
        try:
            resp = self._exec(td)
            return resp
        finally:
            self._after(td, resp)

    def _before(self, td: TaskDefinition):
        """
        Called before the execution
        """
        pass

    def _after(self, td: TaskDefinition, resp: opt.Response):
        """
        Called after the execution
        """
        pass

    def _exec(self, td: TaskDefinition) -> opt.Response:
        """Actual execution"""
        return self.next.run(td) if self.next else None

    @staticmethod
    def instantiate(td: TaskDefinition) -> Optional['ExecWrapper']:
        """
        Instantiate (or not) the ExecWrapper that will later be chained to other ExecWrappers and
        then executed.
        """
        return None
