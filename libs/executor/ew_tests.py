from libs.executor.defs import ExecWrapper, TaskDefinition
import libs.optimizer as opt
from typing import Optional, Dict, Any
import multiprocessing
import time


class Faker(ExecWrapper):
    """Only useful for tests to mock things"""

    def __init__(self, response: Optional[opt.Response] = None, params: Dict[str, Any] = None):
        super().__init__()
        self.response: Optional[opt.Response] = response

        if not params:
            params = {}
        self.process_time: float = params.get('process_time', 2)
        self.nb_processes: int = params.get('nb_processes', 1)

    def _core_process(self, process_nb: int = 0) -> Optional[opt.Response]:
        time.sleep(self.process_time)
        return None

    def _exec(self, td: TaskDefinition) -> opt.Response:
        if self.nb_processes > 1:
            pool = multiprocessing.Pool(self.nb_processes)
            results = pool.map(self._core_process, range(1, self.nb_processes))
            self.response = results[0]

            # There's no handling of failure (like it's currently done in the refiner)
            pool.close()
            pool.join()
        else:
            self.response = self._core_process()

        if not self.response:
            self.response = opt.Response(solutions=[], elapsed_times={'total': 2})

        return self.response

    @staticmethod
    def instantiate(td: TaskDefinition) -> Optional['ExecWrapper']:
        p = td.params.get('fake')
        return Faker(p) if p is not None else None


class Crasher(ExecWrapper):
    """
    Crashing the execution if a "crash" parameter is specified
    """

    def _exec(self, td: TaskDefinition):
        raise Exception("Crashing !")

    @staticmethod
    def instantiate(td: TaskDefinition) -> Optional[ExecWrapper]:
        if td.params.get("crash", False):
            return Crasher()
        return None
