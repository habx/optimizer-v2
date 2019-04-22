import argparse

from libs.utils.executor import Executor


class ProcessingTask:
    def __init__(self):
        # All these parameters are fetched from the API
        self.blueprint: dict = None
        self.setup: dict = None
        self.params: dict = None
        self.context: dict = None


def _cli():
    parser = argparse.ArgumentParser(description="Optimizer V2 Job v" + Executor.VERSION)
    parser.add_argument(
        "-l", "--blueprint-id", dest="blueprint_id", metavar="ID", help="Blueprint ID"
    )
    parser.add_argument(
        "-s", "--setup-id", dest="setup_id", metavar="ID", help="Setup ID"
    )
    parser.add_argument(
        "-p", "--params-id", dest="params_id", metavar="ID", help="Params ID"
    )
    parser.add_argument(
        "-b", "--batch-execution-id", dest="batch_execution_id", metavar="ID",
        help="BatchExecution ID"
    )
    args = parser.parse_args()

    # Data fetching code
    # Note: The context shall be built directly by the "service-optimizer-results" so that it can
    #       be changed later easily.


_cli()
