"""MQ protocol handling"""
import socket
from typing import TYPE_CHECKING, Dict, Any, List
import libs.version as v

if TYPE_CHECKING:
    import libs.optimizer as opt
    import libs.executor.defs as defs


class MQProto:
    """MQ protocol handling"""
    @staticmethod
    def format_response_success(resp: 'opt.Response', td: 'defs.TaskDefinition', status: str) -> Dict[str, Any]:
        """
        Format a message for a successful response
        :param resp: Optimizer's response
        :param td: Task definition
        :param status: Current status of the task
        :return: MQ message
        """
        # OPT-74: The fields coming from the request are always added to the result
        # If we don't have a data sub-structure, we create one
        data = resp.to_json(status) if resp else {'status': 'unknown'}

        result = {
            'type': 'optimizer-processing-result',
            'data': data,
        }

        return MQProto._add_full_context(result, td)

    @staticmethod
    def format_response_error(ex: Exception, diag: List, times: Dict[str, float], td: 'defs.TaskDefinition') -> Dict[str, Any]:
        """
        Format a message for failed processing
        :param ex: Exception caught
        :param diag: Detailed diagnostic
        :param times: Processing time
        :param td: Task definition
        :return: MQ message
        """
        result = {
            'type': 'optimizer-processing-result',
            'data': {
                'status': 'timeout' if isinstance(ex, TimeoutError) else 'error',
                'error': diag,
                'times': times,
            },
        }

        return MQProto._add_full_context(result, td)

    @classmethod
    def format_request_solution_processing(cls, solution_id, td: 'defs.TaskDefinition'):
        request = {
            'type': 'optimizer-processing-solution',
            'data': {
                'solutionId': solution_id,
                'setup': td.setup,
            },
        }
        return MQProto._add_core_context(request)

    @staticmethod
    def _add_core_context(msg: Dict[str, Any], td: 'defs.TaskDefinition'):
        if td.task_id:
            msg['taskId'] = td.task_id

        data = msg.get('data', None)
        if not data:
            data = {}
            msg['data'] = data

        data['version'] = v.VERSION

        # OPT-116: Transmitting the hostname so that we can at least properly diagnose from
        #          which host the duplicate tasks are coming.
        data['hostname'] = socket.gethostname()

    @staticmethod
    def _add_full_context(msg: Dict[str, Any], td: 'defs.TaskDefinition') -> Dict[str, Any]:
        MQProto._add_core_context(msg, td)

        data = msg['data']

        # OPT-99: All the feedback shall only be done from the source data except for the
        #         context which is allowed to be modified by the processing.
        data['lot'] = td.blueprint
        data['setup'] = td.setup
        data['params'] = td.params
        data['context'] = td.context

        return msg
