"""MQ protocol handling"""
import socket

import libs.optimizer as opt
import libs.executor.defs as defs
from libs.executor.executor import Executor


class MQProto:
    """MQ protocol handling"""
    @staticmethod
    def format_full_response(resp: opt.Response, td: defs.TaskDefinition, status: str):
        # OPT-74: The fields coming from the request are always added to the result
        # If we don't have a data sub-structure, we create one
        data = resp.to_json(status) if resp else {'status': 'unknown'}

        data['version'] = Executor.VERSION

        # OPT-116: Transmitting the hostname so that we can at least properly diagnose from
        #          which host the duplicate tasks are coming.
        data['hostname'] = socket.gethostname()

        # OPT-99: All the feedback shall only be done from the source data except for the
        #         context which is allowed to be modified by the processing.
        data['lot'] = td.blueprint
        data['setup'] = td.setup
        data['params'] = td.params
        data['context'] = td.context

        result = {
            'type': 'optimizer-processing-result',
            'data': data,
        }

        if td.task_id:
            result['taskId'] = td.task_id

        return result
