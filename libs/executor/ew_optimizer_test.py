import unittest

from libs.executor.defs import TaskDefinition
from libs.executor.ew_optimizer import Timeout, Crasher
from libs.executor.ew_tests import ExecTest


def test_exec_wrapper_timeout():
    timeout = Timeout(2)
    timeout.next = ExecTest()
    out = timeout.run(TaskDefinition())
    assert out is not None


def test_exec_wrapper_crash():
    def exec_wrapper_crash():
        crasher = Crasher()
        crasher.run(TaskDefinition())

    unittest.expectedFailure(exec_wrapper_crash)
