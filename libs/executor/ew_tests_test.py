import unittest


def test_exec_wrapper_crash():
    def exec_wrapper_crash():
        crasher = Crasher()
        crasher.run(TaskDefinition())

    unittest.expectedFailure(exec_wrapper_crash)
