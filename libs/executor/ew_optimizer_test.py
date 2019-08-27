import os
import unittest

import psutil

from libs.executor.defs import TaskDefinition
from libs.executor.ew_optimizer import Timeout
from libs.executor.ew_tests import Faker


class TestTimeout(unittest.TestCase):
    def test_exec_wrapper_timeout_ok(self):
        timeout = Timeout(2)
        timeout.next = Faker(params={'process_time': 1})
        out = timeout.run(TaskDefinition())
        assert out is not None

    def test_exec_wrapper_timeout_fail(self):
        timeout = Timeout(2)
        timeout.next = Faker(params={'process_time': 3})
        with self.assertRaises(TimeoutError):
            timeout.run(TaskDefinition())


class TestMultiprocssCleaner(unittest.TestCase):
    @staticmethod
    def get_nb_children():
        return len(psutil.Process(os.getpid()).children(recursive=True))

    def test_multiprocess_ok(self):
        timeout = Timeout(2)
        timeout.next = Faker(params={'nb_processes': 10, 'process_time': 1})
        out = timeout.run(TaskDefinition())
        self.assertIsNotNone(out)
        self.assertEqual(self.get_nb_children(), 0)

    def test_multiprocess_timeout_process_leak(self):
        timeout = Timeout(1, cleanup_sub_processes=False)
        timeout.next = Faker(params={'nb_processes': 10, 'process_time': 2})
        with self.assertRaises(TimeoutError):
            timeout.run(TaskDefinition())
        self.assertGreater(self.get_nb_children(), 0)

    def test_multiprocess_timeout_ok(self):
        timeout = Timeout(1)
        timeout.next = Faker(params={'nb_processes': 10, 'process_time': 2})
        with self.assertRaises(TimeoutError):
            timeout.run(TaskDefinition())
        self.assertEqual(0, self.get_nb_children())

