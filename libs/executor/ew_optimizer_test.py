import os
import time
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


class TestMultiprocessCleaner(unittest.TestCase):
    @staticmethod
    def get_nb_children():
        return len(psutil.Process(os.getpid()).children(recursive=True))

    def test_multiprocess_ok(self):
        # nb_process_before = self.get_nb_children()
        timeout = Timeout(2)
        timeout.next = Faker(params={'nb_processes': 10, 'process_time': 1})
        out = timeout.run(TaskDefinition())
        self.assertIsNotNone(out)
        self.assertEqual(0, self.get_nb_children())

    def test_multiprocess_timeout_process_leak(self):
        time.sleep(2)
        nb_process_before = self.get_nb_children()
        timeout = Timeout(1, cleanup_sub_processes=False)
        timeout.next = Faker(params={'nb_processes': 10, 'process_time': 2})
        with self.assertRaises(TimeoutError):
            timeout.run(TaskDefinition())
        self.assertGreater(self.get_nb_children(), nb_process_before)

    # Unfortunately, the core test passes fine with pytest but then makes it hangs forever. I couldn't find anything
    # on it. Debug doesn't show anything: https://gist.github.com/fclairamb/52a04bb3f31b66a20381e3eed900054b
    @unittest.skip
    def test_multiprocess_timeout_ok(self):
        time.sleep(2)
        nb_process_before = self.get_nb_children()
        timeout = Timeout(1)
        timeout.next = Faker(params={'nb_processes': 10, 'process_time': 2})
        with self.assertRaises(TimeoutError):
            timeout.run(TaskDefinition())
        self.assertEqual(nb_process_before, self.get_nb_children())

