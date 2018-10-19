# coding=utf-8
"""Test of DecoratorCache"""

from libs.utils.decorator_timer import DecoratorTimer


def test_timer():
    """
    Test the timer
    :return:
    """
    @DecoratorTimer()
    def my_function():
        """
        Function to test
        :return:
        """
        return 100000000.0 ** 0.5

    for i in range(1000):
        my_function()
