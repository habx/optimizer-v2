# coding=utf-8
"""
Simple decorator to time a function
"""
import time
import functools
import logging


class DecoratorTimer(object):
    """
    Timer decorator to easily record the time taken by your function !
    Optional keywords parameters :
    • off : Boolean, if True turns off the timer (useful to turn off timer in production
    • memorize : Boolean, if True stores each time the function is called
    """

    def __init__(self, memorize=True, off=False):
        self.off = off
        self.memorize = memorize
        self.timers = []

    def memorize_time(self, duration):
        """

        :param duration:
        :return:
        """
        self.timers.append(duration)
        return self

    @property
    def average_time(self):
        """
        Gets the average execution time of the function
        :return:
        """
        return sum(self.timers)/float(len(self.timers))

    @property
    def number_of_calls(self):
        """
        Number of time the function was called
        :return:
        """
        return len(self.timers)

    def __call__(self, func):

        @functools.wraps(func)
        def decorated(*args, **kwargs):
            """
            Decorator enabling the timing of a function execution time
            :param args:
            :param kwargs:
            :return:
            """
            # if the timer is turned off we just return the function results
            if self.off:
                return func(*args, **kwargs)

            # retrieve the function name for printing purpose
            func_name = func.__name__

            start_time = time.time()

            # compute the output of the function
            output = func(*args, **kwargs)

            end_time = time.time()
            duration = start_time - end_time

            logging.info("TIMER : Function {name} executed" +
                         " in {time} seconds".format(name=func_name, time=duration))

            if self.memorize:
                self.memorize_time(duration)

            if self.number_of_calls > 0:
                logging.info("TIMER : Average executation time" +
                             " of {time} seconds over {x} calls".format(time=self.average_time,
                                                                        x=self.number_of_calls))
            return output

        return decorated
