# coding=utf-8
"""
Exceptions module
"""


class Conflict(Exception):
    """
    Raise a conflict
    """
    pass


class Success(Exception):
    """
    Finds a solution
    """
    pass


class Finished(Exception):
    """
    The solver has finished
    """
    pass


class Restart(Exception):
    """
    The solver must be restarted
    """
    pass
