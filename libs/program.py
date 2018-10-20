# coding=utf-8
"""
Program module : describes what the user wants in its plan
"""


class Program:
    """
    Describes the program of the plan
    """
    def __init__(self, items):
        self.items = items


class ProgramItem:
    """
    One item of the program. The fundamental brick of the program.
    """
    def __init__(self, category, size, constraints):
        self.category = category
        self.size = size
        self.constraints = constraints
