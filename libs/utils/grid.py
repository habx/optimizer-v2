# coding=utf-8
"""
Grid module
"""

class Grid:
    """
    Creates a grid inside a plan.
    1. We select an edge according to the selectors,
    2. We apply the slicers
    3. if new edges are created we recursively apply the grid
    4. Once, we're finished we apply the reducers
    """
    def __init__(self, name, selectors, slicers, reducers):
        self.name = name
        self.selectors = selectors or []
        self.slicers = slicers or []
        self.reducers = reducers or []

    def run(self, plan):
        """
        Returns the modified plan with the created grid
        :param plan:
        :return: a plan with an inside grid
        """
        pass

