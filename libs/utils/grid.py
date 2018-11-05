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

    def apply_to(self, plan):
        """
        Returns the modified plan with the created grid
        :param plan:
        :return: a copy of the plan with an inside grid
        """
        pass


class Selector:
    """
    Returns an iterator on a given mesh face
    """
    def __init__(self, name, predicate):
        self.name = name
        self.predicate = predicate

    def apply_to(self, face):
        """
        Runs the selector
        :param face:
        :return:
        """
        for edge in face.edges:
            if self.predicate(edge):
                yield edge


# examples

def _non_ortho_angle(edge):
    return not edge.previous_is_ortho


non_ortho_selector = Selector('non_ortho', _non_ortho_angle)


if __name__ == '__main__':

    def create_a_grid():
        """
        Test
        :return:
        """
        pass

    create_a_grid()
