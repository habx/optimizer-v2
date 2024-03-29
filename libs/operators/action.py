# coding=utf-8
"""
Operator module
"""

from typing import TYPE_CHECKING, Sequence, Optional, Any
import logging

if TYPE_CHECKING:
    from libs.plan.plan import Space
    from libs.mesh.mesh import Edge
    from libs.operators.selector import Selector
    from libs.operators.mutation import Mutation
    from libs.operators.constraint import Constraint


class Action:
    """
    Action Class
    An action is the combination of a selector and a mutation. The action will loop trough the edges
    of a face or a space yielded by the selector and try to mutate it according to the mutation.
    The mutation will be reversed if it breaks an imperative constraint of fails to improve
    the score of an objective constraint.
    The repeat flag specifies whether to keep applying the mutation after an edge has
    been successfully mutated.
    """

    __slots__ = 'name', 'selector', 'mutation', 'multiple_mutations', '_tried', 'number_of_pass'

    def __init__(self, selector: 'Selector', mutation: 'Mutation', multiple_mutations: bool = False,
                 name: str = '', number_of_pass: int = 1):
        self.name = name or '{0} + {1}'.format(selector.name, mutation.name)
        self.selector = selector
        self.mutation = mutation
        self.multiple_mutations = multiple_mutations
        # we store the couple edge and space that we have already tried TODO : improve this
        self._tried = set()
        self.number_of_pass = number_of_pass

    def __repr__(self):
        return 'Operator: {0}, repeat={1}'.format(self.name, self.multiple_mutations)

    def mark_as_tried(self, space, edge):
        """
        Caches the fact that the mutation has been tried on this space and edge
        :param space:
        :param edge:
        :return:
        """
        self._tried.add(self.try_id(space, edge))

    def check_tried(self, space, edge) -> bool:
        """
        Check if the action has already been tried on this space and edge
        :param space:
        :param edge:
        :return:
        """
        if self.try_id(space, edge) in self._tried:
            logging.debug('Action: Already tried this edge with this space')
            return True
        return False

    @staticmethod
    def try_id(space: 'Space', edge: 'Edge') -> str:
        """
        Computes an id for the tuple space, edge
        :param space:
        :param edge:
        :return:
        """
        return '{0}-{1}'.format(id(space), id(edge))

    def apply_to(self,
                 space: 'Space',
                 selector_optional_args: Sequence[Any],
                 constraints: Optional[Sequence['Constraint']] = None) -> Sequence['Space']:
        """
        Applies the operator
        :param space: the spaces that will be modified by the action
        :param selector_optional_args:
        :param constraints:
        :return:
        """
        logging.debug("Action: Applying the action %s to the space %s", self, space)

        # separate imperative constraints from objective constraints
        if constraints:
            imp_constraints = [cst for cst in constraints if cst.imperative]
            opt_constraints = [cst for cst in constraints if not cst.imperative]
        else:
            imp_constraints = []
            opt_constraints = []

        # for each edge of the space yielded by the selector apply the mutation
        all_modified_spaces = []

        for edge in self.selector.yield_from(space, *selector_optional_args):

            # for performance purpose we check if we have already tried this edge
            if self.check_tried(space, edge):
                continue

            # get the list of the spaces that will be modified
            spaces = self.mutation.spaces_modified(edge, space)

            # if no spaces will be modified continue to the next edge
            if not spaces:
                continue

            # We verify if the mutation increases or decreases the score
            initial_score = self.score(spaces, opt_constraints)

            # we apply the mutation
            modified_spaces = self.mutation.apply_to(edge, space)

            if modified_spaces:

                # check imperative constraints
                for constraint in imp_constraints:
                    if not constraint.check(*modified_spaces):
                        logging.debug('Action: Constraint breached: %s - %s',
                                      constraint.name, space)
                        # reverse the change
                        self.mutation.reverse(modified_spaces)
                        # add the edge and the space to the cache
                        self.mark_as_tried(space, edge)
                        modified_spaces = []
                        break

                # check objective constraints
                if modified_spaces and initial_score is not None:
                    new_score = self.score(modified_spaces, opt_constraints)
                    if new_score >= initial_score:
                        logging.debug("Action: poor global score: %s - %s",
                                      self, space)
                        # reverse the mutation
                        self.mutation.reverse(modified_spaces)
                        # add the edge and the space ot the cache
                        self.mark_as_tried(space, edge)
                        modified_spaces = []

            all_modified_spaces += modified_spaces

            if modified_spaces and not self.multiple_mutations:
                break

        return all_modified_spaces

    @staticmethod
    def score(modified_spaces: Sequence['Space'],
              opt_constraints: Sequence['Constraint']) -> Optional[float]:
        """
        Computes the score of the modified spaces
        TODO : we could do better than a simple arithmetic sum
        :param modified_spaces:
        :param opt_constraints:
        :return:
        """
        total_score = None
        for constraint in opt_constraints:
            for space in modified_spaces:
                if total_score is None:
                    total_score = constraint.score(space)
                else:
                    total_score += constraint.score(space)
        return total_score

    def flush(self):
        """
        removes the cache
        :return:
        """
        self._tried = set()
