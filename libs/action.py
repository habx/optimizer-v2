# coding=utf-8
"""
Operator module
"""

from typing import TYPE_CHECKING, Sequence, Union, Optional, Any
import logging

if TYPE_CHECKING:
    from libs.plan import Space
    from libs.mesh import Face
    from libs.selector import Selector
    from libs.mutation import Mutation
    from libs.constraint import Constraint


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
    def __init__(self,
                 selector: 'Selector',
                 mutation: 'Mutation',
                 repeat: bool = False,
                 name: str = ''):
        self.name = name or '{0} + {1}'.format(selector.name, mutation.name)
        self.selector = selector
        self.mutation = mutation
        self.repeat = repeat
        # we store the couple edge and space that we have already tried TODO : improve this
        self._seen = set()

    def __repr__(self):
        return 'Operator: {0}, repeat={1}'.format(self.name, self.repeat)

    def apply_to(self,
                 space_or_face: Union['Space', 'Face'],
                 selector_optional_args: Sequence[Any],
                 constraints: Optional[Sequence['Constraint']] = None) -> Sequence['Space']:
        """
        Applies the operator
        :param space_or_face:
        :param selector_optional_args:
        :param constraints:
        :return:
        """
        # separate imperative constraints from objective constraints
        if constraints is not None:
            imp_constraints = [constraint for constraint in constraints if constraint.imperative]
            opt_constraints = [constraint for constraint in constraints
                               if not constraint.imperative]
        else:
            imp_constraints = []
            opt_constraints = []

        all_modified_spaces = []
        for edge in self.selector.yield_from(space_or_face, *selector_optional_args):
            # check if we already tried
            _id = '{0}-{1}'.format(id(space_or_face), id(edge))
            if _id in self._seen:
                logging.debug('Already tried this edge with this space')
                continue

            initial_score = self.score(self.mutation.will_modify(edge), opt_constraints)
            modified_spaces = self.mutation.apply_to(edge)
            if modified_spaces:
                for constraint in imp_constraints:
                    if not constraint.check(modified_spaces[0]):
                        logging.debug('Constraint breached: {0} - {1}'
                                      .format(constraint.name, modified_spaces[0]))
                        self.mutation.reverse(edge, modified_spaces)
                        modified_spaces = []
                        self._seen.add('{0}-{1}'.format(id(space_or_face), id(edge)))
                        break
                if initial_score is not None:
                    new_score = self.score(modified_spaces, opt_constraints)
                    if new_score >= initial_score:
                        self.mutation.reverse(edge, modified_spaces)
                        self._seen.add('{0}-{1}'.format(id(space_or_face), id(edge)))
                        modified_spaces = []

            all_modified_spaces += modified_spaces
            if modified_spaces and not self.repeat:
                break

        return all_modified_spaces

    @staticmethod
    def score(modified_spaces: Sequence['Space'],
              opt_constraints: Sequence['Constraint']) -> Optional[float]:
        """
        Computes the score of the modified spaces
        TODO : better than a simple arithmetic sum
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
