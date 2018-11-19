# coding=utf-8
"""
Operator module
"""

from typing import TYPE_CHECKING, Sequence, Union, Optional, Any

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
        constraints = constraints or []
        all_modified_spaces = []
        for edge in self.selector.yield_from(space_or_face, *selector_optional_args):
            modified_spaces = self.mutation.apply_to(edge)
            if modified_spaces:
                for constraint in constraints:
                    if not constraint.check(modified_spaces[0]):
                        self.mutation.reverse(edge, modified_spaces)
                        modified_spaces = []
                        break

                all_modified_spaces += modified_spaces
                if not self.repeat:
                    break

        return all_modified_spaces

