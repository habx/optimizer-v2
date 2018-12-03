# coding=utf-8
"""
Custom Types module
Contains custom types for typing
"""

from typing import Tuple, Callable, Optional

# Vector type and coords type
Vector2d = Coords2d = Tuple[float, float]

# Callback for space
SpaceCutCb = Callable[[Tuple['Edge', 'Edge']], bool]

# rectangular points
FourCoords2d = Tuple[Coords2d, Coords2d, Coords2d, Coords2d]

# any number of points
ListCoords2d = Tuple[Coords2d, ...]

# edge, edge, face
TwoEdgesAndAFace = Optional[Tuple['Edge', 'Edge', Optional['Face']]]
