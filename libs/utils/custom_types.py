# coding=utf-8
"""
Custom Types module
Contains custom types for typing
"""

from typing import Tuple, Callable


# Vector type and coords type
Vector2d = Coords2d = Tuple[float, float]

# Callback for space
SpaceCutCb = Callable[[Tuple['Edge', 'Edge']], bool]

# rectangular points
FourCoords2d = Tuple[Coords2d, Coords2d, Coords2d, Coords2d]
