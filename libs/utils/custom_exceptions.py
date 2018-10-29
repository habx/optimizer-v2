# coding=utf-8
"""
Custom Exceptions module
"""


class OutsideFaceError(ValueError):
    """
    Used to raise an error when trying to insert a face that is not contained
    in the receiving face
    """
    pass


class OutsideVertexError(ValueError):
    """
    Used to raise an error when trying to snap a vertex that is close to the boundary
    of the face to a specific edge
    """
    pass
