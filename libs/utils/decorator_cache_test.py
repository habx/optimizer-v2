# coding=utf-8
"""Test of DecoratorCache"""

from libs.utils.decorator_cache import DecoratorCache
from output import DEFAULT_CACHE_OUTPUT_FOLDER
nb_calls = 0


@DecoratorCache([__file__], DEFAULT_CACHE_OUTPUT_FOLDER)
def _decorated_function(x):
    global nb_calls
    print('computing')
    nb_calls += 1
    return x*x


def test_decorated_function():
    # Same behavior when used directly and cached
    assert _decorated_function(1000) == _decorated_function(1000)

    nb_calls = 0
    _decorated_function(1000)
    assert nb_calls == 0
