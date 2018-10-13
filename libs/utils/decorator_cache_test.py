"""Test of DecoratorCache"""

from libs.utils.decorator_cache import DecoratorCache

nb_calls = 0

@DecoratorCache([__file__], 'output/cache')
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
