"""
Test module for core module
"""
from libs.refiner.core import Toolbox


def test_basics():
    toolbox = Toolbox()
    toolbox.configure("fitness", "", (1.0, -1.0))
    toolbox.configure("individual", "", toolbox.fitness)
    ind = toolbox.individual()

    assert ind.fitness.weights == (1.0, -1.0)

