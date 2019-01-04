# coding=utf-8
"""
Test module for selector module
"""
import pytest

from libs.selector import SELECTORS, SELECTOR_FACTORIES
from libs.plan import Plan

@pytest.fixture
def l_plan():
    """
    Creates a weirdly shaped plan

                 500, 1000       1200, 1200
                     +---------------+
                     |               |
                   |                 |
        0, 500   |                   |
           +--+ 200, 500   1000, 400 |
           |                   +-----+ 1200, 400
           |      500, 200     |
           |      ---*--       |
           |   ---      ---    |
           +---            ----+
         0, 0              1000, 0

    :return:
    """
    boundaries = [(0, 0), (500, 200), (1000, 0), (1000, 400), (1200, 400), (1200, 1200),
                  (500, 1000), (200, 500), (0, 500)]
    return Plan("L_shaped").from_boundary(boundaries)


@pytest.fixture
def weird_plan() -> Plan:
    """
    Creates a weirdly shaped plan

                 500, 1200       1200, 1200
                     +---------------+
                     |               |
                   |                 |
        0, 500   |                   |
           +--+ 200, 500   1000, 400 |
           |                   +-----+ 1200, 400
           |                   |
           |                   |
           +-------------------+
         0, 0              1000, 0

    :return:
    """

    boundaries = [(0, 0), (1000, 0), (1000, 400), (1200, 400), (1200, 1200),
                  (500, 1200), (200, 500), (0, 500)]
    return Plan("weird_shaped").from_boundary(boundaries)


def test_oriented_selector(weird_plan):
    """
    Test
    :param weird_plan:
    :return:
    """
    empty_space_boundary = [(1000, 400), (1200, 400), (1200, 800), (1000, 800)]
    weird_plan.insert_space_from_boundary(empty_space_boundary)
    selector = SELECTOR_FACTORIES["oriented_edges"](["horizontal"])
    edges = list(selector.yield_from(weird_plan.empty_space))
    weird_plan.plot()
    assert str(edges) == "[Edge:[(1000.0, 800.0), (1200.0, 800.0)]]"


def test_boundary_selector(l_plan):
    """
    Test selector
    :return: 
    """
    selector = SELECTORS['space_boundary']
    edges = list(selector.yield_from(l_plan.empty_space))
    other_edges = list(l_plan.empty_space.edges)
    assert edges == other_edges


def test_previous_angle_salient_non_ortho_selector(l_plan):
    """
    Test selector
    :return:
    """
    selector = SELECTORS["previous_angle_salient_non_ortho"]
    edges = list(selector.yield_from(l_plan.empty_space))
    assert str(edges) == ("[Edge:[(500.0, 200.0), (1000.0, 0.0)], " +
                          "Edge:[(200.0, 500.0), (0.0, 500.0)]]")


def test_next_angle_salient_non_ortho(l_plan):
    """
    Test selector
    :param l_plan:
    :return:
    """
    selector = SELECTORS["next_angle_salient_non_ortho"]
    edges = list(selector.yield_from(l_plan.empty_space))
    assert str(edges) == ("[Edge:[(0.0, 0.0), (500.0, 200.0)],"
                          " Edge:[(500.0, 1000.0), (200.0, 500.0)]]")


def test_next_angle_convex_non_ortho(l_plan):
    """
    Test selector
    :param l_plan:
    :return:
    """
    selector = SELECTORS["next_angle_convex_non_ortho"]
    edges = list(selector.yield_from(l_plan.empty_space))
    assert str(edges) == "[Edge:[(1200.0, 1200.0), (500.0, 1000.0)]]"


def test_previous_angle_convex_non_ortho(l_plan):
    """
    Test selector
    :param l_plan:
    :return:
    """
    selector = SELECTORS["previous_angle_convex_non_ortho"]
    edges = list(selector.yield_from(l_plan.empty_space))
    assert str(edges) == "[Edge:[(500.0, 1000.0), (200.0, 500.0)]]"


def test_previous_angle_salient_ortho(l_plan):
    """
    Test selector
    :param l_plan:
    :return:
    """
    selector = SELECTORS["previous_angle_salient_ortho"]
    edges = list(selector.yield_from(l_plan.empty_space))
    assert str(edges) == "[Edge:[(1000.0, 400.0), (1200.0, 400.0)]]"
