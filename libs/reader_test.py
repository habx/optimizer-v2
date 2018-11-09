# coding=utf-8
"""
Test module for reader module
"""

import pytest
import libs.logsetup as ls
import libs.reader as reader

ls.init()

BLUEPRINT_INPUT_FILES = reader.BLUEPRINT_INPUT_FILES
SPECIFICATION_INPUT_FILES = reader.SPECIFICATION_INPUT_FILES


@pytest.mark.parametrize("input_file", BLUEPRINT_INPUT_FILES)
def test_read_plan(input_file):
    """
    Test. We read a bunch of plan
    :return:
    """
    plan = reader.create_plan_from_file(input_file)

    assert plan.check()


@pytest.mark.parametrize("input_file", SPECIFICATION_INPUT_FILES)
def test_read_specification(input_file):
    """
    Test. We read a bunch of plan
    :return:
    """
    reader.create_specification_from_file(input_file)
    assert True
