# coding=utf-8
"""
Test module for reader module
"""

import pytest
import libs.logsetup as ls
import libs.reader as reader

# ls.init()


BLUEPRINT_INPUT_FILES = [
    "Levallois_A3_505.json",
    "Levallois_Parisot.json",
    "Levallois_Tisnes.json",
    "Levallois_Creuze.json",
    "Levallois_Meyronin.json",
    "Levallois_Letourneur.json",
    "Antony_A22.json",
    "Antony_A33.json",
    "Antony_B14.json",
    "Antony_B22.json",
    "Bussy_A001.json",
    "Bussy_A101.json",
    "Bussy_A202.json",
    "Bussy_B002.json",
    "Bussy_B104.json",
    "Bussy_Regis.json",
    "Edison_10.json",
    "Edison_20.json",
    "Massy_C102.json",
    "Massy_C204.json",
    "Massy_C303.json",
    "Noisy_A145.json",
    "Noisy_A318.json",
    "Paris18_A301.json",
    "Paris18_A302.json",
    "Paris18_A402.json",
    "Paris18_A501.json",
    "Paris18_A502.json",
    "Sartrouville_RDC.json",
    "Sartrouville_R1.json",
    "Sartrouville_R2.json",
    "Sartrouville_A104.json",
    "Vernouillet_A002.json",
    "Vernouillet_A003.json",
    "Vernouillet_A105.json",
    "saint-maur-raspail_H01.json",
    "saint-maur-raspail_H02.json",
    "saint-maur-raspail_H03.json",
    "saint-maur-raspail_H04.json",
    "saint-maur-raspail_H05.json",
    "saint-maur-raspail_H06.json",
    "saint-maur-raspail_H07.json",
    "saint-maur-raspail_H08.json",
    "paris-venelles_B001.json"
]

SPECIFICATION_INPUT_FILES = [
    "Antony_A22_setup.json",
    "Antony_A33_setup.json",
    "Antony_B14_setup.json",
    "Antony_B22_setup.json",
    "Bussy_A001_setup.json"
]


# BLUEPRINT_INPUT_FILES = reader.BLUEPRINT_INPUT_FILES
# SPECIFICATION_INPUT_FILES = reader.SPECIFICATION_INPUT_FILES


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
