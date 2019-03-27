# coding=utf-8
"""
Test module for reader module
"""

import pytest
import libs.io.reader as reader

# ls.init()


BLUEPRINT_INPUT_FILES = [
    "grenoble_101.json",
    "grenoble_102.json",
    "grenoble_113.json",
    "grenoble_114.json",
    "grenoble_121.json",
    "grenoble_122.json",
    "grenoble_125.json",
    "grenoble_201.json",
    "grenoble_211.json",
    "grenoble_212.json",
    "Levallois_A2-601.json",
    "meurice_LT01.json",
    "meurice_LT02.json",
    "meurice_LT04.json",
    "meurice_LT06.json",
    "meurice_LT07.json",
    "meurice_LT09.json",
    "meurice_LT100.json",
    "meurice_LT101.json",
    "paris-mon18_A601.json",
    "paris-mon18_A602.json",
    "paris-mon18_A603.json",
    "paris-mon18_A604.json",
    "paris-mon18_A605.json",
    "paris-mon18_A606.json",
    "saint-maur-faculte_A001.json",
    "saint-maur-faculte_A102.json",
    "saint-maur-faculte_A103.json",
    "saint-maur-faculte_A104.json",
    "saint-maur-faculte_B001.json",
    "saint-maur-faculte_B002.json",
    "saint-maur-faculte_B011.json",
    "saint-maur-faculte_B112.json",
    "saint-maur-faculte_B121.json",
    "saint-maur-faculte_B153.json",
    "saint-maur-raspail_H01.json",
    "saint-maur-raspail_H02.json",
    "saint-maur-raspail_H03.json",
    "saint-maur-raspail_H04.json",
    "saint-maur-raspail_H05.json",
    "saint-maur-raspail_H06.json",
    "saint-maur-raspail_H07.json",
    "saint-maur-raspail_H08.json",
    "antony_A33.json",
    "antony_B14.json",
    "antony_B22.json",
    "bagneux_A124.json",
    "bagneux_B232.json",
    "bussy_B002.json",
    "bussy_B104.json",
    "edison_10.json",
    "edison_20.json",
    "florent.json",
    "nantes-unile_B701.json",
    "paris18_A402.json",
    "paris18_A501.json",
    "sartrouville_R1.json",
    "vernouillet_A002.json"
]

SPECIFICATION_INPUT_FILES = [
    "grenoble_101_setup0.json",
    "grenoble_102_setup0.json",
    "grenoble_113_setup0.json",
    "grenoble_114_setup0.json",
    "grenoble_115_setup0.json",
    "grenoble_121_setup0.json",
    "grenoble_122_setup0.json",
    "grenoble_125_setup0.json",
    "grenoble_201_setup0.json",
    "grenoble_211_setup0.json",
    "grenoble_212_setup0.json",
    "Levallois_A2-601_setup0.json",
    "meurice_LT01_setup0.json",
    "meurice_LT02_setup0.json",
    "meurice_LT04_setup0.json",
    "meurice_LT06_setup0.json",
    "meurice_LT07_setup0.json",
    "meurice_LT09_setup0.json",
    "meurice_LT100_setup0.json",
    "meurice_LT101_setup0.json",
    "paris-mon18_A601_setup0.json",
    "paris-mon18_A602_setup0.json",
    "paris-mon18_A603_setup0.json",
    "paris-mon18_A604_setup0.json",
    "paris-mon18_A605_setup0.json",
    "paris-mon18_A606_setup0.json",
    "saint-maur-faculte_A001_setup0.json",
    "saint-maur-faculte_A102_setup0.json",
    "saint-maur-faculte_A103_setup0.json",
    "saint-maur-faculte_A104_setup0.json",
    "saint-maur-faculte_B001_setup0.json",
    "saint-maur-faculte_B002_setup0.json",
    "saint-maur-faculte_B011_setup0.json",
    "saint-maur-faculte_B112_setup0.json",
    "saint-maur-faculte_B121_setup0.json",
    "saint-maur-faculte_B153_setup0.json",
    "saint-maur-raspail_H01_setup0.json",
    "saint-maur-raspail_H02_setup0.json",
    "saint-maur-raspail_H03_setup0.json",
    "saint-maur-raspail_H04_setup0.json",
    "saint-maur-raspail_H05_setup0.json",
    "saint-maur-raspail_H06_setup0.json",
    "saint-maur-raspail_H07_setup0.json",
    "saint-maur-raspail_H08_setup0.json",
    "antony_A33_setup0.json",
    "antony_B14_setup0.json",
    "antony_B22_setup0.json",
    "bagneux_A124_setup0.json",
    "bagneux_B232_setup0.json",
    "bussy_B002_setup0.json",
    "bussy_B104_setup0.json",
    "edison_10_setup0.json",
    "edison_20_setup0.json",
    "florent_setup0.json",
    "nantes-unile_B701_setup0.json",
    "paris18_A402_setup0.json",
    "paris18_A501_setup0.json",
    "sartrouville_R1_setup0.json",
    "vernouillet_A002_setup0.json"
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
