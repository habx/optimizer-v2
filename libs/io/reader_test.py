# coding=utf-8
"""
Test module for reader module
"""

import pytest
import libs.io.reader as reader

# ls.init()


BLUEPRINT_INPUT_FILES = [
    "011.json",
    "012.json",
    "013.json",
    "014.json",
    "016.json",
    "017.json",
    "018.json",
    "019.json",
    "020.json",
    "021.json",
    "022.json",
    "024.json",
    "025.json",
    "026.json",
    "027.json",
    "028.json",
    "029.json",
    "030.json",
    "031.json",
    "035.json",
    "036.json",
    "037.json",
    "038.json",
    "039.json",
    "040.json",
    "043.json",
    "044.json",
    "045.json",
    "046.json",
    "047.json",
    "048.json",
    "049.json",
    "050.json",
    "051.json",
    "052.json",
    "053.json",
    "054.json",
    "055.json",
    "056.json",
    "057.json",
    "058.json",
    "059.json",
    "060.json",
    "001.json",
    "002.json",
    "003.json",
    "004.json",
    "005.json",
    "006.json",
    "007.json",
    "008.json",
    "009.json",
    "010.json",
    "034.json",
    "041.json",
    "042.json",
    "061.json",
    "062.json"
]

SPECIFICATION_INPUT_FILES = [
    "011_setup0.json",
    "012_setup0.json",
    "013_setup0.json",
    "014_setup0.json",
    "015_setup0.json",
    "016_setup0.json",
    "017_setup0.json",
    "018_setup0.json",
    "019_setup0.json",
    "020_setup0.json",
    "021_setup0.json",
    "022_setup0.json",
    "024_setup0.json",
    "025_setup0.json",
    "026_setup0.json",
    "027_setup0.json",
    "028_setup0.json",
    "029_setup0.json",
    "030_setup0.json",
    "031_setup0.json",
    "035_setup0.json",
    "036_setup0.json",
    "037_setup0.json",
    "038_setup0.json",
    "039_setup0.json",
    "040_setup0.json",
    "043_setup0.json",
    "044_setup0.json",
    "045_setup0.json",
    "046_setup0.json",
    "047_setup0.json",
    "048_setup0.json",
    "049_setup0.json",
    "050_setup0.json",
    "051_setup0.json",
    "052_setup0.json",
    "053_setup0.json",
    "054_setup0.json",
    "055_setup0.json",
    "056_setup0.json",
    "057_setup0.json",
    "058_setup0.json",
    "059_setup0.json",
    "060_setup0.json",
    "001_setup0.json",
    "002_setup0.json",
    "003_setup0.json",
    "004_setup0.json",
    "005_setup0.json",
    "006_setup0.json",
    "007_setup0.json",
    "008_setup0.json",
    "009_setup0.json",
    "010_setup0.json",
    "034_setup0.json",
    "041_setup0.json",
    "042_setup0.json",
    "061_setup0.json",
    "062_setup0.json"
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
