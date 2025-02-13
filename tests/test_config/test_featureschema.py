import pytest

from plottah.config import FeatureSchema


@pytest.fixture
def return_feature():
    return {"name": "A_TENURE_MONTHS_N", "type": "float", "bins": [0, 1, 2]}


### Features Tests ###
def test_correct_feature(return_feature):
    feature = return_feature.copy()
    FeatureSchema(**feature)


### Features Tests ###
def test_incorrect_type(return_feature):
    feature = return_feature.copy()
    feature["type"] = "footype"

    with pytest.raises(ValueError):
        FeatureSchema(**feature)
