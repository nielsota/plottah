from typing import Optional
from pathlib import Path

import re
import yaml
import pandas as pd

from pydantic import BaseModel, ValidationError, validator

# use this over Literal to make custom error containing more info
ALLOWED_TYPES = ["float", "int", "categorical"]
ROOT_DIR = Path(__file__).parent.parent


def parse_from_yaml(path_to_yaml):
    """
    ability to parse config.yaml
    """
    print(ROOT_DIR)
    with open(path_to_yaml) as f:
        config = yaml.safe_load(f)

    return config


def is_in_unit_interval(number):
    return 0 <= number <= 1


def validate_color_string(color_string):
    pattern = r"^\d{1,3},\s?\d{1,3},\s?\d{1,3}$"
    return re.match(pattern, color_string) is not None


class FeatureSchema(BaseModel):
    name: str
    type: Optional[str]
    n_bins: Optional[int] = 10
    bins: Optional[list[int]] = None
    distplot_q_min: Optional[float] = None
    distplot_q_max: Optional[float] = None

    # needs to be hashable: https://stackoverflow.com/questions/63721614/unhashable-type-in-fastapi-request
    class Config:
        frozen = True

    @validator("type")
    def validate_type(cls, v, values):
        if v not in ALLOWED_TYPES:
            raise ValueError(
                f'Type for {values["name"]} must be in {ALLOWED_TYPES}, you passed type: {v}'
            )
        return v

    @validator("bins")
    def validate_bins_and_nbins(cls, bins, values):

        if len(bins) != values["n_bins"]:
            values["n_bins"] = len(bins)
            print(f'setting n_bins for {values["name"]} to {len(bins)} based on bins')
        return bins

    @validator("distplot_q_min")
    def validate_distplot_q_min(cls, v, values):

        # check the type
        if not isinstance(v, float):
            raise ValueError(
                f'The min quantile you provided for {values["name"]} must be of type float, you passed type: {type(v)}'
            )

        # check on the unit interval
        if not is_in_unit_interval(v):
            raise ValueError(
                f'The min quantile you for {values["name"]} must be between 0 and 1, you passed: {v}'
            )
        return v

    @validator("distplot_q_max")
    def validate_distplot_q_max(cls, v, values):

        # check the type
        if not isinstance(v, float):
            raise ValueError(
                f'The max quantile you provided for {values["name"]} must be of type float, you passed type: {type(v)}'
            )

        # check on the unit interval
        if not is_in_unit_interval(v):
            raise ValueError(
                f'The max quantile you for {values["name"]} must be between 0 and 1, you passed: {v}'
            )

        # check ordering
        distplot_q_min = values.get("distplot_q_min", None)
        if distplot_q_min is not None and v is not None and distplot_q_min > v:
            raise ValueError("distplot_q_min must not be greater than distplot_q_max")

        return v


class Settings(BaseModel):
    file_path: Path
    images_output_path: Path
    powerpoint_output_path: Path
    features: list[FeatureSchema]
    target: str
    primary_color: str
    secondary_color: str
    tertiary_color: str
    grey_tint_color: str

    @validator("file_path", "images_output_path", pre=True)
    def path_must_exist(cls, v):
        """
        validates path specified in config. the path/file must exist before the code can execute the ensure valid reading and writing locations exist
        """

        if not isinstance(v, Path):
            v = Path(v)

        if not v.exists():
            raise ValueError(
                f"Directory: {v} does not exist \n Please ensure the config.yaml contains a valid directory"
            )
        return v

    @validator("powerpoint_output_path")
    def parent_dir_must_exist(cls, v):
        """
        validates path specified in config. the path/file must exist before the code can execute the ensure valid reading and writing locations exist
        """

        if not isinstance(v, Path):
            v = Path(v)

        if not v.parent.exists():
            raise ValueError(
                f"Directory: {v.parent} does not exist \n Please ensure the config.yaml contains a valid directory"
            )
        return v

    @validator(
        "primary_color",
        "secondary_color",
        "tertiary_color",
        "grey_tint_color",
        pre=True,
    )
    def test_color_pattern(cls, v):
        """
        matches the string pattern provided against a format of 3 digits, 3 digits, 3 digits
        """

        if not validate_color_string(v):
            raise ValueError(f"Colors must be of format DDD, DDD, DDD (now {v})")
        return v

    @validator("features")
    def check_columns_exist(cls, v, values):
        """
        validates that each of the columns provided in the config exists in the dataframe
        """

        # can only use file path if test was succesfull
        if "file_path" not in values.keys():
            raise ValueError("cannot test features because file_path invalid")
        sample_df = pd.read_csv(values["file_path"], nrows=10)

        for feature in v:
            if feature.name not in sample_df.columns:
                raise ValueError(
                    f"Column: {feature} does not exist is dataframe \n Please ensure the config.yaml contains only valid columns"
                )

        return v

    @validator("target")
    def check_target_exists(cls, v, values):
        """
        validates that the target exists in the dataframe
        """

        # can only use file path if test was succesfull
        if "file_path" not in values.keys():
            raise ValueError("cannot test features because file_path invalid")
        sample_df = pd.read_csv(values["file_path"], nrows=10)

        if v not in sample_df.columns:
            raise ValueError(
                f"Column: {v} does not exist is dataframe \n Please ensure the config.yaml contains only valid columns"
            )
        return v


# create a config object containing all the validated settings
config = parse_from_yaml(str(ROOT_DIR / "config.yaml"))
settings = Settings(**config)

# for debugging purposes
if __name__ == "__main__":
    config = parse_from_yaml(str(ROOT_DIR / "config.yaml"))
    print(config)
    settings = Settings(**config)
    print(settings)
