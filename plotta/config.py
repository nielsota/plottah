from typing import Optional
import yaml
import pandas as pd
import re

from pathlib import Path
from pydantic import BaseModel, ValidationError, validator

ROOT_DIR = Path(__file__).parent.parent


def parse_from_yaml(path_to_yaml):
    """
    ability to parse config.yaml
    """
    with open(path_to_yaml) as f:
        config = yaml.safe_load(f)

    return config


def validate_color_string(color_string):
    pattern = r"^\d{1,3},\s?\d{1,3},\s?\d{1,3}$"
    return re.match(pattern, color_string) is not None


class FeatureSchema(BaseModel):
    name: str
    bins: Optional[list[int]]
    type: Optional[str]

    # needs to be hashable: https://stackoverflow.com/questions/63721614/unhashable-type-in-fastapi-request
    class Config:
        frozen = True


class Settings(BaseModel):
    file_path: Path
    output_path: Path
    features: list[FeatureSchema]
    target: str
    primary_color: str
    secondary_color: str
    tertiary_color: str
    grey_tint_color: str

    @validator("file_path", "output_path")
    def path_must_exist(cls, v):
        """
        validates path specified in config. the path/file must exist before the code can execute the ensure valid reading and writing locations exist
        """

        if not v.exists():
            raise ValueError(
                f"Directory: {v.relative_to(ROOT_DIR)} does not exist \n Please ensure the config.yaml contains a valid directory"
            )
        return v

    @validator("primary_color", "secondary_color", "tertiary_color", "grey_tint_color")
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

        sample_df = pd.read_csv(values["file_path"], nrows=10)

        if v not in sample_df.columns:
            raise ValueError(
                f"Column: {v} does not exist is dataframe \n Please ensure the config.yaml contains only valid columns"
            )
        return v


config = parse_from_yaml(str(ROOT_DIR / "config.yaml"))
settings = Settings(**config)

if __name__ == "__main__":
    config = parse_from_yaml(str(ROOT_DIR / "config.yaml"))
    print(config)
    settings = Settings(**config)
    print(settings)
