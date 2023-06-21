from typing import Optional
import yaml
import pandas as pd

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
        validators are classmethods that take a instance variable (v) and do validation, and return v if tests passed
        """
        # make directory if not exists
        if not v.exists():
            raise ValueError(
                f"Directory: {v.relative_to(ROOT_DIR)} does not exist \n Please ensure the config.yaml contains a valid directory"
            )
        return v

    @validator("features")
    def check_columns_exist(cls, v, values):
        """
        validators are classmethods that take a instance variable (v) and do validation, and return v if tests passed
        """
        # make directory if not exists

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
        validators are classmethods that take a instance variable (v) and do validation, and return v if tests passed
        """
        # make directory if not exists

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
