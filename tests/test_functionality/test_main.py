from plottah.__main__ import main
from plottah.config import Settings, settings
from plottah.plot_builder import build_univariate_plots, build_univariate_plot
from plottah.colors import PlotColors
import pytest
import pandas as pd


@pytest.fixture
def return_settings():
    return {
        "file_path": "./data/example_data/test_sample.csv",
        "output_path": "./data/images",
        "features": [
            {"name": "A_TENURE_MONTHS_N"},
            {"name": "T_PPG_AVG_6M_N", "n_bins": 4},
            {"name": "D_MAX_DAYS_PAST_DUE_6M_N", "bins": [0, 1, 5, 10, 100]},
            {"name": "LN_LEXISNEXIS_SBFE_SCORE_CURRENT_N"},
            {"name": "T_TOTAL_TRX_NON_FUEL_PROPORTION_1M_N"},
            {"name": "LN_LEXISNEXIS_THIN_FILE_FLAG_B", "type": "categorical"},
        ],
        "target": "FLAG_60_DPD_366_DAYS",
        "primary_color": "231, 30, 87",
        "secondary_color": "153, 204, 235",
        "tertiary_color": "254, 189, 64",
        "grey_tint_color": "110, 111, 115",
    }


def test_main(return_settings):
    config = return_settings.copy()

    # settings will not accept None for output path, but build_univariate_plots will. When passed None to build_univariate_plots, result is that output is not saved
    settings = Settings(**config)
    settings.output_path = None

    # set color palette to use
    color_palette = PlotColors(
        primary_color=settings.primary_color,
        secondary_color=settings.secondary_color,
        tertiary_color=settings.tertiary_color,
        grey_tint_color=settings.grey_tint_color,
    )

    # parse feature names from settings
    features = [feature_schema.name for feature_schema in settings.features]

    # create mapping from feature name to binning
    bins = {
        feature_schema.name: feature_schema.bins
        if feature_schema.bins is not None
        else None
        for feature_schema in settings.features
    }

    # create mapping from feature name to number of bins
    n_bins = {
        feature_schema.name: feature_schema.n_bins
        for feature_schema in settings.features
    }

    # create mapping from feature name to feature type
    feature_types = {
        feature_schema.name: feature_schema.type
        if feature_schema.type is not None
        else "float"
        for feature_schema in settings.features
    }

    build_univariate_plots(
        df=pd.read_csv(settings.file_path),
        features=features,
        target=settings.target,
        feature_types=feature_types,
        save_directory=None,
        colors=color_palette,
        bins=bins,
        n_bins=n_bins,
    )


def test_build_univariate_plot(return_settings):
    # retrieve settings
    config = return_settings.copy()

    # settings will not accept None for output path, but build_univariate_plots will. When passed None to build_univariate_plots, result is that output is not saved
    settings = Settings(**config)
    settings.output_path = None

    # set color palette to use
    color_palette = PlotColors(
        primary_color=settings.primary_color,
        secondary_color=settings.secondary_color,
        tertiary_color=settings.tertiary_color,
        grey_tint_color=settings.grey_tint_color,
    )

    FEATURE, TARGET = settings.features[1].name, settings.target
    fig = build_univariate_plot(
        df=pd.read_csv(settings.file_path),
        feature_col=FEATURE,
        target=TARGET,
        colors=color_palette,
        show_plot=False,
        n_bins=10,
        bins=None,
        specs=None,
        hoverinfo="none",
    )
