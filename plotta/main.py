from plot_builder import build_univariate_plots
from config import settings
from colors import PlotColors

import pandas as pd


def main():
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

    build_univariate_plots(
        pd.read_csv(settings.file_path),
        features,
        settings.target,
        settings.output_path,
        colors=color_palette,
        bins=bins,
    )


if __name__ == "__main__":
    main()
