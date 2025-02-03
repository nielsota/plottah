import logging
import pathlib
import argparse
from datetime import datetime

import pandas as pd

from plottah.plot_builder import build_univariate_plots, build_powerpoint
from plottah.config import settings
from plottah.colors import PlotColors

parser = argparse.ArgumentParser(description="Run the univariate analyses workflow")
logging.basicConfig(level=logging.WARN)


def main():
    # parse command line arguments
    parser.add_argument(
        "-p",
        "--build_powerpoint",
        nargs="?",
        const=1,
        type=int,
        default=0,
        choices=[0, 1],
        help="Specify whether you would like to automatically add figures to a powerpoint",
    )

    args = parser.parse_args()
    build_pp = args.build_powerpoint

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
        feature_schema.name: (
            feature_schema.bins if feature_schema.bins is not None else None
        )
        for feature_schema in settings.features
    }

    # create mapping from feature name to number of bins
    n_bins = {
        feature_schema.name: feature_schema.n_bins
        for feature_schema in settings.features
    }

    # create mapping from feature name to feature type
    feature_types = {
        feature_schema.name: (
            feature_schema.type if feature_schema.type is not None else "float"
        )
        for feature_schema in settings.features
    }

    # create mapping from feature name optional dist plot quantile clipping lower bound
    distplot_q_min = {
        feature_schema.name: (
            feature_schema.distplot_q_min
            if feature_schema.distplot_q_min is not None
            else None
        )
        for feature_schema in settings.features
    }

    # create mapping from feature name optional dist plot quantile clipping upper bound
    distplot_q_max = {
        feature_schema.name: (
            feature_schema.distplot_q_max
            if feature_schema.distplot_q_max is not None
            else None
        )
        for feature_schema in settings.features
    }

    # build all the univariate plots
    _, fig_locs = build_univariate_plots(
        df=pd.read_csv(settings.file_path),
        features=features,
        target=settings.target,
        feature_types=feature_types,
        save_directory=settings.images_output_path,
        colors=color_palette,
        bins=bins,
        n_bins=n_bins,
        distplot_q_max=distplot_q_max,
        distplot_q_min=distplot_q_min,
    )

    if build_pp:
        now = datetime.now()
        current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
        save_path = pathlib.Path(
            f"./data/powerpoints/univariate_analyses_{current_time}.pptx"
        ).resolve()
        print(f"Building output powerpoint in location {save_path}")
        build_powerpoint(
            fig_locs=fig_locs,
            feature_names=features,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
