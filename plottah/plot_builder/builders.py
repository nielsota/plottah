import pathlib
from collections import defaultdict
from typing import Dict

from plottah.colors import PlotColors
from plottah.plot_builder.specific_builders import PLOT_BUILDERS_DICT
from plottah.plot_handler import PlotHandler


def build_univariate_plot(
    df,
    feature_col: str,
    target: str,
    feature_type: str = "float",
    specs: list = None,
    colors: PlotColors = PlotColors(),
    show_plot: bool = False,
    hoverinfo="all",
    n_bins: int = 10,
    bins: list = None,
    distplot_q_min: float = None,
    distplot_q_max: float = None,
):
    """
    buils standard univariate plot from days 'ye

    Returns
    """

    # if feature type not-categoriocal, assume numerical plot type
    feature_type = feature_type if feature_type == "categorical" else "numerical"

    # get appropriate plot builder
    plot_builder_function = PLOT_BUILDERS_DICT[feature_type]

    # build plot
    plot = plot_builder_function(
        df,
        feature_col=feature_col,
        target=target,
        feature_type=feature_type,
        colors=colors,
        show_plot=show_plot,
        hoverinfo=hoverinfo,
        n_bins=n_bins,
        bins=bins,
        specs=specs,
        distplot_q_min=distplot_q_min,
        distplot_q_max=distplot_q_max,
    )

    # show the plot if show_plot set to true
    if show_plot:
        plot.show()

    return plot


def build_univariate_plots(
    df,
    features: list[str],
    target: str,
    feature_types: dict[str, str],
    n_bins: dict[str, int] | None = None,
    bins: dict[str, list] | None = None,
    distplot_q_min: dict[str, float] | None = None,
    distplot_q_max: dict[str, float] | None = None,
    save_directory: pathlib.Path | None = None,
    colors: PlotColors = PlotColors(),
    show_plot: bool = False,
    hoverinfo="none",
) -> tuple[dict[str, PlotHandler], list[pathlib.Path]]:
    """
    function that generates standard univariate plots

    Args:
        df (pd.DataFrame): dataframe containing columns to analyze,
        features (list): list of columns,
        save_directory (pathlib.Path()): where to store figures:

    Returns:
        Dict: map from feature name to figure

    """

    # if only a single feature passed wrap in a list
    if isinstance(features, str):
        features = [features]

    # create mapping from bins to none if not passed as argument
    if bins is None:
        bins = {feature: None for feature in features}

    # create mapping from features to 10 if nbins not passed as argument
    if n_bins is None:
        n_bins = {feature: 10 for feature in features}
    else:
        # n_bins = defaultdict(lambda: 10, n_bins)
        for feature in features:
            # user does not have to provide complete mapping
            if feature not in n_bins.keys():
                n_bins[feature] = 10

    # create mapping from features to float if type not provided
    if feature_types is None:
        feature_types = {feature: "float" for feature in features}

    # test - need to return None if user did not provive q min for all features but only some
    if distplot_q_min is None:
        distplot_q_min = {feature: None for feature in features}
    else:
        # user does not have to provide complete mapping
        distplot_q_min = defaultdict(lambda: None, distplot_q_min)

    # test - need to return None if user did not provive q min for all features but only some
    if distplot_q_max is None:
        distplot_q_max = {feature: None for feature in features}
    else:
        # user does not have to provide complete mapping
        distplot_q_max = defaultdict(lambda: None, distplot_q_max)

    # Run loop
    figs = {}
    save_locs = []
    for i, feature in enumerate(features):
        print(f"[{i+1}/{len(features)}] Starting univariate analysis for: {feature}")
        if feature not in df.columns:
            raise ValueError(f"{feature} not in columns of dataframe")

        fig = build_univariate_plot(
            df=df,
            feature_col=feature,
            target=target,
            feature_type=feature_types[feature],
            colors=colors,
            show_plot=show_plot,
            hoverinfo=hoverinfo,
            n_bins=n_bins[feature],
            bins=bins[feature],
            distplot_q_min=distplot_q_min[feature],
            distplot_q_max=distplot_q_max[feature],
        )

        if save_directory is not None:
            save_loc = fig.save_fig(save_directory)
            print(f"[{i+1}/{len(features)}] Saving univariate anaylsis for {feature}")
            save_locs.append(save_loc)

        figs[feature] = fig

    return figs, save_locs


if __name__ == "__main__":

    import pandas as pd
    import numpy as np

    # generate some data
    df = pd.DataFrame(
        {
            "feature": np.random.randint(0, 100, size=100),
            "target": np.random.randint(0, 2, size=100),
        }
    )

    # build plots
    fig = build_univariate_plot(
        df=df,
        feature_col="feature",
        target="target",
        feature_type="float",
        show_plot=False,
    )

    print("done")
