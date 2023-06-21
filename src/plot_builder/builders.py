from plot_handler import PlotHandler
from config import settings
from colors import PlotColors

import pathlib
import pandas as pd

from src.plots import DistPlot, RocCurvePlot
from src.plots.BinEventRatePlot import BinEventRatePlot


def build_univariate_plot(
    df,
    feature_col: str,
    target: str,
    colors: PlotColors = PlotColors(),
    show_plot: bool = True,
    hoverinfo="all",
    n_bins: int = 10,
    bins: list = None,
    specs: list = None,
):
    """
    buils standard univariate plot from days 'ye

    Returns
    """

    roc_plot = RocCurvePlot(hoverinfo=hoverinfo, colors=colors)
    dist_plot = DistPlot(hoverinfo=hoverinfo, colors=colors)
    event_plot = BinEventRatePlot(
        hoverinfo=hoverinfo, colors=colors, n_bins=n_bins, bins=bins
    )

    specs = (
        [[{}, {}], [{"colspan": 2, "secondary_y": True}, None]]
        if specs is None
        else specs
    )

    plot = PlotHandler(feature_col, target, specs)

    plot.build(df, feature_col, target, roc_plot, dist_plot, event_plot, show_fig=False)

    if show_plot:
        plot.show()

    return plot


def build_univariate_plots(
    df,
    features: list,
    target: str,
    save_directory: pathlib.Path() = None,
    colors: PlotColors = PlotColors(),
    show_plot: bool = False,
    hoverinfo="all",
    n_bins: int = 10,
    bins: list = None,
):
    if isinstance(features, str):
        features = [features]

    figs = []
    for i, feature in enumerate(features):
        print(f"[{i+1}/{len(features)}] Starting univariate analysis for: {feature}")
        if feature not in df.columns:
            raise ValueError(f"{feature} not in columns of dataframe")

        fig = build_univariate_plot(
            df,
            feature,
            target,
            colors=colors,
            show_plot=show_plot,
            hoverinfo=hoverinfo,
            n_bins=n_bins,
            bins=bins,
        )

        if save_directory is not None:
            fig.save_fig(save_directory)
            print(f"[{i+1}/{len(features)}] Saving univariate anaylsis for {feature}")

        figs.append(figs)

        print("\n")

    return figs
