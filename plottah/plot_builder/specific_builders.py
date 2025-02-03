import pathlib
import pandas as pd
from typing import Dict

from plottah.plots import DistPlot, RocCurvePlot, BinEventRatePlot
from plottah.plot_handler import PlotHandler
from plottah.colors import PlotColors

#### Responsibility: put together specific Plots and Handlers into graphs


def build_standard_numerical_univariate_plot(
    df,
    feature_col: str,
    target: str,
    feature_type: str = "float",
    colors: PlotColors = PlotColors(),
    show_plot: bool = False,
    hoverinfo="all",
    n_bins: int = 10,
    bins: list = None,
    specs: list = None,
    **kwargs,
) -> PlotHandler:
    """
    buils standard univariate plot from days 'ye

    Returns
    """

    # unpack the keyword arguments
    distplot_q_min, distplot_q_max = kwargs["distplot_q_min"], kwargs["distplot_q_max"]

    # build all 3 subplots for general plot

    ### CREATION -> should not be here? But this is like main, here we put things together ###
    roc_plot = RocCurvePlot(hoverinfo=hoverinfo, colors=colors)
    dist_plot = DistPlot(
        hoverinfo=hoverinfo,
        colors=colors,
        distplot_q_max=distplot_q_max,
        distplot_q_min=distplot_q_min,
    )
    event_plot = BinEventRatePlot(
        hoverinfo=hoverinfo,
        colors=colors,
        n_bins=n_bins,
        bins=bins,
        feature_type=feature_type,
    )

    # need different specs, user can override but not required
    specs = (
        [[{}, {}], [{"colspan": 2, "secondary_y": True}, None]]
        if specs is None
        else specs
    )

    plot = PlotHandler(
        feature_col=feature_col, target_col=target, specs=specs, plot_title=None
    )

    ### USE ###
    # set show fig to false, show explicitly below
    plot.build(
        df=df,
        feature_col=feature_col,
        target=target,
        topleft=roc_plot,
        topright=dist_plot,
        bottom=event_plot,
        show_fig=show_plot,
    )

    return plot


def build_standard_categorical_univariate_plot(
    df,
    feature_col: str,
    target: str,
    feature_type: str = "categorical",
    colors: PlotColors = PlotColors(),
    show_plot: bool = False,
    hoverinfo="all",
    n_bins: int = 10,
    bins: list = None,
    specs: list = None,
    **kwargs,
) -> PlotHandler:
    """
    buils standard univariate plot from days 'ye

    Returns
    """

    # create plot and do math
    event_plot = BinEventRatePlot(
        hoverinfo=hoverinfo,
        colors=colors,
        n_bins=n_bins,
        bins=bins,
        feature_type=feature_type,
    )
    event_plot.do_math(df, feature_col, target)

    # need different specs, user can override but not required
    specs = [[{"colspan": 2, "secondary_y": True}, None]] if specs is None else specs

    # set up handler and build only subplot
    plot = PlotHandler(
        feature_col=feature_col,
        target_col=target,
        specs=specs,
        plot_title=f"{feature_col}: Event Rates",
    )
    plot.build_subplot(event_plot, 1, 1)

    return plot


# Map
PLOT_BUILDERS_DICT = {
    "categorical": build_standard_categorical_univariate_plot,
    "numerical": build_standard_numerical_univariate_plot,
}
