from typing import Callable, Literal, Sequence

import pandas as pd  # type: ignore
from plotly.graph_objects import Figure  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from plottah.colors import PlotColors
from plottah.plot_handler import PlotHandler
from plottah.plots import BinEventRatePlot, DistPlot, RocCurvePlot
from plottah.plots.bin_event_rate_plot import (
    _get_max_bar_height,
    _get_max_event_rate_height,
)

#### Responsibility: put together specific Plots and Handlers into graphs


def build_standard_numerical_univariate_plot(
    df: pd.DataFrame,
    feature_col: str,
    target: str,
    feature_type: Literal["categorical", "numerical"] = "numerical",
    colors: PlotColors = PlotColors(),
    hoverinfo: Literal["all", "none", "skip"] = "all",
    n_bins: int = 10,
    bins: Sequence[int | float | str] | None = None,
    specs: Sequence[Sequence[dict]] | None = None,
    **kwargs,
) -> PlotHandler:
    """
    Build a standard numerical univariate plot consisting of ROC Curve, Distribution Plot, and Bin Event Rate Plot.

    This function compiles various subplots into a comprehensive plot that visualizes the relationship between a numerical feature and a target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data to plot.
    feature_col : str
        The name of the feature column to analyze.
    target : str
        The name of the target column.
    feature_type : Literal["categorical", "numerical"], optional
        The type of the feature, by default "numerical".
    colors : PlotColors, optional
        An instance of `PlotColors` to define the color scheme, by default `PlotColors()`.
    hoverinfo : Literal["all", "none", "skip"], optional
        Defines the hover information behavior, by default "all".
    n_bins : int, optional
        The number of bins to use for the Bin Event Rate Plot, by default 10.
    bins : Sequence[int | float | str] | None, optional
        Custom bin edges. If `None`, bins will be determined automatically, by default None.
    specs : Sequence[Sequence[dict]] | None, optional
        Custom layout specifications for the subplots. If `None`, default specifications are used, by default None.
    **kwargs
        Additional keyword arguments, such as `distplot_q_min` and `distplot_q_max`.

    Returns
    -------
    PlotHandler
        An instance of `PlotHandler` containing the assembled plot.
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
    )

    return plot


def build_standard_categorical_univariate_plot(
    df: pd.DataFrame,
    feature_col: str,
    target: str,
    feature_type: Literal["categorical", "numerical"] = "categorical",
    colors: PlotColors = PlotColors(),
    hoverinfo: Literal["all", "none", "skip"] = "all",
    n_bins: int = 10,
    bins: Sequence[int | float | str] | None = None,
    specs: Sequence[Sequence[dict]] | None = None,
    **kwargs,
) -> PlotHandler:
    """
    Build a standard categorical univariate plot consisting of Bin Event Rate Plot.

    This function creates a plot that visualizes the event rates across different categories of a categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data to plot.
    feature_col : str
        The name of the categorical feature column to analyze.
    target : str
        The name of the target column.
    feature_type : Literal["categorical", "numerical"], optional
        The type of the feature, by default "categorical".
    colors : PlotColors, optional
        An instance of `PlotColors` to define the color scheme, by default `PlotColors()`.
    hoverinfo : Literal["all", "none", "skip"], optional
        Defines the hover information behavior, by default "all".
    n_bins : int, optional
        The number of bins to use for the Bin Event Rate Plot, by default 10.
    bins : Sequence[int | float | str] | None, optional
        Custom bin edges. If `None`, bins will be determined automatically, by default None.
    specs : Sequence[Sequence[dict]] | None, optional
        Custom layout specifications for the subplot. If `None`, default specifications are used, by default None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    PlotHandler
        An instance of `PlotHandler` containing the assembled plot.
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


# TODO: event rate should span the entire plot
# TODO: want a type that preserves the order of bins, but discrete values...
def build_split_bin_event_rate_plot(
    df_top: pd.DataFrame,
    df_bottom: pd.DataFrame,
    feature_col: str,
    target: str,
    bins: Sequence[int | float | str],
    top_x_title: str,
    bottom_x_title: str,
    feature_type: Literal["numerical", "categorical"] = "numerical",
    colors: PlotColors = PlotColors(),
    hoverinfo="all",
    horizontal_spacing: float = 0.1,
    vertical_spacing: float = 0.15,
    tick_font_size: int = 12,
    title_font_size: int = 14,
    legend_font_size: int = 12,
    y_title: str | None = None,
    secondary_y_title: str | None = None,
    title_standoff: int = 5,
    fillna_event_rate: bool = True,
) -> Figure:
    """
    Build a split bin event rate plot with separate subplots for two different dataframes.

    This function creates a Plotly figure with two subplots, each showing the event rates across specified bins for the top and bottom dataframes.

    Parameters
    ----------
    df_top : pd.DataFrame
        The top dataframe containing the data to plot.
    df_bottom : pd.DataFrame
        The bottom dataframe containing the data to plot.
    feature_col : str
        The name of the feature column to analyze.
    target : str
        The name of the target column.
    bins : Sequence[int | float | str]
        The bin edges to use for the event rate plots.
    top_x_title : str
        The title for the x-axis of the top subplot.
    bottom_x_title : str
        The title for the x-axis of the bottom subplot.
    feature_type : Literal["numerical", "categorical"], optional
        The type of the feature, by default "numerical".
    colors : PlotColors, optional
        An instance of `PlotColors` to define the color scheme, by default `PlotColors()`.
    hoverinfo : str, optional
        Defines the hover information behavior, by default "all".
    horizontal_spacing : float, optional
        The horizontal spacing between subplots, by default 0.1.
    vertical_spacing : float, optional
        The vertical spacing between subplots, by default 0.15.
    tick_font_size : int, optional
        The font size for the tick labels, by default 12.
    title_font_size : int, optional
        The font size for the titles, by default 14.
    legend_font_size : int, optional
        The font size for the legend text, by default 12.
    y_title : str | None, optional
        The title for the primary y-axis, by default None.
    secondary_y_title : str | None, optional
        The title for the secondary y-axis, by default None.
    title_standoff : int, optional
        The distance of the title from the axis, by default 5.
    fillna_event_rate : bool, optional
        Whether to fill NaN values in the event rate, by default True.

    Returns
    -------
    Figure
        A Plotly `Figure` object containing the assembled split bin event rate plots.
    """

    # Create empty plotly figure using subplots, 2x2
    fig: Figure = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"secondary_y": True}] * 1, [{"secondary_y": True}] * 1],
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    # create the plots
    top_plot = BinEventRatePlot(
        bins=bins,
        feature_type=feature_type,
        tick_font_size=tick_font_size,
        title_font_size=title_font_size,
        x_title=top_x_title,
        y_title=y_title,
        secondary_y_title=secondary_y_title,
        colors=colors,
        hoverinfo=hoverinfo,
        title_standoff=title_standoff,
        fillna_event_rate=fillna_event_rate,
        # FIXED ARGUMENTS
        show_legend=True,
    )
    bottom_plot = BinEventRatePlot(
        bins=bins,
        feature_type=feature_type,
        tick_font_size=tick_font_size,
        title_font_size=title_font_size,
        x_title=bottom_x_title,
        y_title=y_title,
        secondary_y_title=secondary_y_title,
        colors=colors,
        hoverinfo=hoverinfo,
        title_standoff=title_standoff,
        fillna_event_rate=fillna_event_rate,
        # FIXED ARGUMENTS
        show_legend=False,
    )

    # do the math
    top_plot.do_math(
        df=df_top,
        feature_col=feature_col,
        target_col=target,
    )
    bottom_plot.do_math(
        df=df_bottom,
        feature_col=feature_col,
        target_col=target,
    )

    # max bar height
    max_bar_height = _get_max_bar_height([top_plot, bottom_plot])
    max_event_rate_height = _get_max_event_rate_height([top_plot, bottom_plot])

    # Replace the dictionary with a list of tuples
    plot_positions = [
        (top_plot, 1, 1, "yaxis", "yaxis2"),
        (bottom_plot, 2, 1, "yaxis3", "yaxis4"),
    ]

    # Update the loop to use the list of tuples
    for plot, row, col, primary_y, secondary_y in plot_positions:
        for trace_dict in plot.get_traces():
            fig.add_trace(
                trace_dict["trace"],
                secondary_y=trace_dict["secondary_y"],
                row=row,
                col=col,
            )

            # update axes layout if specifed
            if plot.get_x_axes_layout(row, col) is not None:
                fig.update_xaxes(**plot.get_x_axes_layout(row, col))

            if plot.get_y_axes_layout(row, col) is not None:
                fig.update_yaxes(**plot.get_y_axes_layout(row, col))

            # Set consistent y-axis ranges for both primary and secondary y-axes
            fig.update_yaxes(
                range=[0, max_bar_height],
                row=row,
                col=col,
                secondary_y=False,
            )  # For fraction of observations
            fig.update_yaxes(
                range=[0, max_event_rate_height],
                row=row,
                col=col,
                secondary_y=True,
            )  # For event rate

            # TODO: this is a quick fix -- SHOULD BE FIXED
            # set title for secondary y-axis
            fig.layout[primary_y].title.text = y_title or "Fraction of Observations"
            fig.layout[secondary_y].title.text = secondary_y_title or "Event Rate"

            # After all the traces are added, update the legend font size
            fig.update_layout(legend=dict(font=dict(size=legend_font_size)))

    # show the figure
    return fig


SinglePlotCallable = Callable[..., PlotHandler | Figure]
PLOT_BUILDERS_DICT: dict[str, SinglePlotCallable] = {
    "categorical": build_standard_categorical_univariate_plot,
    "numerical": build_standard_numerical_univariate_plot,
    "split_bin_event_rate": build_split_bin_event_rate_plot,
}


if __name__ == "__main__":
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    from plottah.colors import PlotColors

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # Create two dataframes with slightly different distributions
    df = pd.DataFrame(
        {
            "predictor": np.concatenate(
                [
                    np.random.normal(0, 1, n_samples // 2),  # First half
                    np.random.normal(2, 1, n_samples // 2),  # Second half
                ]
            ),
            "target": np.random.binomial(1, 0.3, n_samples),
            "split": np.concatenate(
                [
                    np.ones(n_samples // 2),  # First half
                    np.zeros(n_samples // 2),  # Second half
                ]
            ),
        }
    )

    # Test the split bin event rate plot
    plot = build_split_bin_event_rate_plot(
        df_top=df.loc[df["split"] == 1],
        df_bottom=df.loc[df["split"] == 0],
        feature_col="predictor",
        target="target",
        bins=[0, 1, 2, 3, 4, 5],
        top_x_title="Predictor for (split == 1)",
        bottom_x_title="Predictor for (split == 0)",
        colors=PlotColors(),
        tick_font_size=18,
        title_font_size=18,
        legend_font_size=18,
        fillna_event_rate=False,
        title_standoff=10,
    )

    plot.show()
    print("Plot generated successfully")
