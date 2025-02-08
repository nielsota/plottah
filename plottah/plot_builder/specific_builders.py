from typing import Literal

import pandas as pd
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from plottah.colors import PlotColors
from plottah.plot_handler import PlotHandler
from plottah.plots import BinEventRatePlot, DistPlot, RocCurvePlot

#### Responsibility: put together specific Plots and Handlers into graphs


def build_standard_numerical_univariate_plot(
    df: pd.DataFrame,
    feature_col: str,
    target: str,
    feature_type: str = "float",
    colors: PlotColors = PlotColors(),
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
    )

    return plot


def build_standard_categorical_univariate_plot(
    df: pd.DataFrame,
    feature_col: str,
    target: str,
    feature_type: str = "categorical",
    colors: PlotColors = PlotColors(),
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


# TODO: event rate should span the entire plot
# TODO: want a type that preserves the order of bins, but discrete values...
def build_split_bin_event_rate_plot(
    df_top: pd.DataFrame,
    df_bottom: pd.DataFrame,
    feature_col: str,
    target: str,
    bins: list[float],
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
    primary_y: Literal["fraction", "event_rate"] = "fraction",
) -> Figure:
    """
    buils standard univariate plot from days 'ye

    Returns
    """

    # Create empty plotly figure using subplots, 2x2
    fig: Figure = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"secondary_y": True}] * 1, [{"secondary_y": True}] * 1],
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    fig.print_grid()

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
        primary_y=primary_y,
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
        primary_y=primary_y,
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
    max_bar_height = (
        np.ceil(
            max(
                [
                    top_plot.max_bar_height,
                    bottom_plot.max_bar_height,
                ]
            )
            * 11
        )
        / 10
    )
    max_event_rate_height = (
        np.ceil(
            max(
                [
                    top_plot.max_event_rate_height,
                    bottom_plot.max_event_rate_height,
                ]
            )
            * 11
        )
        / 10
    )

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
                secondary_y=True if primary_y == "fraction" else False,
            )  # For fraction of observations
            fig.update_yaxes(
                range=[0, max_event_rate_height],
                row=row,
                col=col,
                secondary_y=True if primary_y == "event_rate" else False,
            )  # For event rate

            # TODO: this is a quick fix -- SHOULD BE FIXED
            # set title for secondary y-axis
            fig.layout[primary_y].title.text = y_title or (
                "Fraction of Observations" if primary_y == "fraction" else "Event Rate"
            )
            fig.layout[secondary_y].title.text = secondary_y_title or (
                "Event Rate" if primary_y == "fraction" else "Fraction of Observations"
            )
            # After all the traces are added, update the legend font size
            fig.update_layout(legend=dict(font=dict(size=legend_font_size)))

    # show the figure
    return fig


# Map
PLOT_BUILDERS_DICT = {
    "categorical": build_standard_categorical_univariate_plot,
    "numerical": build_standard_numerical_univariate_plot,
    "split_bin_event_rate": build_split_bin_event_rate_plot,
}


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

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
        top_x_title="Predictor for Split 1",
        bottom_x_title="Predictor for Split 0",
        colors=PlotColors(),
        tick_font_size=18,
        title_font_size=18,
        legend_font_size=18,
        fillna_event_rate=False,
        primary_y="event_rate",
        title_standoff=10,
    )

    plot.show()
    print("Plot generated successfully")
