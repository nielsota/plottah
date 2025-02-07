import pandas as pd

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


# TODO: should have the same y-axes
# TODO: should not have 2 legends
# TODO: bins should be the same
# TODO: title fonts should be customizable
# TODO: event rate should span the entire plot
# TODO: want a type that preserves the order of bins, but discrete values...
def build_split_bin_event_rate_plot(
    df_top: pd.DataFrame,
    df_bottom: pd.DataFrame,
    feature_col: str,
    target: str,
    bins: list[float],
    feature_type: str = "numerical",
    colors: PlotColors = PlotColors(),
    hoverinfo="all",
) -> PlotHandler:
    """
    buils standard univariate plot from days 'ye

    Returns
    """

    # create plot and do math
    event_plot_top = BinEventRatePlot(
        hoverinfo=hoverinfo,
        colors=colors,
        n_bins=len(bins),
        bins=bins,
        feature_type=feature_type,
    )
    event_plot_top.do_math(df_top, feature_col, target)

    event_plot_bottom = BinEventRatePlot(
        hoverinfo=hoverinfo,
        colors=colors,
        n_bins=len(bins),
        bins=bins,
        feature_type=feature_type,
    )
    event_plot_bottom.do_math(df_bottom, feature_col, target)

    # override event rate for both plots so they are the same
    event_rate = np.mean(pd.concat([df_top, df_bottom])[target])
    event_plot_top.event_rate = event_rate
    event_plot_bottom.event_rate = event_rate

    # 2 rows, each with 2 columns and a secondary y axis
    specs = [
        [{"colspan": 2, "secondary_y": True}, None],
        [{"colspan": 2, "secondary_y": True}, None],
    ]

    # set up handler and build only subplot
    plot = PlotHandler(
        feature_col=feature_col,
        target_col=target,
        specs=specs,
        plot_title=f"{feature_col}: Event Rates",
    )
    plot.build_subplot(event_plot_top, 1, 1)
    plot.build_subplot(event_plot_bottom, 2, 1)

    # Update legend position and hide second legend
    plot.fig.update_layout(
        legend=dict(
            x=1.30,  # Move legend further right
            y=0.9,  # Position for top subplot legend
        ),
        legend2=dict(
            visible=False,
        ),
        legend_tracegroupgap=180,
    )

    return plot


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
            "feature": np.concatenate(
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
        feature_col="feature",
        target="target",
        bins=[0, 1, 2, 3, 4],
        colors=PlotColors(),
    )

    plot.show()
    print("Plot generated successfully")
