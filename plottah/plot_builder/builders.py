from plottah.plot_handler import PlotHandler
from plottah.colors import PlotColors
from typing import Dict

import pathlib
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt


from plottah.plots import DistPlot, RocCurvePlot, BinEventRatePlot


def build_univariate_plot(
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
):
    """
    buils standard univariate plot from days 'ye

    Returns
    """

    # if feature type is categorical, standard plot is only event rate plot
    if feature_type == "categorical":
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
        specs = (
            [[{"colspan": 2, "secondary_y": True}, None]] if specs is None else specs
        )

        # set up handler and build only subplot
        plot = PlotHandler(
            feature_col=feature_col,
            target_col=target,
            specs=specs,
            plot_title=f"{feature_col}: Event Rates",
        )
        plot.build_subplot(event_plot, 1, 1)
    else:
        # build all 3 subplots for general plot
        roc_plot = RocCurvePlot(hoverinfo=hoverinfo, colors=colors)
        dist_plot = DistPlot(hoverinfo=hoverinfo, colors=colors)
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

        plot = PlotHandler(feature_col, target, specs)
        # set show fig to false, show explicitly below
        plot.build(
            df=df,
            feature_col=feature_col,
            target=target,
            topleft=roc_plot,
            topright=dist_plot,
            bottom=event_plot,
            show_fig=False,
        )

    # show the plot if show_plot set to true
    if show_plot:
        plot.show()

    return plot


def build_univariate_plots(
    df,
    features: list,
    target: str,
    feature_types: dict,
    n_bins: dict = None,
    bins: dict = None,
    save_directory: pathlib.Path() = None,
    colors: PlotColors = PlotColors(),
    show_plot: bool = False,
    hoverinfo="none",
) -> Dict[str, PlotHandler]:
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

    if n_bins is None:
        n_bins = {feature: 10 for feature in features}
    else:
        for feature in features:
            if feature not in n_bins.keys():
                n_bins[feature] = 10

    if feature_types is None:
        feature_types = {feature: "float" for feature in features}

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
        )

        if save_directory is not None:
            save_loc = fig.save_fig(save_directory)
            print(f"[{i+1}/{len(features)}] Saving univariate anaylsis for {feature}")
            save_locs.append(save_loc)

        figs[feature] = fig

    return figs, save_locs


def build_powerpoint(
    fig_locs: list,
    feature_names: list,
    save_path: pathlib.Path(),
):
    pres = Presentation()

    for i, fig_loc in enumerate(fig_locs):
        s_register = pres.slide_layouts[7]
        s = pres.slides.add_slide(s_register)

        pic = s.shapes.add_picture(
            str(pathlib.Path(fig_loc).resolve()),
            Inches(0.5),
            Inches(1.75),
            width=Inches(7),
            height=Inches(5),
        )
        pic.left = int((pres.slide_width - pic.width) / 2)

        title = s.shapes.title
        title.text = f"{i} Univariate analysis for {(feature_names)[i]}"
        title_para = s.shapes.title.text_frame.paragraphs[0]
        title_para.font.size = Pt(24)

    pres.save(save_path)
