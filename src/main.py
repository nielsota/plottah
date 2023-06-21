from plots import RocCurvePlot, DistPlot, BinEventRatePlot
from plot_handler import PlotHandler
from plot_builder import build_univariate_plots
from config import settings
from colors import PlotColors

import pathlib
import pandas as pd


def main():
    color_palette = PlotColors(
        primary_color=settings.primary_color,
        secondary_color=settings.secondary_color,
        tertiary_color=settings.tertiary_color,
        grey_tint_color=settings.grey_tint_color,
    )

    build_univariate_plots(
        pd.read_csv(settings.file_path),
        settings.features,
        settings.target,
        settings.output_path,
        colors=color_palette,
    )


if __name__ == "__main__":
    main()
