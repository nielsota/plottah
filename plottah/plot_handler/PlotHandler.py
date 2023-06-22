from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from plottah.plots import PlotProtocol
import pathlib

import pandas as pd


@dataclass
class PlotHandler:
    """
    class combining multiple PlotProtocol objects into a single figure for a single variable

    Args:
        feature_col: feature to analyze
        target_col: target column in dataframe
        specs: figure layout

    """

    feature_col: str
    target_col: str
    specs: list = field(
        default_factory=lambda: [[{}, {}], [{"colspan": 2, "secondary_y": True}, None]]
    )
    nan_count_specs: int = field(default_factory=lambda: 0)

    def __post_init__(self):
        """
        Plotly does not make it easy to find the correct axes-references for each subplot.

        If you have a standard 2x2 plot with 4 plots with a single y-axes, the xref and yref for each plot will be:
        [(x1, y1), (x2, y2)],
        [(x3, y3), (x4, y4)]

        If you have a  2x2 plot, where the bottom plot spans the whole row and has a secondary y_axis, the xref and yref
        [(x1, y1), (x2, y2)],
        [(x3, y3 & y4), (None, None)]

        If you have a  2x2 plot, where the top plot spans the whole row and has a secondary y_axis, the xref and yref
        [(x1, y1 & y2), (None, None)],
        [(x2, y3), (x3, y4)]

        This last option means that we cannot simply use the row and column number to map to xref - the 2,1 element does not always map to x3.

        This post init method uses the specs to obtain a map from the position of the subplot to what x and y ref it should use.

        """
        # check if default spec was used
        self.default_spec = self.specs == [
            [{}, {}],
            [{"colspan": 2, "secondary_y": True}, None],
        ]

        # make these in init later
        self.nrows = 2
        self.ncols = 2

        counter = 1
        self.xrefs = [
            [f"x?" for j, el in enumerate(row)] for i, row in enumerate(self.specs)
        ]

        self.yrefs = [
            [f"y{i * self.ncols + j + 1}" for j, el in enumerate(row)]
            for i, row in enumerate(self.specs)
        ]

        self.yaxes = [
            [f"yaxis{i * self.ncols + j + 1}" for j, el in enumerate(row)]
            for i, row in enumerate(self.specs)
        ]

        self.xaxes = [
            [f"xaxis{i * self.ncols + j + 1}" for j, el in enumerate(row)]
            for i, row in enumerate(self.specs)
        ]

        rows = range(self.nrows)
        cols = range(self.ncols)
        for i in rows:
            for j in cols:
                if self.specs[i][j] == None:
                    self.legend_xref = i
                else:
                    self.xrefs[i][j] = f"x{counter}"
                    # self.yrefs[i][j] = f'y{counter}'
                    counter += 1

        # make figure
        self.fig = make_subplots(
            rows=self.nrows,
            cols=self.ncols,
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=self.specs,
        )

        # update layout of figure
        self.fig.update_layout(
            showlegend=True,
            title=dict(
                text=f"{self.feature_col}: Roc Curve | Densities | Event Rates",
                font=dict(size=20),
            ),
            margin=dict(t=40),
            title_x=0.5,
            width=1000,
            height=800,
            legend=dict(
                # the below if through experimentation
                x=0.94,
                y=0.36 + (1 - self.legend_xref) * 0.57,
                xanchor="right",
                yanchor="bottom",
            ),
        )

    def build_subplot(self, subplot: PlotProtocol, row: int, col: int):
        """
        function that - given a PlotProtocol object - add the plot data to the figure

        Args:
            self: class instance containing the figure to add plots to
            subplot (PlotProtocol): plot object containing data to add to figure
            row, col (int): where on figure to add subplot

        Returns:
            None: but updates self.fig

        """
        # add traces
        for trace_dict in subplot.get_traces():
            self.fig.add_trace(
                trace_dict["trace"],
                secondary_y=trace_dict["secondary_y"],
                row=row,
                col=col,
            )

        # add annotations
        for annotation in subplot.get_annotations(
            self.xrefs[row - 1][col - 1], self.yrefs[row - 1][col - 1]
        ):
            self.fig.add_annotation(**annotation)

        # update axes layout if specifed
        if subplot.get_x_axes_layout(row, col) is not None:
            self.fig.update_xaxes(**subplot.get_x_axes_layout(row, col))

        if subplot.get_y_axes_layout(row, col) is not None:
            self.fig.update_yaxes(**subplot.get_y_axes_layout(row, col))

        # update secondary y_axis if applicable
        if subplot.get_secondary_y_axis_title() is not None:
            # name of axis contained in self.yaxes on index (row - 1, col - 1) is the primary axes, plotly will store the secondary yaxis one further; i.e. stored at index (row - 1 , col)
            secondary_yaxis = self.yaxes[row - 1][col]
            self.fig.layout[
                secondary_yaxis
            ].title.text = subplot.get_secondary_y_axis_title()

    def build(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target: str,
        topleft: PlotProtocol,
        topright: PlotProtocol,
        bottom: PlotProtocol,
        show_fig: bool = True,
    ):
        """
        function that - given a PlotProtocol object - add the plot data to the figure

        Args:
            df (pd.DataFrame): dataframe containing columns to analyze,
            feature_col (str): feature column,
            target (str): target column:
            topleft (PlotProtocol): Plot object to store in the top-left position of the plot
            topright (PlotProtocol): Plot object to store in the top-right position of the plot
            bottom (PlotProtocol): Plot object to store in the bottom row of the plot
            show_fig (bool, default = True): Whether or not to show the plot

        Returns:
            None: but updates self.fig

        """
        if not self.default_spec:
            raise ValueError(
                "build function only works for default specs, use build_subplot with each individual subplot for your specs"
            )

        # do the math for each subplot
        topleft.do_math(df, feature_col, target)
        topright.do_math(df, feature_col, target)
        bottom.do_math(df, feature_col, target)

        # build each of the subplots
        self.build_subplot(topleft, 1, 1)
        self.build_subplot(topright, 1, 2)
        self.build_subplot(bottom, 2, 1)

        if show_fig:
            self.show()

    def show(self):
        """
        show plot
        """

        self.fig.show()

    def save_fig(self, directory: pathlib.Path):
        """
        save plot
        """

        # make directory if not exists
        if not directory.exists():
            raise ValueError(f"directory: {directory} does not exist")

        # save the image in the directory
        self.fig.write_image(directory / f"univariate_plot_{self.feature_col}.jpeg")

    def get_fig(self):
        """
        Return figure object
        """
        return self.fig
