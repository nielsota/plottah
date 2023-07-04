from typing import Protocol, List
from plotly.subplots import make_subplots

import plotly.graph_objects as go


class PlotProtocol(Protocol):
    """
    Defines an interface for all the plots that can be combined in the OtaPlotter Object

    (Cohesion) Plot classes are responsible for:

        (do_math: RUN BEFORE PLOTTING)
        - Accepting a dataframe, a feature and target column and doing all the math required to generate a plot.
          If a lot of code is required here, advice is to seperate out into _function_name functions

        (get_traces)
        -  Returning a list of dictionaries, with one entry per trace to plot where the keys are
            1. trace: the trace to add
            2. secondary_y: whether the trace is on the secondary y-axes

        (get_x_axes_layout)
        -  Return a dictionary with x-axes settings

        (get_y_axes_layout)
        -  Return a dictionary with y-axes settings

        (get_annotation)
        -  Return a list of dictionaries containing settings for annotations

        (get_secondary_y_axis_title)
        - Set the secondary y-axis title (only required if secondary y-axis exists)

        (show_plot)
        - Function that creates figure and adds all elemets of Plot to figure, mainly for development/debugging
          of individual classes

    """

    def do_math(self, df, feature_col, target_col, fillna: bool = False):
        """
        does the required math to generate the traces, annotations and axes for the roc-curve plot
        """
        ...

    def get_traces(self) -> List[dict]:
        ...

    def get_x_axes_layout(self, row: int, col: int) -> dict:
        ...

    def get_y_axes_layout(self, row: int, col: int) -> dict:
        ...

    def get_annotations(self, ref: str) -> List[dict]:
        ...

    def get_secondary_y_axis_title(self):
        return None

    def show_plot(self, show_figure: bool = True) -> go.Figure:
        # create figure with single graph
        figure = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        row, col = 1, 1

        # add traces
        for trace_dict in self.get_traces():
            figure.add_trace(
                trace_dict["trace"],
                secondary_y=trace_dict["secondary_y"],
            )

        # add annotations
        for annotation in self.get_annotations("x1", "y1"):
            figure.add_annotation(**annotation)

        # update axes layout if specifed
        if self.get_x_axes_layout(row, col) is not None:
            figure.update_xaxes(**self.get_x_axes_layout(row, col))

        if self.get_y_axes_layout(row, col) is not None:
            figure.update_yaxes(**self.get_y_axes_layout(row, col))

        # update secondary y_axis if applicable
        if self.get_secondary_y_axis_title() is not None:
            print(self.get_secondary_y_axis_title())
            figure.layout["yaxis2"].title.text = self.get_secondary_y_axis_title()

        if show_figure:
            figure.show()

        return figure
