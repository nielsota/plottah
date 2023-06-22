from typing import Protocol, List
from plotly.subplots import make_subplots

import plotly.graph_objects as go


class PlotProtocol(Protocol):
    """
    Defines an interface for all the plots that can be combined in the OtaPlotter Object
    """

    def do_math(self, df, feature_col, target_col, fillna: bool = False):
        """
        does the required math to generate the traces, annotations and axes for the roc-curve plot
        """
        ...

    def get_traces(self) -> List:
        ...

    def get_x_axes_layout(self, row: int, col: int):
        ...

    def get_y_axes_layout(self, row: int, col: int):
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
