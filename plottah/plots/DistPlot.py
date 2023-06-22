import plotly.graph_objects as go
import plotly.figure_factory as ff
from dataclasses import dataclass, field

from .PlotProtocol import PlotProtocol
from plottah.utils import remove_or_impute_nan_infs
from plottah.colors import PlotColors


@dataclass
class DistPlot(PlotProtocol):
    # set the colorway
    colors: PlotColors = field(default_factory=lambda: PlotColors())

    # set hover setting
    hoverinfo: str = field(default_factory=lambda: "skip")

    def do_math(self, df, feature_col, target_col, fillna: bool = False):
        """
        does the required math to generate the traces, annotations and axes for the roc-curve plot

        1. imputes/removes missing values
        2. extract traces from the distplot function from plotly
        3. get the max density and feature value after imputing
        """

        # 1. impute/remove missing values
        self.df_imputed = remove_or_impute_nan_infs(df.copy(), feature_col, target_col)

        # 2. extract traces from the distplot function from plotly
        self.hist_data = [
            self.df_imputed.loc[(self.df_imputed[target_col] == 0), feature_col].values,
            self.df_imputed.loc[(self.df_imputed[target_col] == 1), feature_col].values,
        ]
        self.group_labels = ["0", "1"]
        self.distplot = ff.create_distplot(self.hist_data, self.group_labels)

        # 3. get the max density and feature value after imputing
        self.max_density = max(
            self.distplot["data"][2].y.max(), self.distplot["data"][3].y.max()
        )
        self.max_val_adj = self.df_imputed[feature_col].max()

    def get_traces(self):
        return [
            # plot the first distribution:
            {
                "trace": go.Scatter(
                    self.distplot["data"][2],
                    line=dict(color=self.colors.get_rgba(), width=0.5),
                    fill="tonexty",
                    fillcolor=self.colors.get_rgba(opacity=0.2),
                    hoverinfo=self.hoverinfo,
                ),
                # share y
                "secondary_y": False,
            },
            # plot the second distribution
            {
                "trace": go.Scatter(
                    self.distplot["data"][3],
                    line=dict(color=self.colors.get_rgba("secondary_color"), width=0.5),
                    fill="tozeroy",
                    fillcolor=self.colors.get_rgba("secondary_color", opacity=0.2),
                    hoverinfo=self.hoverinfo,
                ),
                # share y
                "secondary_y": False,
            },
        ]

    def get_x_axes_layout(self, row, col):
        return None

    def get_y_axes_layout(self, row, col):
        return dict(
            title_text="Density",
            title_font={"size": 12},
            row=row,
            col=col,
            title_standoff=5,  # decrease space between title and plot
        )

    def get_annotations(self, xref, yref):
        return [
            dict(
                x=0.9 * self.max_val_adj,
                y=1 * self.max_density,
                xref=xref,
                yref=yref,
                text=f"Class: 0",
                font=dict(color=self.colors.get_rgba()),
                showarrow=False,
                bordercolor="rgba(255,255,255,1)",
                borderwidth=2,
                borderpad=4,
                bgcolor="rgba(255,255,255,1)",
                opacity=0.8,
            ),
            dict(
                x=0.9 * self.max_val_adj,
                y=0.9 * self.max_density,
                xref=xref,
                yref=yref,
                text=f"Class: 1",
                font=dict(color=self.colors.get_rgba("secondary_color")),
                showarrow=False,
                bordercolor="rgba(255,255,255,1)",
                borderwidth=2,
                borderpad=4,
                bgcolor="rgba(255,255,255,1)",
                opacity=0.8,
            ),
        ]
