import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass, field
from sklearn import metrics

from plottah.plots.PlotProtocol import PlotProtocol
from plottah.utils import remove_or_impute_nan_infs
from plottah.colors import PlotColors


@dataclass
class RocCurvePlot(PlotProtocol):
    # set the colorway
    colors: PlotColors = field(default_factory=lambda: PlotColors())

    # set hover setting
    hoverinfo: str = field(default_factory=lambda: "skip")

    def do_math(self, df, feature_col, target_col, fillna: bool = False):
        """
        does the required math to generate the traces, annotations and axes for the roc-curve plot

        1. imputes/removes missing values
        2. calculated fpr, tpr and AUC
        """

        # 1. impute/remove missing values
        self.df_imputed = remove_or_impute_nan_infs(df.copy(), feature_col, target_col)

        # 2. calculate fpr, tpr and AUC
        self.fpr, self.tpr, _ = metrics.roc_curve(
            self.df_imputed[target_col], -self.df_imputed[feature_col], pos_label=1
        )
        self.auc = metrics.auc(self.fpr, self.tpr)

        if self.auc < 0.5:
            self.fpr, self.tpr, _ = metrics.roc_curve(
                self.df_imputed[target_col], self.df_imputed[feature_col], pos_label=1
            )
            self.auc = metrics.auc(self.fpr, self.tpr)

    def get_traces(self):
        return [
            # plot the roc-curve
            {
                "trace": go.Scatter(
                    x=self.fpr,
                    y=self.tpr,
                    mode="lines",
                    line=dict(
                        color=self.colors.get_rgba(),
                        width=1.5,
                    ),
                    hoverinfo=self.hoverinfo,
                    showlegend=False,
                ),
                # share y
                "secondary_y": False,
            },
            # plot the baseline
            {
                "trace": go.Scatter(
                    x=np.linspace(0, 1, 10),
                    y=np.linspace(0, 1, 10),
                    mode="lines",
                    line=dict(
                        color=self.colors.get_grey_rgba(),
                        dash="dash",
                        width=0.5,
                    ),
                    hoverinfo=self.hoverinfo,
                    showlegend=False,
                ),
                # share y
                "secondary_y": False,
            },
        ]

    def get_x_axes_layout(self, row, col):
        return dict(
            title_text="Cumulated negatives",
            title_font={"size": 12},
            range=[0, 1],
            row=row,
            col=col,
            title_standoff=5,  # decrease space between title and plot
        )

    def get_y_axes_layout(self, row, col):
        return dict(
            title_text="Cumulated positives",
            title_font={"size": 12},
            range=[0, 1.05],
            row=row,
            col=col,
            title_standoff=5,  # decrease space between title and plot
        )

    def get_annotations(self, xref, yref):
        return [
            dict(
                x=0.65,
                y=0.1,
                xref=xref,
                yref=yref,
                text=f"area = {self.auc:.3f}",
                showarrow=False,
                bordercolor="rgba(255,255,255,1)",
                borderwidth=2,
                borderpad=4,
                bgcolor="rgba(255,255,255,1)",
                opacity=0.8,
            )
        ]
