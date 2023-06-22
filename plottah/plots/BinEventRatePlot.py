from typing import Dict, List
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .PlotProtocol import PlotProtocol
from plottah.colors import PlotColors
from plottah.utils import get_bins, get_min_max_adj, get_labels_from_bins


@dataclass
class BinEventRatePlot(PlotProtocol):
    # set the colorway
    colors: PlotColors = field(default_factory=lambda: PlotColors())

    # set (number of) bins
    bins: list = field(default_factory=lambda: None)
    n_bins: int = field(default_factory=lambda: 10)

    # set hover setting
    hoverinfo: str = field(default_factory=lambda: "skip")

    def do_math(
        self,
        df,
        feature_col,
        target_col,
        fillna: bool = False,
        method: str = "quantile",
    ):
        """
        does the required math to generate the traces, annotations and axes for the roc-curve plot

        1. imputes/removes missing values
        2. extract traces from the distplot function from plotly
        3. get the max density and feature value after imputing
        """

        # set feature and target column names
        self.feature_col = feature_col
        self.target_col = target_col

        # make fresh copy of df
        self.df = df.copy()

        # Calculate global event rate
        self.event_rate = np.mean(self.df[target_col])

        # Adjust n_bins if less unique values exist
        self.n_bins = self.n_bins if self.bins is None else len(self.bins)
        n_unique_feat_vals = df[feature_col].nunique()
        self.n_bins = np.minimum(n_unique_feat_vals, self.n_bins)

        # Get unadjusted min and max
        min_val, max_val = get_min_max_adj(self.df, feature_col)
        min_val_adj, max_val_adj = get_min_max_adj(self.df, feature_col)

        ## BINNING
        self.bins = (
            self.bins
            if self.bins is not None
            else get_bins(
                self.df,
                self.feature_col,
                self.target_col,
                min_val_adj,
                max_val_adj,
                n_bins=self.n_bins,
                method=method,
            )
        )
        # convert type to list
        self.bins = list(self.bins)

        # set first and last value back to min and max before imputing
        self.bins[0] = min_val
        self.bins[self.n_bins - 1] = max_val

        # update number of bins
        assert self.n_bins == len(self.bins)

        # Create bins (return None if binning is not successfull)
        try:
            # create new column indicating what bin record belongs to
            self.df = self.df.assign(
                bins=pd.cut(
                    x=self.df.loc[:, feature_col],
                    bins=self.bins,
                    include_lowest=True,
                    right=True,
                    labels=False,
                )
            )
        except Exception as e:
            raise ValueError("{self.feature_col} cannot be binned")

        # Create plot labels: [(4, 6), (6, 10), ...]
        self.labels = get_labels_from_bins(self.bins)

        ## CLIPPING

        # Clip values according to min/max values
        df[feature_col].clip(lower=min_val, upper=max_val, inplace=True)

        # Ensure clipping values does not remove all but a single value
        if df[feature_col].nunique() < 2:
            raise ValueError(
                "{self.feature_col} contains less than 2 features after clipping outliers!!"
            )

        ## NA's

        # Handle NAs
        if self.df["bins"].isna().sum() > 0:
            # replace the NA bin w/ n_bins - 1
            self.df.loc[:, "bins"] = self.df.loc[:, "bins"].where(
                ~self.df.loc[:, "bins"].isna(), self.n_bins - 1
            )
            self.labels.append("NA")
            self.n_bins += 1

        # Convert bins to categories
        self.df.bins = self.df.bins.astype("category")

        # Set all categories
        self.df.bins = self.df.bins.cat.set_categories(list(range(self.n_bins - 1)))

        # Group into bins and calculate required metrics
        self.df_binned = self.df.groupby("bins").agg(
            {feature_col: [len], target_col: ["mean"]}
        )

        # Rename columns
        level_one = self.df_binned.columns.get_level_values(0).astype(str)
        level_two = self.df_binned.columns.get_level_values(1).astype(str)
        column_separator = ["_" if x != "" else "" for x in level_two]
        self.df_binned.columns = level_one + column_separator + level_two

        # Set NA counts to zero
        self.df_binned[f"{feature_col}_len"] = self.df_binned[
            f"{feature_col}_len"
        ].fillna(0)

        # make fractions
        self.df_binned[f"{feature_col}_len"] = (
            self.df_binned[f"{feature_col}_len"]
            / self.df_binned[f"{feature_col}_len"].sum()
        )

    def get_traces(self) -> List[Dict]:
        return [
            # plot the first distribution:
            {
                "trace": go.Bar(
                    x=self.labels,
                    y=self.df_binned[f"{self.feature_col}_len"],
                    marker_color=self.colors.get_rgba("secondary_color", opacity=0.1),
                    marker_line_color=self.colors.get_rgba("secondary_color"),
                    marker_line_width=1.5,
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo=self.hoverinfo,
                ),
                # share y
                "secondary_y": False,
            },
            # plot binned event rate baseline
            {
                "trace": go.Scatter(
                    x=self.labels,
                    y=[self.event_rate] * len(self.labels),
                    mode="lines",
                    line=dict(
                        color=self.colors.get_grey_rgba(),
                        dash="dash",
                        width=1,
                    ),
                    hoverinfo=self.hoverinfo,
                    name=f"General Event Rate: ({'{:.1%}'.format(self.event_rate)})",
                ),
                # share y
                "secondary_y": True,
            },
            {
                "trace": go.Scatter(
                    x=self.labels,
                    y=self.df_binned[f"{self.target_col}_mean"],
                    mode="lines+markers",
                    line=dict(
                        color=self.colors.get_rgba(),
                        width=1,
                    ),
                    hoverinfo=self.hoverinfo,
                    name="Event Rate",
                ),
                # share y
                "secondary_y": True,
            },
        ]

    def get_x_axes_layout(self, row, col):
        return dict(
            title_text=f"{self.feature_col}",
            title_font={"size": 12},
            row=row,
            col=col,
            title_standoff=5,  # decrease space between title and plot
            tickangle=22.5,
        )

    def get_y_axes_layout(self, row, col):
        return dict(
            title_text=f"Fraction of Observations",
            title_font={"size": 12},
            range=[0, 1.2 * self.df_binned[f"{self.feature_col}_len"].max()],
            row=row,
            col=col,
            title_standoff=5,  # decrease space between title and plot
        )

    def get_secondary_y_axis_title(self):
        return "Event Rate"

    def get_annotations(self, xref, yref):
        return []
