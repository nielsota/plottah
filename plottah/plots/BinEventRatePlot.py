from typing import Dict, List
from dataclasses import dataclass, field
import logging

import plotly.graph_objects as go
import numpy as np
import pandas as pd

from plottah.plots.PlotProtocol import PlotProtocol
from plottah.colors import PlotColors
from plottah.utils import get_bins, get_min_max_adj, get_labels_from_bins

MIN_N_UNIQUE = 25


@dataclass
class CategoricalBinner:
    """
    seperated out binning to increase cohesion -> only responsible for
        1. Adding bins to dataframe given as input
        2. Returning labels based on bins

    """

    _labels: list = field(default_factory=lambda: None)

    def get_labels(self):
        return self._labels.copy()

    def add_bins(self, df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
        # Extract unique values
        unique_vals = df[feature_col].sort_values().unique()

        # Convert sorted elements to int
        # mapping = pd.factorize(unique_vals, na_sentinel=len(unique_vals))
        mapping = pd.factorize(unique_vals, use_na_sentinel=False)

        # Create mapping
        mapping_values = list(mapping[0])
        mapping_keys = list(mapping[1])

        # Create mapping dictionary
        mapping_dict = dict(zip(mapping_keys, mapping_values))

        # Apply mapping to column
        df = df.assign(bins=df[feature_col].map(mapping_dict))

        # Set labels for plotting
        self._labels = [str(key) for key in mapping_keys]
        if df["bins"].isna().sum() > 0:
            self._labels.append("NA")

        return df, self._labels


@dataclass
class StandardBinner:
    """
    seperated out binning to increase cohesion -> only responsible for
        1. Adding bins to dataframe given as input
        2. Returning labels based on bins

    """

    _labels: list = field(default_factory=lambda: None)

    def get_labels(self):
        return self._labels.copy()

    def add_bins(
        self,
        df: pd.DataFrame,
        feature_col: str,
        n_bins: int,
        bins=None,
        method="quantile",
    ) -> pd.DataFrame:
        # Get unadjusted min and max
        min_val, max_val = get_min_max_adj(df, feature_col)
        min_val_adj, max_val_adj = get_min_max_adj(df, feature_col)

        # boolean checking if bins were provided
        provided_bins = bins is not None

        ## BINNING
        bins = (
            bins
            if provided_bins
            else get_bins(
                df,
                feature_col,
                min_val_adj,
                max_val_adj,
                n_bins=n_bins,
                method=method,
            )
        )

        # convert type to list
        bins = list(bins)

        # set first and last value back to min and max before imputing - could lead to duplicate bins if min/max already a bin so remove duplicates
        if (bins[0] != min_val) & provided_bins:
            logging.warning(
                f"For feature {feature_col} you provided custom bins, the smallest bin provided is {bins[0]} but minimum is {min_val}, so replacing {bins[0]} by {min_val}"
            )
            bins[0] = min_val
            seen = set()
            bins = [x for x in bins if x not in seen and (seen.add(x) or True)]

        if (bins[n_bins - 1] != max_val) & provided_bins:
            logging.warning(
                f"For feature {feature_col} you provided custom bins, the largest bin is {bins[n_bins - 1] } but max is {max_val}, so replacing {bins[n_bins - 1]} by {max_val}"
            )
            bins[n_bins - 1] = max_val
            seen = set()
            bins = [x for x in bins if x not in seen and (seen.add(x) or True)]

        # error if not all bins are smaller than the last bin after replacing last bin by max
        if (not all(i < bins[n_bins - 1] for i in bins[:-1])) & provided_bins:
            logging.warning(
                f"For feature {feature_col} you provided custom bins, the largest bin is {bins[n_bins - 1]} but other bins provided by you are larger, and since bins must increase monotonically, this will lead to errors. Please revise the bins.Please revise the bins. Currently (after replacing smallest and largest bin by min/max): {bins}"
            )

        # error if not all bins are larger than the first bin after replacing first bin by min
        if (not all(i > bins[0] for i in bins[1:])) & provided_bins:
            logging.warning(
                f"For feature {feature_col} you provided custom bins, the smallest bin is {bins[0]} but other bins provided by you are smaller, and since bins must increase monotonically, this will lead to errors. Please revise the bins. Currently (after replacing smallest and largest bin by min/max): {bins}"
            )
        logging.info(f"using bins: {bins}")

        # update number of bins

        # Create bins (return None if binning is not successfull)
        try:
            # create new column indicating what bin record belongs to
            df = df.assign(
                bins=pd.cut(
                    x=df.loc[:, feature_col],
                    bins=bins,
                    include_lowest=True,
                    right=True,
                    labels=False,
                )
            )
        except Exception as e:
            raise ValueError(f"{self.feature_col} cannot be binned using bins: {bins}")

        # Create plot labels: [(4, 6), (6, 10), ...]
        self._labels = get_labels_from_bins(bins)
        logging.info(f"using labels: {self._labels}")

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
        if df["bins"].isna().sum() > 0:
            # replace the NA bin w/ n_bins - 1
            df.loc[:, "bins"] = df.loc[:, "bins"].where(
                ~df.loc[:, "bins"].isna(), n_bins - 1
            )
            self._labels.append("NA")
            n_bins += 1

        # Convert bins to categories
        df.bins = df.bins.astype("category")

        # Set all categories
        df.bins = df.bins.cat.set_categories(list(range(n_bins - 1)))

        return df, self._labels


@dataclass
class BinEventRatePlot(PlotProtocol):
    # set the colorway
    colors: PlotColors = field(default_factory=lambda: PlotColors())

    # set (number of) bins
    bins: list = field(default_factory=lambda: None)
    n_bins: int = field(default_factory=lambda: 10)

    # set hover setting
    hoverinfo: str = field(default_factory=lambda: "skip")

    # set default feature type
    feature_type: str = field(default_factory=lambda: "float")

    def do_math(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        method: str = "quantile",
    ):
        """
        does the required math to generate the traces, annotations and axes for the roc-curve plot

        1. imputes/removes missing values
        2. extract traces from the distplot function from plotly
        3. get the max density and feature value after imputing
        """

        logging.info(f"Started math for BinEventRatePlot for {feature_col}")

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
        if n_unique_feat_vals < self.n_bins:
            logging.warning(
                f"{self.feature_col} only has {n_unique_feat_vals} distinct values, decreasing n_bins from {self.n_bins} to {n_unique_feat_vals} "
            )
        self.n_bins = np.minimum(n_unique_feat_vals, self.n_bins)

        # add bins column using strategy depending on feature type
        if self.feature_type == "categorical":
            logging.info("using categorical binner")
            if self.df[self.feature_col].nunique() > self.n_bins:
                raise ValueError(
                    f"Too many unique values for feature: {self.feature_col} ({self.df[self.feature_col].nunique()}) to only use {self.n_bins} bins. Increase n_bins to at least {self.df[self.feature_col].nunique()}!"
                )
            binner = CategoricalBinner()
            self.df, self.labels = binner.add_bins(self.df, self.feature_col)
        else:
            logging.info("Using standard binner")
            if self.df[self.feature_col].nunique() < MIN_N_UNIQUE:
                logging.warning(
                    f"{self.feature_col} only has {self.df[self.feature_col].nunique()} distinct values, consider switching feature type for {self.feature_col} to categorical (currenly {self.feature_type})"
                )
            binner = StandardBinner()
            self.df, self.labels = binner.add_bins(
                self.df, self.feature_col, self.n_bins, self.bins, method=method
            )

        # Group into bins and calculate required metrics - observed = False doesn't impact anything but needed for warning
        self.df_binned = self.df.groupby("bins", observed=False).agg(
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
            # plot bar chart
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
            # plot binned event rate
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
            title_text="Fraction of Observations",
            title_font={"size": 12},
            # range=[0, 1.2 * self.df_binned[f"{self.feature_col}_len"].max()],
            row=row,
            col=col,
            title_standoff=5,  # decrease space between title and plot
        )

    def get_secondary_y_axis_title(self):
        return "Event Rate"

    def get_annotations(self, xref, yref):
        return []
