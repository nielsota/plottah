from dataclasses import dataclass, field
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from plottah.colors import PlotColors
from plottah.plots.plot_protocol import PlotProtocol
from plottah.utils import (
    generate_bins,
    get_labels_from_bins,
    get_min_max_adj,
    validate_binary_target,
)

MIN_N_UNIQUE = 25
NUMERICAL_BINS = list[int, float]


@dataclass
class CategoricalBinner:
    """
    Separated out binning to increase cohesion -> only responsible for
        1. Adding bins to dataframe given as input
        2. Returning labels based on bins
    """

    _labels: list = field(default_factory=lambda: None)

    def get_labels(self):
        return self._labels.copy()

    def _create_value_mapping(self, unique_vals) -> tuple[dict, list]:
        """Create mapping dictionary from unique values to integer bins."""
        logger.debug(f"Creating value mapping for unique values: {unique_vals}")
        mapping = pd.factorize(unique_vals, use_na_sentinel=False)
        mapping_values = list(mapping[0])
        mapping_keys = list(mapping[1])
        mapping_dict = dict(zip(mapping_keys, mapping_values))
        return mapping_dict, mapping_keys

    def _assign_bins(
        self, df: pd.DataFrame, feature_col: str, mapping_dict: dict
    ) -> pd.DataFrame:
        """Assign bins to dataframe using mapping dictionary."""
        logger.debug("Assigning bins to dataframe using mapping")
        return df.assign(bins=df[feature_col].map(mapping_dict))

    def _set_labels(self, mapping_keys: list, df: pd.DataFrame):
        """Set labels for plotting."""
        logger.debug("Setting plot labels")
        self._labels = [str(key) for key in mapping_keys]
        if df["bins"].isna().sum() > 0:
            self._labels.append("NA")

    def add_bins(self, df: pd.DataFrame, feature_col: str) -> tuple[pd.DataFrame, list]:
        """Add bins to dataframe for categorical features."""
        # Extract unique values
        unique_vals = df[feature_col].sort_values().unique()

        # Create mapping from values to integers
        mapping_dict, mapping_keys = self._create_value_mapping(unique_vals)

        # Apply mapping to create bins
        df = self._assign_bins(df, feature_col, mapping_dict)

        # Set labels for plotting
        self._set_labels(mapping_keys, df)

        return df, self._labels


@dataclass
class StandardBinner:
    """
    seperated out binning to increase cohesion -> only responsible for
        1. Adding bins to dataframe given as input
        2. Returning labels based on bins
    """

    _labels: list = field(default_factory=lambda: None)

    def _remove_duplicate_bins_preserve_order(
        self, bins: NUMERICAL_BINS
    ) -> NUMERICAL_BINS:
        """Remove duplicates from bins while preserving the original order."""
        logger.debug(f"Removing duplicate bins while preserving order from: {bins}")
        unique_items = []
        seen = set()

        for item in bins:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)

        return NUMERICAL_BINS(unique_items)

    def _validate_provided_bins(
        self,
        bins: NUMERICAL_BINS,
        min_val: float | int,
        n_bins: int,
        max_val: float | int,
        feature_col: str,
    ):
        """Validate provided bins."""
        logger.debug(f"Validating provided bins for feature {feature_col}: {bins}")

        # check if bins are provided and are a list
        if bins is not None:
            if not isinstance(bins, list):
                raise ValueError("bins must be a list")
            if not all(isinstance(bin, (int, float)) for bin in bins):
                raise ValueError(
                    "bins must contain only int or float for numerical features"
                )

        # get lowest and highest bin
        lowest_bin = bins[0]
        highest_bin = bins[n_bins - 1]

        # set first and last value back to min and max before imputing - could lead to duplicate bins if min/max already a bin so remove duplicates
        if min_val < lowest_bin:
            logger.warning(
                f"For feature {feature_col} you provided custom bins, "
                f"the smallest bin provided is {lowest_bin} but minimum is {min_val}, "
                f"so replacing {lowest_bin} by {min_val}"
            )
            bins[0] = min_val
            bins = self._remove_duplicate_bins_preserve_order(bins)

        if max_val > highest_bin:
            logger.warning(
                f"For feature {feature_col} you provided custom bins, "
                f"the largest bin is {bins[n_bins - 1]} but max is {max_val}, "
                f"so replacing {bins[n_bins - 1]} by {max_val}"
            )
            bins[n_bins - 1] = max_val
            bins = self._remove_duplicate_bins_preserve_order(bins)

        # error if not all bins are smaller than the last bin after replacing last bin by max
        if not all(i < bins[n_bins - 1] for i in bins[:-1]):
            raise ValueError(
                f"For feature {feature_col} you provided custom bins, "
                f"the largest bin is {bins[n_bins - 1]} but other bins provided by you are larger, "
                f"and since bins must increase monotonically, this will lead to errors. "
                f"Please revise the bins. Currently (after replacing smallest and largest bin by min/max): {bins}"
            )

        # error if not all bins are larger than the first bin after replacing first bin by min
        if not all(i > bins[0] for i in bins[1:]):
            raise ValueError(
                f"For feature {feature_col} you provided custom bins, "
                f"the smallest bin is {bins[0]} but other bins provided by you are smaller, "
                f"and since bins must increase monotonically, this will lead to errors. "
                f"Please revise the bins. Currently (after replacing smallest and largest bin by min/max): {bins}"
            )
        logger.info(f"using bins: {bins}")

    def _assign_bins(
        self,
        df: pd.DataFrame,
        bins: NUMERICAL_BINS,
        feature_col: str,
    ):
        """Assign bins to dataframe."""
        logger.debug(f"Assigning bins to dataframe: {bins}")
        # Create bins (return None if binning is not successfull)
        try:
            # create new column indicating what bin record belongs to
            df = df.assign(
                bins=pd.cut(
                    x=df.loc[:, feature_col],
                    bins=bins,
                    include_lowest=True,
                    # ensures that the rightmost bin is closed i.e. [1, 2, 3, 4] -> (1,2], (2,3], (3,4]
                    right=True,
                    labels=False,
                )
            )
        except Exception as e:
            raise ValueError(
                f"{feature_col} cannot be binned using bins: {bins}. Error: {str(e)}"
            )
        return df

    def _clip_values(
        self, df: pd.DataFrame, feature_col: str, min_val: float, max_val: float
    ):
        """Clips values in feature_col between min_val and max_val."""
        logger.debug(
            f"Clipping values in {feature_col} between {min_val} and {max_val}"
        )
        df[feature_col] = df[feature_col].clip(lower=min_val, upper=max_val)
        # Ensure clipping values does not remove all but a single value
        if df[feature_col].nunique() < 2:
            raise ValueError(
                "{self.feature_col} contains less than 2 features after clipping outliers!!"
            )
        return df

    def _handle_nas(self, df: pd.DataFrame, n_bins: int):
        """Handle NAs by adding a new bin with the NA label"""
        logger.debug(f"Handling NA values in bins column. Current n_bins: {n_bins}")
        if df["bins"].isna().sum() > 0:
            # Add a new bin for NA values at n_bins
            df.loc[:, "bins"] = df.loc[:, "bins"].where(
                ~df.loc[:, "bins"].isna(), n_bins
            )
            self._labels.append("NA")
            n_bins += 1
        return df, n_bins

    def _set_bins_to_categories(self, df: pd.DataFrame, n_bins: int):
        """Set bins to categories."""
        logger.debug(f"Converting bins to categorical with {n_bins-1} categories")
        df.bins = df.bins.astype("category")
        df.bins = df.bins.cat.set_categories(list(range(n_bins - 1)))
        return df

    def get_labels(self):
        return self._labels.copy()

    def add_bins(
        self,
        df: pd.DataFrame,
        feature_col: str,
        n_bins: int,
        bins: NUMERICAL_BINS | None = None,
        method: Literal["quantile", "linear"] = "quantile",
    ) -> pd.DataFrame:
        """Add bins to dataframe.

        Examples:
            >>> df = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
            >>> plot = BinEventRatePlot()
            >>> df = plot.add_bins(df, 'feature', n_bins=3)
            >>> df['bins'].tolist()
            [0, 0, 1, 2, 2]

            >>> df = pd.DataFrame({'values': [1, 5, 10, 15, 20]})
            >>> plot = BinEventRatePlot()
            >>> df = plot.add_bins(df, 'values', n_bins=2, bins=[0, 10, 20])
            >>> df['bins'].tolist()
            [0, 0, 0, 1, 1]
        """
        # Get unadjusted min and max
        min_val, max_val = get_min_max_adj(df, feature_col)
        min_val_adj, max_val_adj = get_min_max_adj(df, feature_col)

        # Validate bins if they were provided and generate bins if not
        if bins is not None:
            self._validate_provided_bins(bins, min_val, n_bins, max_val, feature_col)
        else:
            # TODO: get bins should be on this class....
            bins = generate_bins(
                df,
                feature_col,
                min_val_adj,
                max_val_adj,
                n_bins=n_bins,
                method=method,
            )

        # Create bins (return None if binning is not successfull)
        df = self._assign_bins(df, bins, feature_col)

        # Create plot labels: [(4, 6], (6, 10], ...]
        self._labels = get_labels_from_bins(bins)

        # Clip values according to min/max values
        df = self._clip_values(df, feature_col, min_val, max_val)

        # Handle NAs by adding a new bin
        df, n_bins = self._handle_nas(df, n_bins)

        # Convert bins to categories
        df = self._set_bins_to_categories(df, n_bins)

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

        logger.info(f"Started math for BinEventRatePlot for {feature_col}")

        # set feature and target column names
        self.feature_col = feature_col
        self.target_col = target_col

        # validate the target column is binary
        validate_binary_target(self.df[self.target_col])

        # make fresh copy of df
        self.df = df.copy()

        # Calculate global event rate
        self.event_rate = np.mean(self.df[target_col])

        # Adjust n_bins if less unique values exist
        self.n_bins = self.n_bins if self.bins is None else len(self.bins)
        n_unique_feat_vals = df[feature_col].nunique()
        if n_unique_feat_vals < self.n_bins:
            logger.warning(
                f"{self.feature_col} only has {n_unique_feat_vals} distinct values, decreasing n_bins from {self.n_bins} to {n_unique_feat_vals} "
            )
        self.n_bins = np.minimum(n_unique_feat_vals, self.n_bins)

        # add bins column using strategy depending on feature type
        if self.feature_type == "categorical":
            logger.info("using categorical binner")
            if self.df[self.feature_col].nunique() > self.n_bins:
                raise ValueError(
                    f"Too many unique values for feature: {self.feature_col} ({self.df[self.feature_col].nunique()}) to only use {self.n_bins} bins. Increase n_bins to at least {self.df[self.feature_col].nunique()}!"
                )
            binner = CategoricalBinner()
            self.df, self.labels = binner.add_bins(self.df, self.feature_col)
        else:
            logger.info("Using standard binner")
            if self.df[self.feature_col].nunique() < MIN_N_UNIQUE:
                logger.warning(
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
