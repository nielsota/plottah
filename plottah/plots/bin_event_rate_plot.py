from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from loguru import logger  # type: ignore

from plottah.colors import PlotColors
from plottah.plots.plot_protocol import PlotProtocol
from plottah.utils import (
    generate_bins,
    get_labels_from_bins,
    get_min_max_adj,
    validate_binary_target,
    validate_feature_column_presence,
)

MIN_N_UNIQUE = 3
NUMERICAL_BINS = list[int | float]


# TODO: binning should be in a seperate file - gotten too large
@dataclass
class CategoricalBinner:
    """
    Separated out binning to increase cohesion -> only responsible for
        1. Adding bins to dataframe given as input
        2. Returning labels based on bins
    """

    _labels: list[str] | None = field(default_factory=lambda: None)

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

    def add_bins(
        self,
        df: pd.DataFrame,
        feature_col: str,
        **kwargs,  # Accept any additional arguments
    ) -> tuple[pd.DataFrame, list]:
        """Add bins to dataframe for categorical features.

        Args:
            df: Input dataframe
            feature_col: Column to bin
            **kwargs: Additional arguments for interface compatibility with StandardBinner
        """
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

    _labels: list[str | float | int] | None = field(default_factory=lambda: None)

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

        # set first and last value back to min and max before imputing - could lead to duplicate
        # bins if min/max already a bin so remove duplicates
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
    ) -> tuple[pd.DataFrame, list[str | float | int]]:
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


def get_binner(
    feature_type: Literal["categorical", "float"]
) -> CategoricalBinner | StandardBinner:
    logger.debug(f"Getting binner for feature type: {feature_type}")
    if feature_type == "categorical":
        logger.debug("Using CategoricalBinner")
        return CategoricalBinner()
    else:
        logger.debug("Using StandardBinner")
        return StandardBinner()


def _verify_categorical_binner_eligibility(
    df: pd.DataFrame, feature_col: str, n_bins: int
) -> bool:
    """Verify if the CategoricalBinner is eligible for the given dataframe and feature column."""
    if df[feature_col].nunique() > n_bins:
        raise ValueError(
            f"Too many unique values for feature: {feature_col} "
            f"({df[feature_col].nunique()}) to only use {n_bins} bins. "
            f"Increase n_bins to at least {df[feature_col].nunique()}!"
        )
    return True


def _verify_standard_binner_eligibility(
    df: pd.DataFrame, feature_col: str, n_bins: int
) -> bool:
    """Verify if the StandardBinner is eligible for the given dataframe and feature column."""
    if df[feature_col].nunique() < MIN_N_UNIQUE:
        logger.warning(
            f"{feature_col} only has {df[feature_col].nunique()} distinct values, "
            f"consider switching feature type for {feature_col} "
            f"to categorical)"
        )
    return True


def get_binner_verifier(
    feature_type: Literal["categorical", "float"],
) -> Callable:
    logger.debug(f"Verifying binner eligibility for feature type: {feature_type}")
    if feature_type == "categorical":
        return _verify_categorical_binner_eligibility
    else:
        return _verify_standard_binner_eligibility


@dataclass
class BinEventRatePlot(PlotProtocol):
    # set the colorway
    colors: PlotColors = field(default_factory=lambda: PlotColors())

    # set (number of) bins
    bins: list[int | float | str] | None = field(default_factory=lambda: None)
    n_bins: int = field(default_factory=lambda: 10)

    # set hover setting
    hoverinfo: str = field(default_factory=lambda: "skip")

    # set default feature type
    feature_type: Literal["categorical", "float"] = field(
        default_factory=lambda: "float"
    )

    # set default titles; can hide if not needed
    show_legend: bool = field(default_factory=lambda: True)
    tick_font_size: int = field(default_factory=lambda: 10)
    title_font_size: int = field(default_factory=lambda: 12)
    x_title: str | None = field(default_factory=lambda: None)
    y_title: str | None = field(default_factory=lambda: None)
    secondary_y_title: str | None = field(default_factory=lambda: None)
    title_standoff: int = field(default_factory=lambda: 5)

    # max bar height and event rate height
    max_bar_height: float = field(default_factory=lambda: np.nan)
    max_event_rate_height: float = field(default_factory=lambda: np.nan)
    fillna_event_rate: bool = field(
        default_factory=lambda: True
    )  # used if no samples in a bin; i.e., divide by 0

    def __post_init__(self):
        # Set default y-axis titles if not provided
        if self.y_title is None:
            self.y_title = "Fraction of Observations"

        if self.secondary_y_title is None:
            self.secondary_y_title = "Event Rate"

    def _adjust_n_bins(self, df: pd.DataFrame, feature_col: str, n_bins: int) -> int:
        """Adjust n_bins if less unique values exist"""
        n_unique_feat_vals = df[feature_col].nunique()
        if n_unique_feat_vals < n_bins:
            logger.warning(
                f"{feature_col} only has {n_unique_feat_vals} distinct values, "
                f"decreasing n_bins from {n_bins} to {n_unique_feat_vals}"
            )
        n_bins = np.minimum(n_unique_feat_vals, n_bins)
        return n_bins

    def _get_event_rate_and_size_per_bin_df(
        self, df: pd.DataFrame, feature_col: str, target_col: str, fillna: bool = True
    ) -> pd.DataFrame:
        """Calculate event rate and population size for each bin.

        For each bin, computes:
        1. Population size (count of records)
        2. Event rate (mean of target variable)

        Args:
            df: DataFrame containing the binned data
            feature_col: Name of feature column used for binning
            target_col: Name of target/label column (must be binary 0/1)

        Returns:
            DataFrame with multi-level columns containing bin statistics:
                - {feature_col}_len: Count of records in bin
                - {target_col}_mean: Mean of target variable (event rate) in bin
        """
        # Group by bins and calculate metrics
        event_rate_and_size_per_bin_df = df.groupby("bins", observed=False).agg(
            {feature_col: [len], target_col: ["mean"]}
        )

        if fillna:
            event_rate_and_size_per_bin_df = event_rate_and_size_per_bin_df.fillna(0)

        # Convert multi-level column names to single level
        # e.g. (feature, len) -> feature_len
        level_one = event_rate_and_size_per_bin_df.columns.get_level_values(0).astype(
            str
        )
        level_two = event_rate_and_size_per_bin_df.columns.get_level_values(1).astype(
            str
        )

        # Add separator only between non-empty strings
        column_separator = ["_" if x != "" else "" for x in level_two]

        # Combine column levels into single names
        event_rate_and_size_per_bin_df.columns = (
            level_one + column_separator + level_two
        )

        # Set NA counts to zero
        event_rate_and_size_per_bin_df[f"{feature_col}_len"] = (
            event_rate_and_size_per_bin_df[f"{feature_col}_len"].fillna(0)
        )

        # make fractions
        event_rate_and_size_per_bin_df[f"{feature_col}_frac"] = (
            event_rate_and_size_per_bin_df[f"{feature_col}_len"]
            / event_rate_and_size_per_bin_df[f"{feature_col}_len"].sum()
        )

        # set max bar height and event rate height
        self.max_bar_height = event_rate_and_size_per_bin_df[
            f"{feature_col}_frac"
        ].max()
        self.max_event_rate_height = event_rate_and_size_per_bin_df[
            f"{target_col}_mean"
        ].max()

        return event_rate_and_size_per_bin_df

    def _get_bar_trace(self) -> Dict:
        return {
            "trace": go.Bar(
                x=self.labels,
                y=self.event_rate_and_size_per_bin_df[f"{self.feature_col}_frac"],
                marker_color=self.colors.get_rgba("secondary_color", opacity=0.1),
                marker_line_color=self.colors.get_rgba("secondary_color"),
                marker_line_width=1.5,
                opacity=0.6,
                showlegend=False,
                hoverinfo=self.hoverinfo,
            ),
            # share y
            "secondary_y": False,
        }

    def _get_event_rate_trace(self) -> Dict:
        return {
            "trace": go.Scatter(
                x=self.labels,
                y=self.event_rate_and_size_per_bin_df[f"{self.target_col}_mean"],
                mode="lines+markers",
                line=dict(
                    color=self.colors.get_rgba(),
                    width=1,
                ),
                hoverinfo=self.hoverinfo,
                name="Event Rate",
                showlegend=self.show_legend,
            ),
            # share y
            "secondary_y": True,
        }

    def _get_general_event_rate_trace(self) -> Dict:
        return {
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
                showlegend=self.show_legend,
            ),
            # share y
            "secondary_y": True,
        }

    def do_math(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        method: Literal["quantile", "linear"] = "quantile",
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
        self.x_title = self.x_title or feature_col

        # validate the target column is binary
        validate_binary_target(df[target_col])

        # validate the feature column is present in the dataframe
        validate_feature_column_presence(df, feature_col)

        # make fresh copy of df
        self.df = df.copy()

        # Calculate global event rate
        self.event_rate = np.mean(self.df[target_col])

        # Adjust n_bins if less unique values exist
        self.n_bins = self.n_bins if self.bins is None else len(self.bins)
        self.n_bins = self._adjust_n_bins(self.df, self.feature_col, self.n_bins)

        # get binner and verifier
        binner, binner_verifier = get_binner(self.feature_type), get_binner_verifier(
            self.feature_type
        )

        # verify binner eligibility
        binner_verifier(self.df, self.feature_col, self.n_bins)

        # add bins
        self.df, self.labels = binner.add_bins(
            df=self.df,
            feature_col=self.feature_col,
            n_bins=self.n_bins,
            bins=self.bins,
            method=method,
        )

        # Calculate event rate and size for each bin. This will be used to plot the bar chart and event rate line
        self.event_rate_and_size_per_bin_df = self._get_event_rate_and_size_per_bin_df(
            df=self.df,
            feature_col=self.feature_col,
            target_col=self.target_col,
            fillna=self.fillna_event_rate,
        )

    def get_traces(self) -> List[Dict]:
        return [
            # plot bar chart or event rate, depending on primary_y
            self._get_bar_trace(),
            # plot binned event rate baseline
            self._get_general_event_rate_trace(),
            # plot binned event rate or bar chart, depending on primary_y
            self._get_event_rate_trace(),
        ]

    def get_x_axes_layout(self, row, col):
        return dict(
            title_text=self.x_title,
            title_font={"size": self.title_font_size},
            tickfont={"size": self.tick_font_size},
            row=row,
            col=col,
            title_standoff=5,  # decrease space between title and plot
            tickangle=22.5,
        )

    def get_y_axes_layout(self, row, col):
        return dict(
            title_text=self.y_title,
            title_font={"size": self.title_font_size},
            tickfont={"size": self.tick_font_size},
            row=row,
            col=col,
            title_standoff=5,  # decrease space between title and plot
        )

    def get_secondary_y_axis_title(self):
        return self.secondary_y_title

    def get_annotations(self, xref, yref):
        return []


if __name__ == "__main__":
    """Example usage of BinEventRatePlot with both StandardBinner and CategoricalBinner."""

    from plotly.subplots import make_subplots

    # Create sample data
    df = pd.DataFrame(
        {
            "numerical_feature": np.random.normal(0, 1, 1000),
            "categorical_feature": np.random.choice(["A", "B", "C"], 1000),
            "target": np.random.choice([0, 1], 1000),
        }
    )

    # Create empty plotly figure using subplots, 2x2
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"secondary_y": True}] * 2, [{"secondary_y": True}] * 2],
        horizontal_spacing=0.1,  # Adjust space between columns (default is 0.2)
        vertical_spacing=0.15,
    )

    # create the plots
    tl = BinEventRatePlot(
        n_bins=10,
        feature_type="float",
        tick_font_size=12,
        x_title="Numerical Feature",
        y_title="Fraction of Observations",
        secondary_y_title="Event Rate",
    )
    tr = BinEventRatePlot(
        bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        show_legend=False,
        tick_font_size=14,
        x_title="Numerical Feature but custom bins",
        y_title="Fraction of Observations",
        secondary_y_title="Event Rate",
    )
    bl = BinEventRatePlot(
        n_bins=3,
        feature_type="categorical",
        show_legend=False,
        tick_font_size=10,
        x_title="Categorical Feature, small font",
        y_title="Fraction of Observations",
        secondary_y_title="Does this work?",
    )
    br = BinEventRatePlot(
        bins=["A", "B", "C"],
        feature_type="categorical",
        show_legend=False,
        tick_font_size=10,
        x_title="Categorical Feature, custom bins",
        y_title="Fraction of Observations",
        secondary_y_title="Event Rate",
    )

    # do the math
    tl.do_math(
        df=df,
        feature_col="numerical_feature",
        target_col="target",
    )
    tr.do_math(
        df=df,
        feature_col="numerical_feature",
        target_col="target",
    )
    bl.do_math(
        df=df,
        feature_col="categorical_feature",
        target_col="target",
    )
    br.do_math(
        df=df,
        feature_col="categorical_feature",
        target_col="target",
    )

    # max bar height
    max_bar_height = (
        np.ceil(
            max(
                [
                    tl.max_bar_height,
                    tr.max_bar_height,
                    bl.max_bar_height,
                    br.max_bar_height,
                ]
            )
            * 11
        )
        / 10
    )
    max_event_rate_height = (
        np.ceil(
            np.max(
                [
                    tl.max_event_rate_height,
                    tr.max_event_rate_height,
                    bl.max_event_rate_height,
                    br.max_event_rate_height,
                ]
            )
            * 11
        )
        / 10
    )

    # Replace the dictionary with a list of tuples
    plot_positions = [
        (tl, 1, 1, "yaxis2"),
        (tr, 1, 2, "yaxis4"),
        (bl, 2, 1, "yaxis6"),
        (br, 2, 2, "yaxis8"),
    ]

    # Update the loop to use the list of tuples
    for plot, row, col, secondary_y in plot_positions:
        for trace_dict in plot.get_traces():
            fig.add_trace(
                trace_dict["trace"],
                secondary_y=trace_dict["secondary_y"],
                row=row,
                col=col,
            )

            # update axes layout if specifed
            if plot.get_x_axes_layout(row, col) is not None:
                fig.update_xaxes(**plot.get_x_axes_layout(row, col))

            if plot.get_y_axes_layout(row, col) is not None:
                fig.update_yaxes(**plot.get_y_axes_layout(row, col))

            # Set consistent y-axis ranges for both primary and secondary y-axes
            fig.update_yaxes(
                range=[0, max_bar_height], row=row, col=col, secondary_y=False
            )  # For fraction of observations
            fig.update_yaxes(
                range=[0, max_event_rate_height], row=row, col=col, secondary_y=True
            )  # For event rate

            # set title for secondary y-axis
            fig.layout[secondary_y].title.text = "Event Rate"

    # show the figure
    fig.show()


def _get_max_bar_height(plots: list[BinEventRatePlot]) -> float:
    max_bar_height = np.nanmax([plot.max_bar_height for plot in plots])
    logger.debug(f"max_bar_height: {max_bar_height}")
    return np.ceil(max_bar_height * 11) / 10


def _get_max_event_rate_height(plots: list[BinEventRatePlot]) -> float:
    max_event_rate_height = np.nanmax([plot.max_event_rate_height for plot in plots])
    logger.debug(f"max_event_rate_height: {max_event_rate_height}")
    return np.ceil(max_event_rate_height * 11) / 10
