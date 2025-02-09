from typing import Literal

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from loguru import logger  # type: ignore

# TODO: seperate utils from binning
NUMERICAL_BINS = list[int | float]


def quantile_clipping(
    df: pd.DataFrame, feature_col: str, q_clip_min: float, q_clip_max: float
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        dataframe you want to clip based on quantiles of feature col
    feature_col : str
        column name of column you want quantiles of
    q_clip_min : int
        number between 1-100 - if 99, remove the top 1%
    q_clip_max : int
        number between 1-100 - if 1, remove the bottom 1%

    Returns
    -------
    pd.DataFrame
        quantile clipped dataframe
    """
    min_q = df[feature_col].quantile(q_clip_min)
    max_q = df[feature_col].quantile(q_clip_max)

    return df.loc[(df[feature_col] >= min_q) & (df[feature_col] <= max_q)]


# TODO: update type hinting
def generate_n_quantile_bins(
    df: pd.DataFrame,
    feature_col: str,
    min_val_adj: float,
    max_val_adj: float,
    n_bins: int,
) -> np.ndarray:
    """ensures we use the appropriate number of quantiles to get nbin bins

    Example:
        For a heavily skewed distribution, when using 10 quantiles, we might only get 2 values
        for example with [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], all but the last quantile will be 0, and the only bin
        will be (0,1).

    Args:
        df: pd.DataFrame,
        feature_col: str,
        min_val_adj: float,
        max_val_adj: float,
        n_bins: int

    Returns:
        Bins
    """

    logger.debug(
        f"Generating {n_bins} bins for {feature_col} between {min_val_adj} and {max_val_adj}"
    )

    # start count at number of bins
    i = n_bins

    bins = np.unique(
        df[feature_col].clip(min_val_adj, max_val_adj).quantile(np.linspace(0, 1, i))
    )

    # if after creating n_bin quantiles, we don't have n_bin bins, increase number of quantiles
    while (len(bins)) < n_bins:
        bins = np.unique(
            df[feature_col]
            .clip(min_val_adj, max_val_adj)
            .quantile(np.linspace(0, 1, i))
        )
        i += 1

    return NUMERICAL_BINS(bins)


def generate_linear_bins(
    min_val_adj: float,
    max_val_adj: float,
    n_bins: int,
) -> NUMERICAL_BINS:
    """Generate linear bins."""
    logger.debug(
        f"Generating {n_bins} linear bins between {min_val_adj} and {max_val_adj}"
    )
    return NUMERICAL_BINS(np.linspace(min_val_adj, max_val_adj, n_bins))


def remove_or_impute_nan_infs(
    df: pd.DataFrame, feature_col: str, target_col: str, fillna: bool = False
):
    """
    remove nan, inf and -inf elements or impute w/ median
    """
    if fillna:
        df.loc[df[feature_col].isin([-np.inf, np.inf, np.nan]), feature_col] = df.loc[
            ~df[feature_col].isin([-np.inf, np.inf, np.nan]), feature_col
        ].median()
    else:
        df = df.loc[~df[feature_col].isin([-np.inf, np.inf, np.nan])]

    return df


# TODO: update type hinting
def generate_bins(
    df: pd.DataFrame,
    feature_col: str,
    min_val_adj: float,
    max_val_adj: float,
    n_bins: int = 10,
    method: Literal["quantile", "linear"] = "quantile",
) -> NUMERICAL_BINS:
    """
    generates bins using various methods
    """

    # check: if method not manual then must be quantile or linear
    if method not in ["quantile", "linear"]:
        raise ValueError(
            f"{method} not one of the allowed methods, must be in [quantile, linear]"
        )

    # Create equidistant grid
    if method == "quantile":
        # use this function to ensure we get nbin bins even when using quantiles
        bins = generate_n_quantile_bins(
            df, feature_col, min_val_adj, max_val_adj, n_bins
        )

    else:
        # get bins using linspace
        bins = generate_linear_bins(min_val_adj, max_val_adj, n_bins)

    return bins


def get_min_max(df: pd.DataFrame, feature_col: str) -> tuple[float, float]:
    """
    get min and max while ignoring nan values

    see: https://stackoverflow.com/questions/62865647/why-is-max-and-min-of-numpy-array-nan
    """

    # Get unadjusted min and max
    min_val = np.nanmin(df[feature_col])
    max_val = np.nanmax(df[feature_col])

    return (min_val, max_val)


def get_min_max_adj(df: pd.DataFrame, feature_col: str) -> tuple[float, float]:
    """
    get min and max while ignoring nan values and possible +/- inf

    see: https://stackoverflow.com/questions/62865647/why-is-max-and-min-of-numpy-array-nan
    """

    # Get unadjusted min and max
    min_val = np.nanmin(df[feature_col])
    max_val = np.nanmax(df[feature_col])

    # get min & max excluding +/- inf
    idx_neg_inf = df[feature_col] == -np.inf
    idx_pos_inf = df[feature_col] == np.inf
    min_val_excl_inf = np.nanmin(df[feature_col][~idx_neg_inf])
    max_val_excl_inf = np.nanmax(df[feature_col][~idx_pos_inf])

    # take largest minimum and smallest maximum
    min_val_adj = np.maximum(min_val, min_val_excl_inf)
    max_val_adj = np.minimum(max_val, max_val_excl_inf)

    return (min_val_adj, max_val_adj)


def get_labels_from_bins(bins: NUMERICAL_BINS) -> list[str]:
    """
    Generate labels from bins. The precision (decimal places) is automatically determined
    to ensure all bin values are uniquely represented.

    Example:
        [0, 1, 2] -> ['(0.00, 1.00]', '(1.00, 2.00]']

    If bins are very close together (e.g., [1.11111, 1.11112]), precision will increase
    until all values can be distinguished.

    Parameters
    ----------
    bins : NUMERICAL_BINS
        List of bin edges

    Returns
    -------
    list
        List of formatted bin labels as right-inclusive intervals
    """
    precision = 0
    while len(bins) > len(np.unique(np.round(bins, precision))):
        logger.debug(f"precision currently: {precision}")
        precision += 1

    def _format_bin_value(value: float, precision: int) -> str:
        return "{:,.{prec}f}".format(value, prec=precision)

    labels = []
    for i in range(len(bins) - 1):
        left_bracket = "[" if i == 0 else "("
        left_value = _format_bin_value(bins[i], precision)
        right_value = _format_bin_value(bins[i + 1], precision)
        labels.append(f"{left_bracket}{left_value}, {right_value}]")
    return labels


def validate_binary_target(values: pd.Series | pd.DataFrame | np.ndarray) -> bool:
    """Validates that input contains only 0 and 1 values."""

    # Convert input to numpy array for consistent handling
    if isinstance(values, (pd.Series, pd.DataFrame)):
        values = values.values

    # Get unique values, excluding NaN
    unique_vals = np.unique(values[~np.isnan(values)])

    # Log unique values found
    logger.debug(f"Found unique target values: {unique_vals}")

    # Check if only 0 or only 1 present
    if len(unique_vals) == 1:
        raise ValueError(
            f"Target contains only value {unique_vals[0]}. Both 0 and 1 must be present."
        )

    # Check if values other than 0 and 1 are present
    if not set(unique_vals) <= {0, 1}:
        raise ValueError(f"Target contains values other than 0 and 1: {unique_vals}")

    return True


def validate_feature_column_presence(df: pd.DataFrame, feature_col: str) -> bool:
    """Validates that the feature column is present in the dataframe."""
    if feature_col not in df.columns:
        raise ValueError(f"Feature column {feature_col} not found in dataframe")
    return True


if __name__ == "__main__":
    # Create sample data
    logger.info("Creating sample data...")
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, np.inf, -np.inf, np.nan],
            "other": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )

    # Test quantile_clipping
    logger.info("Testing quantile_clipping...")
    df_clipped = quantile_clipping(df, "feature", 0.1, 0.9)
    logger.info(f"Original shape: {df.shape}, Clipped shape: {df_clipped.shape}")

    # Test generate_n_quantile_bins
    logger.info("Testing generate_n_quantile_bins...")
    min_val, max_val = get_min_max_adj(df, "feature")
    bins = generate_n_quantile_bins(df, "feature", min_val, max_val, n_bins=3)
    logger.info(f"Generated bins: {bins}")

    # Test get_min_max_adj
    logger.info("Testing get_min_max_adj...")
    min_adj, max_adj = get_min_max_adj(df, "feature")
    logger.info(f"Adjusted min: {min_adj}, Adjusted max: {max_adj}")

    # Test get_labels_from_bins
    logger.info("Testing get_labels_from_bins...")
    test_bins = [0, 1, 2, 3]
    labels = get_labels_from_bins(test_bins)
    logger.info(f"Generated labels: {labels}")

    logger.success("All tests completed successfully!")
