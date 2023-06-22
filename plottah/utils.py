import numpy as np
import pandas as pd

from typing import Tuple


def get_n_quantile_bins(
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

    return bins


def remove_or_impute_nan_infs(
    df: pd.DataFrame, feature_col: str, target_col: str, fillna: bool = False
):
    """
    remove nan, inf and -inf elements or impute w/ median
    """
    if fillna == True:
        df.loc[df[feature_col].isin([-np.inf, np.inf, np.nan]), feature_col] = df.loc[
            ~df[feature_col].isin([-np.inf, np.inf, np.nan]), feature_col
        ].median()
    else:
        df = df.loc[~df[feature_col].isin([-np.inf, np.inf, np.nan])]

    return df


def get_bins(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    min_val_adj: float,
    max_val_adj: float,
    n_bins: int = 10,
    method: str = "quantile",
    bins: list = None,
) -> np.ndarray:
    """
    creates bins using various methods
    """

    # check: if method not manual then must be quantile or linear
    if method not in ["quantile", "linear"]:
        raise ValueError(
            f"{method} not one of the allowed methods, must be in [quantile, linear]"
        )

    # Create equidistant grid
    if method == "quantile":
        # use this function to ensure we get nbin bins even when using quantiles
        bins = get_n_quantile_bins(df, feature_col, min_val_adj, max_val_adj, n_bins)

    else:
        # get bins using linspace
        bins = np.linspace(min_val_adj, max_val_adj, n_bins)

    return bins


def get_min_max(df: pd.DataFrame, feature_col: str) -> Tuple[float, float]:
    """
    get min and max while ignoring nan values

    see: https://stackoverflow.com/questions/62865647/why-is-max-and-min-of-numpy-array-nan
    """

    # Get unadjusted min and max
    min_val = np.nanmin(df[feature_col])
    max_val = np.nanmax(df[feature_col])

    return (min_val, max_val)


def get_min_max_adj(df: pd.DataFrame, feature_col: str) -> Tuple[float, float]:
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


def get_labels_from_bins(bins: list) -> list:
    """
    generate labels from bins

    example: [0, 1, 2] -> ['(0.00, 1.00]', '(1.00, 2.00]']
    """
    labels = [
        f'({"{:,.2f}".format(bins[i])}, {"{:,.2f}".format(bins[i+1])}]'
        for i in range(len(bins) - 1)
    ]
    return labels
