"""
Plot builder module for creating univariate analysis plots.

This module provides functions for building univariate plots to analyze feature distributions
and their relationships with target variables.
"""

from .builders import build_univariate_plot, build_univariate_plots
from .specific_builders import build_split_bin_event_rate_plot

__all__ = [
    "build_univariate_plot",
    "build_univariate_plots",
    "build_split_bin_event_rate_plot",
]
