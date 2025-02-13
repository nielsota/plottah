import numpy as np
import pandas as pd
import pytest

from plottah.colors import PlotColors
from plottah.plot_builder.specific_builders import \
    build_split_bin_event_rate_plot


def test_build_split_bin_event_rate_plot():
    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 1000

    # Create test dataframe
    df = pd.DataFrame(
        {
            "predictor": np.concatenate(
                [
                    np.random.normal(0, 1, n_samples // 2),
                    np.random.normal(2, 1, n_samples // 2),
                ]
            ),
            "target": np.random.binomial(1, 0.3, n_samples),
            "split": np.concatenate(
                [
                    np.ones(n_samples // 2),
                    np.zeros(n_samples // 2),
                ]
            ),
        }
    )

    # Test that the function runs without raising any errors
    try:
        plot = build_split_bin_event_rate_plot(
            df_top=df.loc[df["split"] == 1],
            df_bottom=df.loc[df["split"] == 0],
            feature_col="predictor",
            target="target",
            bins=[0, 1, 2, 3, 4, 5],
            top_x_title="Predictor for (split == 1)",
            bottom_x_title="Predictor for (split == 0)",
            colors=PlotColors(),
            tick_font_size=18,
            title_font_size=18,
            legend_font_size=18,
            fillna_event_rate=False,
            title_standoff=10,
        )
        assert plot is not None
    except Exception as e:
        pytest.fail(f"build_split_bin_event_rate_plot raised an exception: {e}")
