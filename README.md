*Copyright © 2024 by Boston Consulting Group. All rights reserved*
# SmartBanking Plotting tool

A Python package for generating standardized univariate analysis plots and visualizations for SmartBanking analysis. The package automatically generates:
- ROC curves
- Distribution plots
- Bin event rate plots

## Installation

There are two ways to use this package:

### 1. Install from PyPI (Recommended)

If you just want to use the package as a library:

```shell
pip install plottah
```

### 2. Local Development with Config (For Batch Processing)

If you want to process multiple features using a configuration file:

1. Clone the repository:
```shell
git clone git@github.com:nielsota/plottah.git
cd plottah
```

2. Install Poetry (if you haven't already):
```shell
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies and set up the development environment:
```shell
poetry install
```

## Usage

### As a Python Package

```python
from plottah import build_univariate_plot
import pandas as pd

# Create your dataframe
df = pd.DataFrame(...)

# Generate a single plot
plot = build_univariate_plot(
    df=df,
    feature_col="your_feature",
    target="target_column",
    feature_type="numerical",  # or "categorical"
    n_bins=10,  # optional
    distplot_q_min=0.01,  # optional
    distplot_q_max=0.99   # optional
)
```

### Using Configuration File

1. Create a `config.yaml` file with your settings:
```yaml
file_path: ./data/your_data.csv
images_output_path: ./data/images
powerpoint_output_path: ./data/powerpoints/output.pptx

features:
  - name: feature_1
    n_bins: 10
  - name: feature_2
    bins: [0, 1, 5, 10, 100]
  - name: feature_3
    type: categorical

target: target_column

# Optional: Custom colors
primary_color: 231, 30, 87
secondary_color: 153, 204, 235
tertiary_color: 254, 189, 64
grey_tint_color: 110, 111, 115
```

2. Run the plotting tool:
```shell
poetry run python -m plottah
```

## Configuration Options

For each feature in the config file, you can specify:
- `name`: Feature column name (required)
- `type`: "numerical" or "categorical" (default: "numerical")
- `n_bins`: Number of bins for numerical features
- `bins`: Custom bin edges for numerical features
- `distplot_q_min`: Lower quantile for distribution plot trimming
- `distplot_q_max`: Upper quantile for distribution plot trimming

## Examples

See the `notebooks/` directory for detailed examples of:
- Basic univariate analysis
- Custom binning strategies
- Distribution plot customization
- ROC curve analysis

## Requirements

* Python >=3.8
* Poetry for development setup
