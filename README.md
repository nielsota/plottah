*Copyright © 2024 by Boston Consulting Group. All rights reserved*
# SmartBanking Plotting tool

A Python package for generating standardized plots and visualizations for SmartBanking analysis.

## Installation

There are two ways to use this package:

### 1. Install from PyPI (Recommended)

If you just want to use the package:

```shell
pip install plottah
```

### 2. Local Development Setup

If you want to contribute or modify the package:

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

4. Activate the poetry shell:
```shell
poetry shell
```

## Usage

### As an installed package

```python
from plottah import generate_plots

# Use the package functions directly
generate_plots()
```

### Local development with config

1. Update the `config.yaml` file with your desired settings
2. Run the plotting tool:
```shell
poetry run python -m plottah
```

## Examples

See the `notebooks/` directory for detailed examples of how to use the package during development.

## Requirements

* Python >=3.8
* Poetry for development setup
