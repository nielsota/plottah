*Copyright © 2024 by Boston Consulting Group. All rights reserved*
# SmartBanking Plotting tool

## Setup

### Requirements

* Python (>=3.8)

### Development environment

Clone the repo, give name plotting_analyis
```shell
git clone git@github.com:nielsota/plottah.git plotting_analysis
```

CD into the directory
```shell
cd plotting_analysis
```

Create a virtual environment by running:

```shell
python -m venv venv
```

The virtual environment should be activated every time you start a new shell session before running subsequent commands:

> On Linux/MacOS:
> ```shell
> source venv/bin/activate
> ```
> On Windows (bash/cmd):
> ```shell
> venv/Scripts/activate
> ```
> On Windows (ps):
> ```shell
> venv\Scripts\activate
> ```
Make sure you have the latest pip version
```shell
python -m pip install --upgrade pip
```

Then install the packages listed in the requirements.txt
```shell
pip install -r requirements.txt
```

Then install the repository locally
```shell
pip install -e .
```

Next, update the config.yaml how you see fit, and then to generate the images run:
```shell
python -m plottah
```

See notebooks for examples on how to run code inline during development

