*Copyright © 2024 by Boston Consulting Group. All rights reserved*
# SmartBanking Plotting tool - Plottah

## Overview
This package provides a set of core modules for creating, handling, and assembling visual plots. The three primary components of the package are:

- **Plots**: Represent visual elements that can be displayed.
- **Handlers**: Interact with an abstraction of a Plot, with varying configurations such as rows and columns.
- **Builders**: Assemble Plots and Handlers to create comprehensive visual compositions.


## Design Choices
- Handlers: These components exclusively interact with an abstraction of a Plot. Handlers come in various configurations, with different numbers of rows and columns based on specific requirements.
- Builders: Functions responsible for combining Plots and Handlers, ensuring that the correct number of plots is created based on the specifications of a given handler: i.e., layout in terms of rows and columns of a specific handler.

