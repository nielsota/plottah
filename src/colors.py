from dataclasses import dataclass


@dataclass
class PlotColors:
    """
    container for colors to use
    """

    primary_color: str = "40, 186, 116"
    secondary_color: str = "41, 94, 126"
    tertiary_color: str = "153, 204, 235"
    grey_tint_color: str = "110, 111, 115"

    def __post_init__(self):
        self.colors = {
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "tertiary_color": self.tertiary_color,
            "grey_tint_color": self.grey_tint_color,
        }

    def get_rgba(self, color: str = "primary_color", opacity: float = 1):
        if color not in self.colors.keys():
            raise ValueError(
                f"{color} is not one of the colors, choose from: {list(self.colors.keys())}"
            )

        return "rgba(" + self.colors[color] + f", {opacity})"

    def get_grey_rgba(self, opacity: float = 1):
        return "rgba(" + self.colors["grey_tint_color"] + f", {opacity})"


SOMEBANK_COLORS = PlotColors(
    primary_color="231, 30, 87",
    secondary_color="153, 204, 235",
    tertiary_color="254, 189, 64",
    grey_tint_color="110, 111, 115",
)

BCG_COLORS = PlotColors(
    primary_color="40, 186, 116",
    secondary_color="41, 94, 126",
    tertiary_color="153, 204, 235",
    grey_tint_color="110, 111, 115",
)
