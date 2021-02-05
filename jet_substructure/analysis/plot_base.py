""" Plot base module.

Defines utilizes and settings.

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import attr
import matplotlib
import numpy as np
import pachyderm.plot
import seaborn as sns


logger = logging.getLogger(__name__)

pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True


label_to_display_string: Dict[str, Dict[str, str]] = {
    "ALICE": dict(
        work_in_progress="ALICE Work in Progress",
        preliminary="ALICE Preliminary",
        final="ALICE",
        simulation="ALICE Simulation",
    ),
    "collision_system": dict(
        PbPb=r"$\text{Pb--Pb}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        embedPythia=r"$\text{{PYTHIA8}} \bigotimes \text{{{main_system}}}\;\text{{Pb--Pb}}\;\sqrt{{s_{{\text{{NN}}}}}} = 5.02$ TeV",
        pp_5TeV=r"$\text{pp}\;\sqrt{s} = 5.02$ TeV",
        pp_5TeV_NN=r"$\text{pp}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        pythia_5TeV=r"$\text{PYTHIA8}\;\sqrt{s} = 5.02$ TeV",
        pythia_5TeV_NN=r"$\text{PYTHIA8}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
    ),
    "jets": {f"R0{i}": (f"$R=0.{i}," + r"\:|\eta_{\text{jet}}| < 0.5$") for i in range(1, 7)},
}
label_to_display_string["jets"]["general"] = r"$\text{Anti-}k_{\text{T}}\:\text{charged jets}$"


def _validate_axis_name(instance: "AxisConfig", attribute: attr.Attribute[str], value: str) -> None:
    if value not in ["x", "y", "z"]:
        raise ValueError("Invalid axis name: {value}")


@attr.s
class AxisConfig:
    axis: str = attr.ib(validator=[_validate_axis_name])
    label: str = attr.ib(default="")
    log: bool = attr.ib(default=False)
    range: Tuple[Optional[float], Optional[float]] = attr.ib(default=None)
    font_size: Optional[float] = attr.ib(default=None)

    def apply(self, ax: matplotlib.axes.Axes) -> None:
        if self.label:
            getattr(ax, f"set_{self.axis}label")(self.label, fontsize=self.font_size)
        if self.log:
            getattr(ax, f"set_{self.axis}scale")("log")
            # Probably need to increase the number of ticks for a log axis. We just assume that's the case.
            # I really wish it handled this better by default...
            # See: https://stackoverflow.com/a/44079725/12907985
            major_locator = matplotlib.ticker.LogLocator(base=10, numticks=12)
            getattr(ax, f"{self.axis}axis").set_major_locator(major_locator)
            minor_locator = matplotlib.ticker.LogLocator(base=10.0, subs=np.linspace(0.2, 0.9, 8), numticks=12)
            getattr(ax, f"{self.axis}axis").set_minor_locator(minor_locator)
            # But we don't want to label these ticks.
            getattr(ax, f"{self.axis}axis").set_minor_formatter(matplotlib.ticker.NullFormatter())
        if self.range:
            min_range, max_range = self.range
            min_current_range, max_current_range = getattr(ax, f"get_{self.axis}lim")()
            if min_range is None:
                min_range = min_current_range
            if max_range is None:
                max_range = max_current_range
            getattr(ax, f"set_{self.axis}lim")([min_range, max_range])


@attr.s
class TextConfig:
    text: str = attr.ib()
    x: float = attr.ib()
    y: float = attr.ib()
    alignment: Optional[str] = attr.ib(default=None)
    color: Optional[str] = attr.ib(default="black")
    font_size: Optional[float] = attr.ib(default=None)

    def apply(self, ax: matplotlib.axes.Axes) -> None:
        # Some reasonable defaults
        if self.alignment is None:
            ud = "upper" if self.y >= 0.5 else "lower"
            lr = "right" if self.x >= 0.5 else "left"
            self.alignment = f"{ud} {lr}"

        alignments = {
            "upper right": dict(
                horizontalalignment="right",
                verticalalignment="top",
                multialignment="right",
            ),
            "upper left": dict(
                horizontalalignment="left",
                verticalalignment="top",
                multialignment="left",
            ),
            "lower right": dict(
                horizontalalignment="right",
                verticalalignment="bottom",
                multialignment="right",
            ),
            "lower left": dict(
                horizontalalignment="left",
                verticalalignment="bottom",
                multialignment="left",
            ),
        }
        alignment_kwargs = alignments[self.alignment]

        # Finally, draw the text.
        ax.text(
            self.x,
            self.y,
            self.text,
            color=self.color,
            fontsize=self.font_size,
            # We always want to place using normalized coordinates.
            # In the rare case that we don't want to, we can place by hand.
            transform=ax.transAxes,
            **alignment_kwargs,
        )


@attr.s
class LegendConfig:
    location: str = attr.ib(default=None)
    # Takes advantage of the fact that None will use the default.
    anchor: Optional[Tuple[float, float]] = attr.ib(default=None)
    font_size: Optional[float] = attr.ib(default=None)
    ncol: Optional[float] = attr.ib(default=1)
    marker_label_spacing: Optional[float] = attr.ib(default=None)

    def apply(
        self,
        ax: matplotlib.axes.Axes,
        legend_handles: Optional[Sequence[matplotlib.container.ErrorbarContainer]] = None,
        legend_labels: Optional[Sequence[str]] = None,
    ) -> None:
        if self.location:
            kwargs = {}
            if legend_handles:
                kwargs["handles"] = legend_handles
            if legend_labels:
                kwargs["labels"] = legend_labels

            ax.legend(
                loc=self.location,
                bbox_to_anchor=self.anchor,
                # If we specify an anchor, we want to reduce an additional padding
                # to ensure that we have accurate placement.
                borderaxespad=(0 if self.anchor else None),
                borderpad=(0 if self.anchor else None),
                frameon=False,
                fontsize=self.font_size,
                ncol=self.ncol,
                handletextpad=self.marker_label_spacing,
                **kwargs,
            )


def _ensure_sequence_of_axis_config(value: Union[AxisConfig, Sequence[AxisConfig]]) -> Sequence[AxisConfig]:
    if isinstance(value, AxisConfig):
        value = [value]
    return value


@attr.s
class Panel:
    axes: Sequence[AxisConfig] = attr.ib(converter=_ensure_sequence_of_axis_config)
    text: Optional[TextConfig] = attr.ib(default=None)
    legend: LegendConfig = attr.ib(default=None)

    def apply(
        self,
        ax: matplotlib.axes.Axes,
        legend_handles: Optional[Sequence[matplotlib.container.ErrorbarContainer]] = None,
        legend_labels: Optional[Sequence[str]] = None,
    ) -> None:
        # Axes
        for axis in self.axes:
            axis.apply(ax)
        # Text
        if self.text is not None:
            self.text.apply(ax)
        # Legend
        if self.legend is not None:
            self.legend.apply(ax, legend_handles=legend_handles, legend_labels=legend_labels)


@attr.s
class Figure:
    edge_padding: Mapping[str, float] = attr.ib(factory=dict)

    def apply(self, fig: matplotlib.figure.Figure) -> None:
        # It shouldn't hurt to align the labels if there's only one.
        fig.align_ylabels()

        # Adjust the layout.
        fig.tight_layout()
        adjust_default_args = dict(
            # Reduce spacing between subplots
            hspace=0,
            wspace=0,
            # Reduce external spacing
            left=0.10,
            bottom=0.105,
            right=0.98,
            top=0.98,
        )
        adjust_default_args.update(self.edge_padding)
        fig.subplots_adjust(**adjust_default_args)


def _ensure_sequence_of_panels(value: Union[Panel, Sequence[Panel]]) -> Sequence[Panel]:
    if isinstance(value, Panel):
        value = [value]
    return value


@attr.s
class PlotConfig:
    name: str = attr.ib()
    panels: Sequence[Panel] = attr.ib(converter=_ensure_sequence_of_panels)
    figure: Figure = attr.ib(factory=Figure)

    def apply(
        self,
        fig: matplotlib.figure.Figure,
        ax: Optional[matplotlib.axes.Axes] = None,
        axes: Optional[Sequence[matplotlib.axes.Axes]] = None,
        legend_handles: Optional[Sequence[matplotlib.container.ErrorbarContainer]] = None,
        legend_labels: Optional[Sequence[str]] = None,
    ) -> None:
        # Validation
        if ax is None and axes is None:
            raise TypeError("Must pass the axis or axes of the figure.")
        if ax is not None and axes is not None:
            raise TypeError("Cannot pass both a single axis and multiple axes.")
        # If we just have a single axis, wrap it up into a list so we can process it along with our panels.
        if ax is not None:
            axes = [ax]
        # Help out mypy...
        assert axes is not None
        if len(axes) != len(self.panels):
            raise ValueError(
                f"Must have the same number of axes and panels. Passed axes: {axes}, panels: {self.panels}"
            )

        # Finally, we can actually apply the stored properties.
        # Apply panels to the axes.
        for ax, panel in zip(axes, self.panels):
            panel.apply(ax, legend_handles=legend_handles, legend_labels=legend_labels)
        # Figure
        self.figure.apply(fig)


@attr.s
class GroomingMethodStyle:
    color: str = attr.ib()
    marker: str = attr.ib()
    fillstyle: str = attr.ib()
    label: str = attr.ib()
    zorder: int = attr.ib()


def define_grooming_styles() -> Dict[str, GroomingMethodStyle]:
    # Setup
    styles = {}

    greens = sns.color_palette("Greens_d", 4)
    purples = sns.color_palette("Purples_d", 3)
    reds = sns.color_palette("Reds_d", 3)
    # greys = sns.color_palette("Greys_r", 5)
    blues = sns.color_palette("Blues_r", 3)
    oranges = sns.color_palette("Oranges_r", 3)
    for label in ["", "_compare"]:
        if label == "":
            # These are our main colors.
            # The methods are similar, but different, so we want to spread out the colors.
            # dynamical_grooming_colors = sns.color_palette(f"GnBu_d", 3)
            dynamical_grooming_colors = sns.color_palette("Greens_d", 4)
            leading_kt_colors = sns.color_palette("Purples_d", 3)
            soft_drop_colors = sns.color_palette("Reds_d", 3)
        else:
            # These are our comparison colors. Similar in order and often shade, but distinct.
            dynamical_grooming_colors = sns.color_palette("Greys_r", 5)
            leading_kt_colors = sns.color_palette("Blues_r", 3)
            soft_drop_colors = sns.color_palette("Oranges_r", 3)
        markers = ["o", "d", "s"]
        grooming_styling = {
            f"dynamical_z{label}": GroomingMethodStyle(
                color=dynamical_grooming_colors[0], marker=markers[0], fillstyle="full", label="DyG. $z$", zorder=10
            ),
            f"dynamical_kt{label}": GroomingMethodStyle(
                color=greens[1],
                marker=markers[0],
                fillstyle="full",
                label=r"DyG $k_{\text{T}}$",
                zorder=10,
            ),
            f"dynamical_time{label}": GroomingMethodStyle(
                color=reds[1], marker=markers[2], fillstyle="full", label=r"DyG time", zorder=10
            ),
            f"dynamical_core{label}": GroomingMethodStyle(
                color=oranges[1], marker=markers[2], fillstyle="full", label=r"DyG core", zorder=10
            ),
            f"leading_kt{label}": GroomingMethodStyle(
                color=purples[1],
                marker=markers[1],
                fillstyle="full",
                label=r"Leading $k_{\text{T}}$",
                zorder=10,
            ),
            f"leading_kt_z_cut_02{label}": GroomingMethodStyle(
                color=blues[1] if not label else purples[1],
                marker=markers[1],
                fillstyle="none",
                label=r"Leading $k_{\text{T}}$ $z > 0.2$",
                zorder=4,
            ),
            f"leading_kt_z_cut_04{label}": GroomingMethodStyle(
                color=leading_kt_colors[1],
                marker=markers[2],
                fillstyle="full",
                label=r"Leading $k_{\text{T}}$ $z > 0.4$",
                zorder=10,
            ),
            # Leading kt with z cuts, but n <= 1
            f"leading_kt_z_cut_02_first_split{label}": GroomingMethodStyle(
                color=leading_kt_colors[0],
                marker="P",
                fillstyle="none",
                label=r"Leading $k_{\text{T}}$ $z > 0.2$, $n \leq 1$",
                zorder=4,
            ),
            f"leading_kt_z_cut_04_first_split{label}": GroomingMethodStyle(
                color=leading_kt_colors[0],
                marker="P",
                fillstyle="full",
                label=r"Leading $k_{\text{T}}$ $z > 0.4$, $n \leq 1$",
                zorder=10,
            ),
            f"soft_drop_z_cut_02{label}": GroomingMethodStyle(
                color=soft_drop_colors[1], marker=markers[1], fillstyle="none", label=r"SoftDrop $z > 0.2$", zorder=4
            ),
            f"soft_drop_z_cut_04{label}": GroomingMethodStyle(
                color=soft_drop_colors[1], marker=markers[2], fillstyle="full", label=r"SoftDrop $z > 0.4$", zorder=5
            ),
        }
        styles.update(grooming_styling)

    return styles
