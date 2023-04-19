""" Plot base module.

Defines utilizes and settings.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging

import attr
import pachyderm.plot
import seaborn as sns
from pachyderm.plot import AxisConfig, Figure, LegendConfig, Panel, PlotConfig, TextConfig  # noqa: F401

logger = logging.getLogger(__name__)

pachyderm.plot.configure()


label_to_display_string: dict[str, dict[str, str]] = {
    "ALICE": dict(  # noqa: C408
        work_in_progress="ALICE Work in Progress",
        preliminary="ALICE Preliminary",
        final="ALICE",
        simulation="ALICE Simulation",
    ),
    "collision_system": dict(  # noqa: C408
        PbPb_5TeV=r"$\text{Pb--Pb}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        embedPythia_5TeV=r"$\text{{PYTHIA8}} \bigotimes \text{{{main_system}}}\;\text{{Pb--Pb}}\;\sqrt{{s_{{\text{{NN}}}}}} = 5.02$ TeV",
        embed_pythia_5TeV=r"$\text{{PYTHIA8}} \bigotimes \text{{{main_system}}}\;\text{{Pb--Pb}}\;\sqrt{{s_{{\text{{NN}}}}}} = 5.02$ TeV",
        pp_PbPb_5TeV=r"$\text{pp},\:\text{Pb--Pb}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        pp_5TeV=r"$\text{pp}\;\sqrt{s} = 5.02$ TeV",
        pp_5TeV_NN=r"$\text{pp}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        pythia_5TeV=r"$\text{PYTHIA8}\;\sqrt{s} = 5.02$ TeV",
        pythia_5TeV_NN=r"$\text{PYTHIA8}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        # Without energy are deprecated
        PbPb=r"$\text{Pb--Pb}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        embedPythia=r"$\text{{PYTHIA8}} \bigotimes \text{{{main_system}}}\;\text{{Pb--Pb}}\;\sqrt{{s_{{\text{{NN}}}}}} = 5.02$ TeV",
        embed_pythia=r"$\text{{PYTHIA8}} \bigotimes \text{{{main_system}}}\;\text{{Pb--Pb}}\;\sqrt{{s_{{\text{{NN}}}}}} = 5.02$ TeV",
    ),
    "jets": {f"R0{i}": (f"$R=0.{i}," + fr"\:|\eta_{{\text{{jet}}}}| < 0.{9-i}$") for i in range(1, 7)},  # noqa: ISC003
}
label_to_display_string["jets"]["general"] = r"$\text{Anti-}k_{\text{T}}\:\text{ch-particle jets}$"


@attr.s
class GroomingMethodStyle:
    color: str = attr.ib()
    marker: str = attr.ib()
    fillstyle: str = attr.ib()
    label: str = attr.ib()
    label_short: str = attr.ib()
    zorder: int = attr.ib()


def define_grooming_styles() -> dict[str, GroomingMethodStyle]:
    # Setup
    styles = {}

    greens = sns.color_palette("Greens_d", 4)
    purples = sns.color_palette("Purples_d", 3)
    reds = sns.color_palette("Reds_d", 3)
    # greys = sns.color_palette("Greys_r", 5)
    blues = sns.color_palette("Blues_r", 3)
    oranges = sns.color_palette("Oranges_r", 3)
    for label in ["", "_compare"]:
        if not label:
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
                color=dynamical_grooming_colors[0], marker=markers[0], fillstyle="full",
                label="Dynamical grooming $a = 0.1$",
                label_short="DyG. $a = 0.1$",
                zorder=10,
            ),
            f"dynamical_kt{label}": GroomingMethodStyle(
                color=greens[1],
                marker=markers[0],
                fillstyle="full",
                label="Dynamical grooming $a = 1.0$",
                label_short=r"DyG $a = 1.0$",
                zorder=10,
            ),
            f"dynamical_time{label}": GroomingMethodStyle(
                color=reds[1], marker=markers[2], fillstyle="full",
                label=r"Dynamical grooming $a = 2.0$",
                label_short=r"DyG $a = 2.0$",
                zorder=10,
            ),
            f"dynamical_core{label}": GroomingMethodStyle(
                color=oranges[1], marker=markers[2], fillstyle="full",
                label=r"Dynamical grooming $a = 0.5$",
                label_short=r"DyG $a = 0.5$",
                zorder=10,
            ),
            # With zcut
            f"dynamical_kt_z_cut_02{label}": GroomingMethodStyle(
                color=greens[1],
                marker=markers[0],
                fillstyle="none",
                label=r"Dynamical grooming $a = 1.0$, $z = 0.2$",
                label_short=r"DyG $a = 1.0$, $z = 0.2$",
                zorder=10,
            ),
            f"dynamical_time_z_cut_02{label}": GroomingMethodStyle(
                color=reds[1],
                marker=markers[2],
                fillstyle="none",
                label=r"Dynamical grooming $a = 2.0$, $z = 0.2$",
                label_short=r"DyG $a = 2.0$, $z = 0.2$",
                zorder=10,
            ),
            f"dynamical_core_z_cut_02{label}": GroomingMethodStyle(
                color=oranges[1],
                marker=markers[2],
                fillstyle="none",
                label=r"Dynamical grooming $a = 0.5$, $z = 0.2$",
                label_short=r"DyG $a = 0.5$, $z = 0.2$",
                zorder=10,
            ),
            f"leading_kt{label}": GroomingMethodStyle(
                color=purples[1],
                marker=markers[1],
                fillstyle="full",
                label=r"Leading $k_{\text{T}}$",
                label_short=r"Lead. $k_{\text{T}}$",
                zorder=10,
            ),
            f"leading_kt_z_cut_02{label}": GroomingMethodStyle(
                color=blues[1] if not label else purples[1],
                marker=markers[1],
                fillstyle="none",
                label=r"Leading $k_{\text{T}}$ $z > 0.2$",
                label_short=r"Lead. $k_{\text{T}}$ $z > 0.2$",
                zorder=4,
            ),
            f"leading_kt_z_cut_04{label}": GroomingMethodStyle(
                color=leading_kt_colors[1],
                marker=markers[2],
                fillstyle="full",
                label=r"Leading $k_{\text{T}}$ $z > 0.4$",
                label_short=r"Lead. $k_{\text{T}}$ $z > 0.4$",
                zorder=10,
            ),
            # Leading kt with z cuts, but n <= 1
            f"leading_kt_z_cut_02_first_split{label}": GroomingMethodStyle(
                color=leading_kt_colors[0],
                marker="P",
                fillstyle="none",
                label=r"Leading $k_{\text{T}}$ $z > 0.2$, $n \leq 1$",
                label_short=r"Lead. $k_{\text{T}}$ $z > 0.2$, $n \leq 1$",
                zorder=4,
            ),
            f"leading_kt_z_cut_04_first_split{label}": GroomingMethodStyle(
                color=leading_kt_colors[0],
                marker="P",
                fillstyle="full",
                label=r"Leading $k_{\text{T}}$ $z > 0.4$, $n \leq 1$",
                label_short=r"Lead. $k_{\text{T}}$ $z > 0.4$, $n \leq 1$",
                zorder=10,
            ),
            f"soft_drop_z_cut_02{label}": GroomingMethodStyle(
                color=soft_drop_colors[1], marker=markers[1], fillstyle="none",
                label=r"Soft drop $z_{\text{cut}} = 0.2$",
                label_short=r"SD $z_{\text{cut}} = 0.2$",
                zorder=4,
            ),
            f"soft_drop_z_cut_04{label}": GroomingMethodStyle(
                color=soft_drop_colors[1], marker=markers[2], fillstyle="full",
                label=r"Soft drop $z_{\text{cut}} = 0.4$",
                label_short=r"SD $z_{\text{cut}} = 0.4$",
                zorder=5,
            ),
        }
        styles.update(grooming_styling)

    return styles


def adjust_lightness(color: str | tuple[float, float, float], amount: float = 0.5) -> tuple[float, float, float]:
    """Adjust lightness of a given color.

    From: https://stackoverflow.com/a/49601444/12907985

    NOTE:
        As I recall in April 2023, this didn't really work as well as I hoped. If I need it further, it may need
        more debugging.
    """
    import colorsys

    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except:
        c = color
    #c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c = colorsys.rgb_to_hls(*c)
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
