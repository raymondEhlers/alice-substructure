#!/usr/bin/env python3

""" Plotting for the tree skim.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, Sequence

import attr
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
from pachyderm import binned_data

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)

pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True


@attr.s
class PlotConfig:
    name: str = attr.ib()
    x_label: str = attr.ib()
    y_label: str = attr.ib()
    legend_location: str = attr.ib(default="center right")
    log_y: bool = attr.ib(default=True)


def _plot_residual(
    hists: Dict[str, bh.Histogram],
    label: str,
    grooming_method: str,
    matching_types: Sequence[str],
    plot_config: PlotConfig,
    output_dir: Path,
    min_hybrid_kt: float = 0,
) -> None:
    """

    Note:
        The min_hybrid_kt is only meaningful for the kt residual because it has the kt axis...

    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig_simplified, ax_simplified = plt.subplots(figsize=(8, 6))

    for matching_type in matching_types:
        logger.debug(
            f"Plotting {label} residual for {grooming_method}, {matching_type}, min_hybrid_kt: {min_hybrid_kt}"
        )

        matches_label = " ".join(matching_type.split("_")).capitalize()
        bh_hist = hists[f"{grooming_method}_hybrid_det_{label}_residuals_matching_type_{matching_type}"]
        h = binned_data.BinnedData.from_existing_data(bh_hist)

        selection_list = [slice(None), slice(None), slice(None)]
        if min_hybrid_kt:
            # Apply a hybrid kt cut.
            selection_list[0] = slice(h.axes[0].find_bin(min_hybrid_kt), None,)
        # Must be a tuple to be used for indexing, but need a list for reassignment.
        selection = tuple(selection_list)

        # Axes: hybrid, det, residual
        # For example, for jet pt, it's: Axes: hybrid_level_jet_pt, det_level_jet_pt, residual
        h_residual = binned_data.BinnedData(
            axes=[h.axes[2]],
            values=np.sum(h.values[selection], axis=(0, 1)),
            variances=np.sum(h.variances[selection], axis=(0, 1)),
        )

        # Normalize
        h_residual /= np.sum(h_residual.values)

        if matching_type in ["all", "pure", "swap"]:
            ax_simplified.errorbar(
                h_residual.axes[0].bin_centers,
                h_residual.values,
                yerr=h_residual.errors,
                xerr=h_residual.axes[0].bin_widths / 2,
                marker=".",
                linestyle="",
                label=matches_label,
            )
        else:
            # Let's rebin otherwise to reduce error bar size for some other the other methods.
            h_residual = binned_data.BinnedData.from_existing_data(h_residual.to_boost_histogram()[:: bh.rebin(4)])
            # Normalize again
            h_residual /= np.sum(h_residual.values)

        ax.errorbar(
            h_residual.axes[0].bin_centers,
            h_residual.values,
            yerr=h_residual.errors,
            xerr=h_residual.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=matches_label,
        )

    # Labeling
    for a, f in [(ax, fig), (ax_simplified, fig_simplified)]:
        text = "Iterative splittings"
        text += "\n" + f"${helpers.RangeSelector(40, 120).display_str(label='hybrid')}$"
        text += "\n" + " ".join(grooming_method.split("_")).capitalize()
        if min_hybrid_kt:
            text += "\n" + fr"$k_{{\text{{T}}}}^{{\text{{hybrid}}}} > {min_hybrid_kt}\:\text{{GeV}}/c$"
        a.text(
            0.97,
            0.97,
            text,
            transform=a.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            multialignment="right",
        )

        # Presentation
        a.set_xlabel(plot_config.x_label)
        a.set_ylabel(plot_config.y_label)
        a.legend(frameon=False, loc="upper left", fontsize=14)
        f.tight_layout()
        f.subplots_adjust(
            # Reduce spacing between subplots
            hspace=0,
            wspace=0,
            # Reduce external spacing
            left=0.10,
            bottom=0.12,
            right=0.99,
            top=0.98,
        )

    # Store and cleanup
    filename = f"{plot_config.name}_iterative_splittings_{grooming_method}"
    if min_hybrid_kt:
        filename += f"_min_hybrid_kt_{min_hybrid_kt}"
    fig.savefig(output_dir / f"{filename}_matching.pdf")
    plt.close(fig)
    fig_simplified.savefig(output_dir / f"{filename}_matching_simplified.pdf")
    plt.close(fig_simplified)


def plot_residuals(
    hists: Dict[str, bh.Histogram], grooming_methods: Sequence[str], matching_types: Sequence[str], output_dir: Path
) -> None:
    for grooming_method in grooming_methods:
        _plot_residual(
            hists=hists,
            label="jet_pt",
            grooming_method=grooming_method,
            matching_types=matching_types,
            plot_config=PlotConfig(
                name="jet_pt_residual_hybrid_detector",
                x_label=r"$(p_{\text{T}}^{\text{hybrid}} - p_{\text{T}}^{\text{det}}) / p_{\text{T}}^{\text{det}}$",
                y_label="",
            ),
            output_dir=output_dir,
        )
        _plot_residual(
            hists=hists,
            label="kt",
            grooming_method=grooming_method,
            matching_types=matching_types,
            plot_config=PlotConfig(
                name="kt_residual_hybrid_detector",
                x_label=r"$(k_{\text{T}}^{\text{hybrid}} - k_{\text{T}}^{\text{det}}) / k_{\text{T}}^{\text{det}}$",
                y_label="",
            ),
            output_dir=output_dir,
        )
        _plot_residual(
            hists=hists,
            label="kt",
            grooming_method=grooming_method,
            matching_types=matching_types,
            plot_config=PlotConfig(
                name="kt_residual_hybrid_detector",
                x_label=r"$(k_{\text{T}}^{\text{hybrid}} - k_{\text{T}}^{\text{det}}) / k_{\text{T}}^{\text{det}}$",
                y_label="",
            ),
            output_dir=output_dir,
            min_hybrid_kt=5,
        )
