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


def _plot_residual_by_matching_type(
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
        # For example, for jet pt, it's: Axes: hybrid_jet_pt, det_level_jet_pt, residual
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


def plot_residuals_by_matching_type(
    hists: Dict[str, bh.Histogram], grooming_methods: Sequence[str], matching_types: Sequence[str], output_dir: Path
) -> None:
    for grooming_method in grooming_methods:
        _plot_residual_by_matching_type(
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
        _plot_residual_by_matching_type(
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
        _plot_residual_by_matching_type(
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


def _plot_residual_mean_and_width(
    hists: Dict[str, bh.Histogram],
    grooming_method: str,
    hybrid_jet_pt_bin: helpers.RangeSelector,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    logger.debug(
        f"Plotting jet pt residual mean and width for {grooming_method} with hybrid jet pt: {hybrid_jet_pt_bin}"
    )

    fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
    fig_width, ax_width = plt.subplots(figsize=(8, 6))

    for jet_types, label, color in [
        ("hybrid_det", "Fluctuations", "red"),
        ("hybrid_true", "Combined", "black"),
        ("det_true", "Detector", "blue"),
    ]:
        bh_hist = hists[f"{grooming_method}_{jet_types}_jet_pt_residual_mean_hybrid_{str(hybrid_jet_pt_bin)}"]
        # Select in hybrid jet pt during conversion.
        # NOTE: We need to use bh to do the sum and projection because it's a profile hist, which requires extra care.
        h = binned_data.BinnedData.from_existing_data(bh_hist)

        # Plot.
        # The values are scaled by the bin centers as a proxy for the true jet pt. Since it's a steeply falling spectra,
        # the bin centers are a bit too large, but it's close enough.
        # Mean is just the values
        ax_mean.errorbar(
            h.axes[0].bin_centers,
            h.values / h.axes[0].bin_centers,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=label,
            alpha=0.8,
            color=color,
        )
        # Width is the errors.
        ax_width.errorbar(
            h.axes[0].bin_centers,
            h.errors / h.axes[0].bin_centers,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=label,
            alpha=0.8,
            color=color,
        )

    # Labeling
    # Individual labeling
    ax_mean.set_ylabel(r"$(p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{part}}) / p_{\text{T}}^{\text{part}}$")
    ax_mean.set_ylim([-1, 2])
    ax_width.set_ylabel(
        r"$\sigma(p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{part}}) / p_{\text{T}}^{\text{part}}$"
    )
    ax_width.set_ylim([0, 0.5])
    # Shared
    for a, f in [(ax_mean, fig_mean), (ax_width, fig_width)]:
        text = "Iterative splittings"
        text += "\n" + f"${hybrid_jet_pt_bin.display_str(label='hybrid')}$"
        text += "\n" + " ".join(grooming_method.split("_")).capitalize()
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
        a.legend(frameon=False, loc="center right", fontsize=14)
        f.tight_layout()
    # Needs extra spacing on the left because the axis goes negative.
    fig_mean.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.13,
        bottom=0.12,
        right=0.99,
        top=0.98,
    )
    fig_width.subplots_adjust(
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
    filename = f"{plot_config.name}_hybrid_{str(hybrid_jet_pt_bin)}_iterative_splittings_{grooming_method}"
    fig_mean.savefig(output_dir / f"{filename}_mean.pdf")
    plt.close(fig_mean)
    fig_width.savefig(output_dir / f"{filename}_width.pdf")
    plt.close(fig_width)


def _plot_jet_pt_residual_distribution(
    hists: Dict[str, bh.Histogram],
    grooming_method: str,
    true_jet_pt_bin: helpers.RangeSelector,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    """ Plot the full jet pt residual for a pt true selection.

    Note:
        The pt true selections was applied when filling. This just plots the values.
    """
    logger.debug(f"Plotting jet pt residual distribution for {grooming_method} with true jet pt {true_jet_pt_bin}")

    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colors are to match the jet substructure semi-central AN
    for jet_types, label, color in [
        ("hybrid_det", "Flucutations", "red"),
        ("hybrid_true", "Combined", "black"),
        ("det_true", "Detector", "blue"),
    ]:
        bh_hist = hists[f"{grooming_method}_{jet_types}_jet_pt_residual_distribution"]
        # NOTE: We need to use bh to do the sum and projection because it's a profile hist, which requires extra care.
        selection = slice(bh.loc(true_jet_pt_bin.min + 0.0001), bh.loc(true_jet_pt_bin.max), bh.sum)
        h = binned_data.BinnedData.from_existing_data(bh_hist[selection, :])

        # Normalize
        h /= np.sum(h.values)

        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=label,
            color=color,
        )

    # Labeling
    text = "Iterative splittings"
    text += "\n" + f"${true_jet_pt_bin.display_str(label='part')}$"
    text += "\n" + " ".join(grooming_method.split("_")).capitalize()
    ax.text(
        0.97,
        0.97,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax.set_xlabel(plot_config.x_label)
    ax.set_ylabel(plot_config.y_label)
    ax.set_ylim([-0.02, 0.32])
    ax.legend(frameon=False, loc="upper left", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(
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
    filename = f"{plot_config.name}_true_{str(true_jet_pt_bin)}_iterative_splittings_{grooming_method}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_residuals(hists: Dict[str, bh.Histogram], grooming_methods: Sequence[str], output_dir: Path) -> None:
    hybrid_jet_pt_bins = [helpers.RangeSelector(40, 120), helpers.RangeSelector(20, 200)]
    true_jet_pt_bin = helpers.RangeSelector(40, 60)
    for grooming_method in grooming_methods:
        for hybrid_jet_pt_bin in hybrid_jet_pt_bins:
            _plot_residual_mean_and_width(
                hists=hists,
                grooming_method=grooming_method,
                hybrid_jet_pt_bin=hybrid_jet_pt_bin,
                plot_config=PlotConfig(
                    name="jet_pt_residual",
                    x_label=r"$p_{\text{T}}^{\text{part}}\:(\text{GeV}/c)$",
                    # Will be set individually during plotting of mean, width
                    y_label="IGNORE",
                ),
                output_dir=output_dir,
            )

        _plot_jet_pt_residual_distribution(
            hists=hists,
            grooming_method=grooming_method,
            true_jet_pt_bin=true_jet_pt_bin,
            plot_config=PlotConfig(
                name="jet_pt_residual_distribution",
                x_label=r"$(p_{\text{T}}^{\text{hybrid}} - p_{\text{T}}^{\text{det}}) / p_{\text{T}}^{\text{det}}$",
                y_label="",
            ),
            output_dir=output_dir,
        )


def _plot_response_by_matching_type(
    hists: Dict[str, bh.Histogram],
    label: str,
    grooming_method: str,
    matching_types: Sequence[str],
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:

    for matching_type in matching_types:
        logger.debug(f"Plotting {label} response for {grooming_method}, {matching_type}")

        matches_label = " ".join(matching_type.split("_")).capitalize()
        bh_input_hist = hists[f"{grooming_method}_hybrid_det_{label}_response_matching_type_{matching_type}"]
        h_input = binned_data.BinnedData.from_existing_data(bh_input_hist)

        # Select the variables (for the example of kt)
        # Axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt
        # NOTE: We already applied the 40 < hybrid jet pt < 120 cut, so it doesn't need an additional selection.
        h = binned_data.BinnedData(
            axes=[h_input.axes[1], h_input.axes[3]],
            values=np.sum(h_input.values, axis=(0, 2)),
            variances=np.sum(h_input.variances, axis=(0, 2)),
        )

        # Normalize the response.
        normalization_values = h.values.sum(axis=0, keepdims=True)
        h.values = np.divide(
            h.values, normalization_values, out=np.zeros_like(h.values), where=normalization_values != 0
        )

        # Finish setup
        fig, ax = plt.subplots(figsize=(8, 6))

        # Determine the normalization range
        z_axis_range = {
            # "vmin": h_proj.values[h_proj.values > 0].min(),
            "vmin": 1e-4,
            "vmax": h.values.max(),
        }

        # Plot
        mesh = ax.pcolormesh(
            h.axes[0].bin_edges.T, h.axes[1].bin_edges.T, h.values.T, norm=matplotlib.colors.LogNorm(**z_axis_range),
        )
        fig.colorbar(mesh, pad=0.02)

        # Labeling
        text = "Iterative splittings"
        text += "\n" + f"${helpers.RangeSelector(40, 120).display_str(label='hybrid')}$"
        text += "\n" + " ".join(grooming_method.split("_")).capitalize()
        text += "\n" + matches_label + " matches"
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            multialignment="left",
        )

        # Presentation
        ax.set_xlabel(plot_config.x_label)
        ax.set_ylabel(plot_config.y_label)
        fig.tight_layout()
        fig.subplots_adjust(
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
        filename = f"{plot_config.name}_iterative_splittings_{grooming_method}_matching_type_{matching_type}"
        fig.savefig(output_dir / f"{filename}.pdf")
        plt.close(fig)


def plot_response_by_matching_type(
    hists: Dict[str, bh.Histogram], grooming_methods: Sequence[str], matching_types: Sequence[str], output_dir: Path
) -> None:
    for grooming_method in grooming_methods:
        _plot_response_by_matching_type(
            hists=hists,
            label="kt",
            grooming_method=grooming_method,
            matching_types=matching_types,
            plot_config=PlotConfig(
                name="response_kt_hybrid_detector",
                x_label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{{GeV}}/c)$",
                y_label=r"$k_{\text{T}}^{\text{det}}\:(\text{{GeV}}/c)$",
            ),
            output_dir=output_dir,
        )
        _plot_response_by_matching_type(
            hists=hists,
            label="delta_R",
            grooming_method=grooming_method,
            matching_types=matching_types,
            plot_config=PlotConfig(
                name="response_delta_R_hybrid_detector", x_label=r"$R^{\text{hybrid}}$", y_label=r"$R^{\text{det}}$",
            ),
            output_dir=output_dir,
        )
