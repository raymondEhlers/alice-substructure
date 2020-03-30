#!/usr/bin/env python3

""" Plotting for the jet substructure analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import attr
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
from pachyderm import binned_data, histogram

from jet_substructure.analysis import substructure
from jet_substructure.base import analysis_objects, helpers


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


def _plot_distribution_old(
    results: Sequence[substructure.SubstructureResult],
    retrieve_values_func: Callable[[substructure.SubstructureResult], substructure.T_Array],
    jet_pt: substructure.T_Array,
    axis: bh.axis.Regular,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    label: PlotConfig,
    path: Path,
) -> None:
    # Validation
    path.mkdir(parents=True, exist_ok=True)

    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    for jet_pt_bin in jet_pt_bins:
        jet_pt_mask = jet_pt_bin.mask_array(jet_pt)
        # Number of jets includes untagged.
        n_jets = len(jet_pt[jet_pt_mask])
        if n_jets == 0:
            logger.warning(f"No jets within {jet_pt_bin.min}-{jet_pt_bin.max}. Skipping bin!")
            continue

        for result in results:
            logger.debug(f"Processing {jet_pt_bin}, {result.title}")
            # To get the jet pt mask to match the result, we need to apply the indices.
            # However, we can't apply them directly because the indices are jagged. By taking
            # those which have a count, we'll match the structure.
            jet_pt_substructure_result_mask = jet_pt_mask[result.indices.counts > 0]
            # TODO: Fix for nspliitings. See TEMP below.
            # TEMP - Should use the above.
            # This works for the splitting but nothing else because for the splittings we fill in 0,
            # but for the others, those are just empty entries.
            # jet_pt_substructure_result_mask = jet_pt_mask
            # ENDTEMP
            bh_hist = bh.Histogram(axis, storage=bh.storage.Weight())
            # print(f"{retrieve_values_func(result)}")
            # print(f"Values: {retrieve_values_func(result)[jet_pt_substructure_result_mask]}")
            bh_hist.fill(retrieve_values_func(result)[jet_pt_substructure_result_mask])
            h = histogram.Histogram1D(
                bin_edges=bh_hist.axes[0].edges,
                y=bh_hist.view().value,
                errors_squared=np.copy(bh_hist.view().variance),
            )
            # Scale by bin width
            h /= h.bin_widths
            # Normalize by number of jets
            h /= n_jets

            ax.errorbar(h.x, h.y, yerr=h.errors, xerr=h.bin_widths / 2, marker=".", linestyle="", label=result.title)

        # Labeling
        text = fr"${jet_pt_bin.min} < p_{{\text{{T}}}}^{{\text{{jet}}}} < {jet_pt_bin.max}$"
        ax.text(
            0.95,
            0.95,
            text,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            multialignment="right",
        )

        # Presentation
        ax.legend(frameon=False, loc="center right")
        ax.set_yscale("log")
        ax.set_xlabel(label.x_label)
        ax.set_ylabel(label.y_label)
        fig.tight_layout()

        # Store and reset
        fig.savefig(path / f"{label.name}_jetPt_{jet_pt_bin.min}_{jet_pt_bin.max}.pdf")
        ax.clear()

    plt.close(fig)


def _plot_distribution(
    attribute_name: str,
    hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    identifier: analysis_objects.Identifier,
    plot_config: PlotConfig,
    path: Path,
    ratio_denominator_hists: Optional[analysis_objects.Hists[analysis_objects.SubstructureHists]] = None,
) -> None:
    # Setup
    if ratio_denominator_hists:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
        ax, ax_ratio = axes
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    logger.info(f"Plotting {attribute_name}, {identifier}{', ratio' if ratio_denominator_hists else ''}")

    for technique, technique_hists in hists:
        # It will fail for 0 jets, so skip it
        if technique_hists.n_jets == 0:
            logger.warning(f"No jets within {identifier}_{technique}. Skipping bin!")
            continue
        # We don't want to include inclusive in the comparison at the moment.
        if technique == "inclusive":
            continue

        h: Union[bh.Histogram, binned_data.BinnedData] = getattr(technique_hists, attribute_name)
        h = binned_data.BinnedData.from_existing_data(h)

        # TODO: Should the hists be normalized earlier??
        # Scale by bin widths and number of jets
        h /= h.axis.bin_widths
        h /= technique_hists.n_jets

        ax.errorbar(
            h.axis.bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axis.bin_widths / 2,
            marker=".",
            linestyle="",
            label=technique_hists.title,
        )

        if ratio_denominator_hists:
            ratio_denominator = getattr(ratio_denominator_hists, technique)

            h_denominator: Union[bh.Histogram, binned_data.BinnedData] = getattr(ratio_denominator, attribute_name)
            h_denominator = binned_data.BinnedData.from_existing_data(h_denominator)

            h_denominator /= ratio_denominator.n_jets
            h_denominator /= h_denominator.axis.bin_widths

            # Don't apply any further normalization! We want the direct ratio of the values!
            h_ratio = h / h_denominator

            logger.warning(f"Ratio integral: {np.sum(h_ratio.values * h_ratio.axis.bin_widths)}")

            # Plot the ratio
            ax_ratio.errorbar(
                h_ratio.axis.bin_centers,
                h_ratio.values,
                yerr=h_ratio.errors,
                xerr=h_ratio.axis.bin_widths / 2,
                marker=".",
                linestyle="",
                alpha=0.6,
            )

    # Labeling
    text = identifier.display_str()
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax.legend(frameon=False, loc=plot_config.legend_location)
    if plot_config.log_y:
        ax.set_yscale("log")
    ax.set_ylabel(plot_config.y_label)
    if ratio_denominator_hists:
        ax_ratio.set_xlabel(plot_config.x_label)
        ax_ratio.set_ylabel("Recur./Iter.")
        # As standard for a ratio.
        ax_ratio.set_ylim([0, 2])
    else:
        ax.set_xlabel(plot_config.x_label)
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.12,
        bottom=0.11,
        right=0.98,
        top=0.98,
    )
    fig.align_ylabels()

    # Store and cleanup
    filename = f"{plot_config.name}_{str(identifier)}"
    if ratio_denominator_hists:
        filename = f"{filename}_ratio"
    fig.savefig(path / f"{filename}.pdf")
    plt.close(fig)


def _plot_total_number_of_splittings(
    masked_recursive_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    masked_iterative_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    identifier: analysis_objects.Identifier,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    ax, ax_ratio = axes
    logger.info(f"Plotting total_number_of_splittings, {identifier} with ratio")

    recursive_hists = masked_recursive_hists.inclusive
    h: Union[bh.Histogram, binned_data.BinnedData] = recursive_hists.total_number_of_splittings
    h = binned_data.BinnedData.from_existing_data(h)

    # Scale by bin widths and number of jets
    h /= h.axis.bin_widths
    h /= recursive_hists.n_jets

    ax.errorbar(
        h.axis.bin_centers,
        h.values,
        yerr=h.errors,
        xerr=h.axis.bin_widths / 2,
        marker=".",
        linestyle="",
        label="Recursive",
    )

    # Plot iterative splittings
    iterative_hists = masked_iterative_hists.inclusive
    h_iterative: Union[bh.Histogram, binned_data.BinnedData] = iterative_hists.total_number_of_splittings
    h_iterative = binned_data.BinnedData.from_existing_data(h_iterative)

    # Scale by bin widths and number of jets
    h_iterative /= h_iterative.axis.bin_widths
    h_iterative /= iterative_hists.n_jets

    ax.errorbar(
        h_iterative.axis.bin_centers,
        h_iterative.values,
        yerr=h_iterative.errors,
        xerr=h_iterative.axis.bin_widths / 2,
        marker=".",
        linestyle="",
        label="Iterative",
    )

    # Ratio
    # Don't apply any further normalization! We want the direct ratio of the values!
    h_ratio = h / h_iterative

    logger.warning(f"Ratio integral: {np.sum(h_ratio.values * h_ratio.axis.bin_widths)}")

    # Plot the ratio
    ax_ratio.errorbar(
        h_ratio.axis.bin_centers,
        h_ratio.values,
        yerr=h_ratio.errors,
        xerr=h_ratio.axis.bin_widths / 2,
        marker=".",
        linestyle="",
        alpha=0.6,
    )

    # Labeling
    text = identifier.display_str()
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax.legend(frameon=False, loc=plot_config.legend_location)
    if plot_config.log_y:
        ax.set_yscale("log")
    ax.set_ylabel(plot_config.y_label)
    ax_ratio.set_xlabel(plot_config.x_label)
    ax_ratio.set_ylabel("Recur./Iter.")
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.12,
        bottom=0.11,
        right=0.98,
        top=0.98,
    )
    fig.align_ylabels()

    # Store and cleanup
    filename = f"{plot_config.name}_inclusive_{str(identifier)}_ratio"
    fig.savefig(path / f"{filename}.pdf")
    plt.close(fig)


def _plot_lund_plane(
    technique: str, identifier: analysis_objects.Identifier, hists: analysis_objects.SubstructureHists, path: Path,
) -> None:
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))
    logger.info(f"Plotting lund plane for {technique}, {identifier}")

    h: Union[bh.Histogram, binned_data.BinnedData] = hists.lund_plane
    if isinstance(h, bh.Histogram):
        h = binned_data.BinnedData.from_existing_data(h)

    # TODO: Should the hists be normalized earlier??
    # Scale by bin width
    x_bin_widths, y_bin_widths = np.meshgrid(*h.axes.bin_widths)
    bin_widths = x_bin_widths * y_bin_widths
    # print(f"x_bin_widths: {x_bin_widths.size}")
    # print(f"y_bin_widths: {y_bin_widths.size}")
    # print(f"bin_widths size: {bin_widths.size}")
    h /= bin_widths
    # Scale by njets.
    h /= hists.n_jets

    # Determine the normalization range
    z_axis_range = {
        "vmin": h.values[h.values > 0].min(),
        "vmax": h.values.max(),
    }
    if technique == "inclusive":
        z_axis_range = {
            "vmin": 10e-3,
            "vmax": 5,
        }

    # Make the plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T, h.axes[1].bin_edges.T, h.values.T, norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Labeling
    text = identifier.display_str()
    text += "\n" + hists.title
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax.set_xlabel(r"$\log{(1/\Delta R)}$")
    ax.set_ylabel(r"$\log{(k_{\text{T}})}$")
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.10,
        bottom=0.11,
        right=0.99,
        top=0.98,
    )

    # Store and reset
    fig.savefig(path / f"lund_plane_{str(identifier)}_{technique}.pdf")
    plt.close(fig)


def lund_plane(
    all_hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]], path: Path
) -> None:
    # Validation
    path.mkdir(parents=True, exist_ok=True)

    # Plot labels
    kt_label = PlotConfig(
        name="kt",
        x_label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
        legend_location="lower left",
    )
    z_label = PlotConfig(
        name="z", x_label=r"$z$", y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}z$", legend_location="lower right",
    )
    delta_R_label = PlotConfig(
        name="delta_R",
        x_label=r"$R$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}R$",
        legend_location="lower right",
    )
    theta_label = PlotConfig(
        name="theta",
        x_label=r"$\theta$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\theta$",
        legend_location="lower right",
    )
    splitting_number_label = PlotConfig(
        name="splitting_number",
        x_label=r"$n$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n$",
        legend_location="center right",
        log_y=False,
    )
    # NOTE: Assumes fixed value of kt > 5!!
    splitting_number_perturbative_label = PlotConfig(
        name="splitting_number_kt_greater_than_5",
        x_label=r"$n$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n$ ($k_{\text{T}} > 5$)",
        legend_location="center right",
        log_y=False,
    )
    total_number_of_splittings_label = PlotConfig(
        name="total_number_of_splittings",
        x_label=r"$n_{\text{total}}$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n_{\text{total}}$",
        legend_location="center right",
        log_y=False,
    )

    distributions: List[Tuple[str, PlotConfig]] = [
        ("kt", kt_label),
        ("z", z_label),
        ("delta_R", delta_R_label),
        ("theta", theta_label),
        ("splitting_number", splitting_number_label),
        ("splitting_number_perturbative", splitting_number_perturbative_label),
        # total number of splittings is intentionally _not_ included because we only case about
        # the inclusive case.
        # ("total_number_of_splittings", total_number_of_splittings_label),
    ]
    for identifier, masked_hists in all_hists.items():
        for attribute_name, plot_config in distributions:
            _plot_distribution(
                attribute_name=attribute_name,
                hists=masked_hists,
                identifier=identifier,
                plot_config=plot_config,
                path=path,
            )
            if not identifier.iterative_splittings:
                # Plot the ratio of recursive to iterative
                _plot_distribution(
                    attribute_name=attribute_name,
                    hists=masked_hists,
                    identifier=identifier,
                    plot_config=plot_config,
                    path=path,
                    ratio_denominator_hists=all_hists[
                        analysis_objects.Identifier(iterative_splittings=True, jet_pt_bin=identifier.jet_pt_bin)
                    ],
                )

        # Plot the inclusive case for total number of splittings.
        # We directly compare the iterative and recursive cases, so we only plot once and show both sets of values.
        if not identifier.iterative_splittings:
            # It will fail for 0 jets, so skip it
            if masked_hists.inclusive.n_jets != 0:
                _plot_total_number_of_splittings(
                    masked_recursive_hists=masked_hists,
                    masked_iterative_hists=all_hists[
                        analysis_objects.Identifier(iterative_splittings=True, jet_pt_bin=identifier.jet_pt_bin)
                    ],
                    identifier=identifier,
                    plot_config=total_number_of_splittings_label,
                    path=path,
                )
            else:
                logger.warning(f"No jets within {identifier}_total_number_of_splittings. Skipping bin!")

        for technique, hists in masked_hists:
            if hists.n_jets == 0:
                logger.warning(f"No jets within {identifier}_{technique}. Skipping bin!")
                continue
            # Plot Lund Plane
            _plot_lund_plane(technique=technique, identifier=identifier, hists=hists, path=path)


def _plot_matching(
    technique: str,
    identifier: analysis_objects.Identifier,
    hists: analysis_objects.SubstructureMatchingSubjetHists,
    path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)

    # NOTE: We convert the hists here to ensure that we're working with copies!
    normalization = binned_data.BinnedData.from_existing_data(hists.all)

    # Both correctly tagged goes in the upper left
    both_correct = binned_data.BinnedData.from_existing_data(hists.both_correct)
    both_correct /= normalization
    axes[0, 0].errorbar(
        both_correct.axes[0].bin_centers,
        both_correct.values,
        yerr=both_correct.errors,
        xerr=both_correct.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    # Skip this panel
    axes[0, 1].set_visible(False)
    # Leading wasn't tagged, but subleading was correctly tagged as subleading.
    leading_failed_subleading_correct = binned_data.BinnedData.from_existing_data(
        hists.leading_failed_subleading_correct
    )
    leading_failed_subleading_correct /= normalization
    axes[0, 2].errorbar(
        leading_failed_subleading_correct.axes[0].bin_centers,
        leading_failed_subleading_correct.values,
        yerr=leading_failed_subleading_correct.errors,
        xerr=leading_failed_subleading_correct.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    # Skip this panel
    axes[1, 0].set_visible(False)
    # Swapped (ie. reversed)
    reversed = binned_data.BinnedData.from_existing_data(hists.reversed)
    reversed /= normalization
    axes[1, 1].errorbar(
        reversed.axes[0].bin_centers,
        reversed.values,
        yerr=reversed.errors,
        xerr=reversed.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    # Leading failed, subleading mistag
    leading_failed_subleading_mistag = binned_data.BinnedData.from_existing_data(hists.leading_failed_subleading_mistag)
    leading_failed_subleading_mistag /= normalization
    axes[1, 2].errorbar(
        leading_failed_subleading_mistag.axes[0].bin_centers,
        leading_failed_subleading_mistag.values,
        yerr=leading_failed_subleading_mistag.errors,
        xerr=leading_failed_subleading_mistag.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    # Leading correct, subleading failed
    leading_correct_subleading_failed = binned_data.BinnedData.from_existing_data(
        hists.leading_correct_subleading_failed
    )
    leading_correct_subleading_failed /= normalization
    axes[2, 0].errorbar(
        leading_correct_subleading_failed.axes[0].bin_centers,
        leading_correct_subleading_failed.values,
        yerr=leading_correct_subleading_failed.errors,
        xerr=leading_correct_subleading_failed.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    # Leading mistag, subleading failed
    leading_mistag_subleading_failed = binned_data.BinnedData.from_existing_data(hists.leading_mistag_subleading_failed)
    leading_mistag_subleading_failed /= normalization
    axes[2, 1].errorbar(
        leading_mistag_subleading_failed.axes[0].bin_centers,
        leading_mistag_subleading_failed.values,
        yerr=leading_mistag_subleading_failed.errors,
        xerr=leading_mistag_subleading_failed.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    # Both failed
    both_failed = binned_data.BinnedData.from_existing_data(hists.both_failed)
    both_failed /= normalization
    axes[2, 2].errorbar(
        both_failed.axes[0].bin_centers,
        both_failed.values,
        yerr=both_failed.errors,
        xerr=both_failed.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )

    # Set range
    for ax in axes.flat:
        ax.set_ylim([0, 1.2])

    # Labeling
    text = identifier.display_str()
    text += "\n" + hists.title
    axes[0, 1].text(
        0.95,
        0.95,
        text,
        transform=axes[0, 1].transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    # Axis labels
    axes[0, 0].set_ylabel("Subleading correct")
    axes[1, 0].set_ylabel("Subleading in leading")
    axes[2, 0].set_ylabel("Subleading in no prong")
    axes[2, 0].set_xlabel("Leading correct")
    axes[2, 1].set_xlabel("Leading in subleading")
    axes[2, 2].set_xlabel("Leading in no prong")
    # ax.set_xlabel(r"$\log{(1/\Delta R)}$")
    # ax.set_ylabel(r"$\log{(k_{\text{T}})}$")
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.10,
        bottom=0.10,
        right=0.99,
        top=0.99,
    )

    # Store and reset
    fig.savefig(path / f"subjet_matching_{identifier.iterative_splittings_label}_splittngs_{technique}.pdf")
    plt.close(fig)


def matching(
    all_matching_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
    ],
    path: Path,
) -> None:
    # Validation
    path.mkdir(parents=True, exist_ok=True)

    for identifier, matching_hists in all_matching_hists.items():
        for technique, hists in matching_hists:
            # Plot matching distributions
            _plot_matching(technique=technique, identifier=identifier, hists=hists, path=path)


def _plot_toy(
    technique: str,
    identifier: analysis_objects.Identifier,
    attribute_name: str,
    hists: analysis_objects.SubstructureToyHists,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))
    logger.info(f"Plotting toy hist for {technique}, {identifier}, {attribute_name}")

    h: Union[bh.Histogram, binned_data.BinnedData] = getattr(hists, attribute_name)
    if isinstance(h, bh.Histogram):
        h = binned_data.BinnedData.from_existing_data(h)

    # TODO: Should the hists be normalized earlier??
    # Scale by bin width
    x_bin_widths, y_bin_widths = np.meshgrid(*h.axes.bin_widths)
    bin_widths = x_bin_widths * y_bin_widths
    # print(f"x_bin_widths: {x_bin_widths.size}")
    # print(f"y_bin_widths: {y_bin_widths.size}")
    # print(f"bin_widths size: {bin_widths.size}")
    h /= bin_widths
    # Scale by njets.
    h /= hists.n_jets

    # Determine the normalization range
    z_axis_range = {
        "vmin": h.values[h.values > 0].min(),
        "vmax": h.values.max(),
    }
    # if technique == "inclusive":
    #    z_axis_range = {
    #        "vmin": 10e-3,
    #        "vmax": 5,
    #    }

    # Make the plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T, h.axes[1].bin_edges.T, h.values.T, norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Labeling
    text = identifier.display_str(jet_pt_label="hybrid")
    text += "\n" + hists.title
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
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
        bottom=0.11,
        right=0.99,
        top=0.98,
    )

    # Store and reset
    fig.savefig(path / f"{attribute_name}_{str(identifier)}_{technique}.pdf")
    plt.close(fig)


def toy(
    all_toy_hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureToyHists]],
    path: Path,
) -> None:
    # Validation
    path.mkdir(parents=True, exist_ok=True)

    # Plot labels
    kt_label = PlotConfig(
        name="kt",
        x_label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$",
        y_label=r"$k_{\text{T}}^{\text{first splitting}}\:(\text{GeV}/c)$",
    )
    z_label = PlotConfig(
        name="z",
        x_label=r"$z^{\text{hybrid}}\:(\text{GeV}/c)$",
        y_label=r"$z^{\text{first splitting}}\:(\text{GeV}/c)$",
    )
    delta_R_label = PlotConfig(
        name="delta_R",
        x_label=r"$R^{\text{hybrid}}\:(\text{GeV}/c)$",
        y_label=r"$R^{\text{first splitting}}\:(\text{GeV}/c)$",
    )
    theta_label = PlotConfig(
        name="theta",
        x_label=r"$\theta^{\text{hybrid}}\:(\text{GeV}/c)$",
        y_label=r"$\theta^{\text{first splitting}}\:(\text{GeV}/c)$",
    )

    distributions: List[Tuple[str, PlotConfig]] = [
        ("kt", kt_label),
        ("z", z_label),
        ("delta_R", delta_R_label),
        ("theta", theta_label),
    ]

    for identifier, toy_hists in all_toy_hists.items():
        for technique, hists in toy_hists:
            if hists.n_jets == 0:
                logger.warning(f"No jets within {identifier}_{technique}. Skipping bin!")
                continue
            for attribute_name, plot_config in distributions:
                # Plot toy distributions
                _plot_toy(
                    technique=technique,
                    identifier=identifier,
                    attribute_name=attribute_name,
                    hists=hists,
                    plot_config=plot_config,
                    path=path,
                )
