#!/usr/bin/env python3

""" Plotting for the jet substructure analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from functools import partial
from pathlib import Path
from typing import Callable, Sequence, Tuple

import attr
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pachyderm.plot
from pachyderm import histogram

from jet_substructure.base import helpers
from jet_substructure.analysis import substructure

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
class PlotLabel:
    name: str = attr.ib()
    x_label: str = attr.ib()
    y_label: str = attr.ib()

def _retrieve_kt(result: substructure.SubstructureResult) -> substructure.T_Array:
    return result.kt

def _retrieve_z(result: substructure.SubstructureResult) -> substructure.T_Array:
    return result.z

def _retrieve_delta_R(result: substructure.SubstructureResult) -> substructure.T_Array:
    return result.delta_R

def _retrieve_theta(result: substructure.SubstructureResult, jet_R: float) -> substructure.T_Array:
    return result.delta_R / jet_R

def _splitting_number(result: substructure.SubstructureResult) -> substructure.T_Array:
    return result.splitting_number

def _plot_distribution(results: Sequence[substructure.SubstructureResult], retrieve_values_func: Callable[[substructure.SubstructureResult], substructure.T_Array], jet_pt: substructure.T_Array, axis: bh.axis.Regular, jet_pt_bins: Sequence[helpers.RangeSelector], label: PlotLabel, path: Path) -> None:
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
            bh_hist = bh.Histogram(axis, storage=bh.storage.Weight())
            #print(f"Values: {retrieve_values_func(result)[jet_pt_substructure_result_mask]}")
            bh_hist.fill(retrieve_values_func(result)[jet_pt_substructure_result_mask])
            h = histogram.Histogram1D(
                bin_edges = bh_hist.axes[0].edges,
                y = bh_hist.view().value,
                errors_squared = np.copy(bh_hist.view().variance),
            )
            # Scale by bin width
            h /= h.bin_widths
            # Normalize by number of jets
            h /= n_jets

            ax.errorbar(h.x, h.y, yerr=h.errors, xerr=h.bin_widths / 2,
                        marker=".", linestyle="", label=result.title)

        # Labeling
        text = fr"${jet_pt_bin.min} < p_{{\text{{T}}}}^{{\text{{jet}}}} < {jet_pt_bin.max}$"
        ax.text(0.95, 0.95, text,
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
                multialignment="right")

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

def kt(results: Sequence[substructure.SubstructureResult], jet_pt: substructure.T_Array, jet_pt_bins: Sequence[helpers.RangeSelector], path: Path) -> None:
    logger.info("Plotting kt")
    _plot_distribution(results = results,
                       retrieve_values_func=_retrieve_kt,
                       axis=bh.axis.Regular(50, 0, 25),
                       jet_pt = jet_pt, jet_pt_bins = jet_pt_bins,
                       label = PlotLabel(
                           name = "kt",
                           x_label = r"$k_{\text{T}}\:(\text{GeV}/c)$",
                           y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                       ),
                       path = path)

def z(results: Sequence[substructure.SubstructureResult], jet_pt: substructure.T_Array, jet_pt_bins: Sequence[helpers.RangeSelector], path: Path) -> None:
    logger.info("Plotting z")
    _plot_distribution(results = results,
                       retrieve_values_func=_retrieve_z,
                       axis=bh.axis.Regular(20, 0, 0.5),
                       jet_pt = jet_pt, jet_pt_bins = jet_pt_bins,
                       label = PlotLabel(
                           name = "z",
                           x_label = r"$z$",
                           y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}z$",
                       ),
                       path = path)

def delta_R(results: Sequence[substructure.SubstructureResult], jet_pt: substructure.T_Array, jet_pt_bins: Sequence[helpers.RangeSelector], path: Path) -> None:
    logger.info("Plotting delta R")
    _plot_distribution(results=results,
                       retrieve_values_func=_retrieve_delta_R,
                       axis=bh.axis.Regular(20, 0, 0.4),
                       jet_pt = jet_pt, jet_pt_bins = jet_pt_bins,
                       label = PlotLabel(
                           name = "delta_R",
                           x_label = r"$R$",
                           y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}R$",
                       ),
                       path = path)

def theta(results: Sequence[substructure.SubstructureResult], jet_R: float, jet_pt: substructure.T_Array, jet_pt_bins: Sequence[helpers.RangeSelector], path: Path) -> None:
    logger.info("Plotting theta")
    _plot_distribution(results = results,
                       retrieve_values_func=partial(_retrieve_theta, jet_R=jet_R),
                       axis=bh.axis.Regular(50, 0, 1),
                       jet_pt = jet_pt, jet_pt_bins = jet_pt_bins,
                       label = PlotLabel(
                           name = "theta",
                           x_label = r"$\theta$",
                           y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\theta$",
                       ),
                       path = path)

def splitting_number(results: Sequence[substructure.SubstructureResult], jet_R: float, jet_pt: substructure.T_Array, jet_pt_bins: Sequence[helpers.RangeSelector], path: Path) -> None:
    logger.info("Plotting splitting number")
    _plot_distribution(results = results,
                       retrieve_values_func=partial(_retrieve_theta, jet_R=jet_R),
                       axis=bh.axis.Regular(10, 0, 10),
                       jet_pt = jet_pt, jet_pt_bins = jet_pt_bins,
                       label = PlotLabel(
                           name = "splitting_number",
                           x_label = r"$n$",
                           y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n$",
                       ),
                       path = path)

def _retrieve_lund_plane(result: substructure.SubstructureResult, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.log(1. / result.delta_R[mask]), np.log(result.kt[mask])

def _plot_lund_plane_hist(results: Sequence[substructure.SubstructureResult],
                          retrieve_values_func: Callable[[substructure.SubstructureResult, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                          axes: Sequence[bh.axis.Regular],
                          jet_pt: substructure.T_Array, jet_pt_bins: Sequence[helpers.RangeSelector],
                          path: Path) -> None:
    for jet_pt_bin in jet_pt_bins:
        jet_pt_mask = jet_pt_bin.mask_array(jet_pt)
        # Number of jets includes untagged.
        n_jets = len(jet_pt[jet_pt_mask])
        if n_jets == 0:
            logger.warning(f"No jets within {jet_pt_bin.min}-{jet_pt_bin.max}. Skipping bin!")
            continue

        for result in results:
            logger.debug(f"Processing {jet_pt_bin}, {result.title}")
            fig, ax = plt.subplots(figsize=(8, 6))

            # To get the jet pt mask to match the result, we need to apply the indices.
            # However, we can't apply them directly because the indices are jagged. By taking
            # those which have a count, we'll match the structure.
            jet_pt_substructure_result_mask = jet_pt_mask[result.indices.counts > 0]

            bh_hist = bh.Histogram(*axes, storage=bh.storage.Weight())
            bh_hist.fill(*retrieve_values_func(result, jet_pt_substructure_result_mask))

            # TODO: Scaling!
            # Scale by bin width
            #h /= h.bin_widths
            # TODO: Scale by njets.
            #h /= n_jets
            x, y = bh_hist.axes.edges
            view = bh_hist.view()
            h = helpers.BinnedData2D(
                x_bin_edges = x, y_bin_edges = y,
                values = view.value,
                errors_squared = view.variance,
            )

            # Make the plot
            mesh = ax.pcolormesh(h.x_bin_edges.T, h.y_bin_edges.T, h.values.T,
                                 norm=matplotlib.colors.LogNorm(vmin=1, vmax=h.values.max()))
            fig.colorbar(mesh)

            # Labeling
            text = fr"${jet_pt_bin.min} < p_{{\text{{T}}}}^{{\text{{jet, part}}}} < {jet_pt_bin.max}$"
            text += "\n" + result.title
            ax.text(0.95, 0.95, text,
                    transform=ax.transAxes,
                    horizontalalignment="right",
                    verticalalignment="top",
                    multialignment="right")

            # Presentation
            ax.set_xlabel(r"$\log{(1/\Delta R)}$")
            ax.set_ylabel(r"$\log{(k_{\text{T}})}$")
            fig.tight_layout()

            # Store and reset
            fig.savefig(path / f"lund_plane_hist_jetPt_{jet_pt_bin.min}_{jet_pt_bin.max}_{result.name}.pdf")
            plt.close(fig)

def _plot_lund_plane_scatter(results: Sequence[substructure.SubstructureResult],
                             retrieve_values_func: Callable[[substructure.SubstructureResult, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                             jet_pt: substructure.T_Array, jet_pt_bins: Sequence[helpers.RangeSelector],
                             path: Path) -> None:
    # Use a scatter plot here
    for jet_pt_bin in jet_pt_bins:
        jet_pt_mask = jet_pt_bin.mask_array(jet_pt)
        # Number of jets includes untagged.
        n_jets = len(jet_pt[jet_pt_mask])
        if n_jets == 0:
            logger.warning(f"No jets within {jet_pt_bin.min}-{jet_pt_bin.max}. Skipping bin!")
            continue

        for result in results:
            fig, ax = plt.subplots(figsize=(8, 6))
            logger.debug(f"Processing {jet_pt_bin}, {result.title}")

            # To get the jet pt mask to match the result, we need to apply the indices.
            # However, we can't apply them directly because the indices are jagged. By taking
            # those which have a count, we'll match the structure.
            jet_pt_substructure_result_mask = jet_pt_mask[result.indices.counts > 0]

            x, y = retrieve_values_func(result, jet_pt_substructure_result_mask)

            # Make the plot
            mesh = ax.scatter(x, y, alpha = 0.2)

            # Labeling
            text = fr"${jet_pt_bin.min} < p_{{\text{{T}}}}^{{\text{{jet, part}}}} < {jet_pt_bin.max}$"
            text += "\n" + result.title
            ax.text(0.95, 0.95, text,
                    transform=ax.transAxes,
                    horizontalalignment="right",
                    verticalalignment="top",
                    multialignment="right")

            # Presentation
            ax.set_xlabel(r"$\log{(1/\Delta R)}$")
            ax.set_ylabel(r"$\log{(k_{\text{T}})}$")
            fig.tight_layout()

            # Store and reset
            fig.savefig(path / f"lund_plane_scatter_jetPt_{jet_pt_bin.min}_{jet_pt_bin.max}_{result.name}.pdf")
            plt.close(fig)

def lund_plane(results: Sequence[substructure.SubstructureResult], jet_pt: substructure.T_Array, jet_pt_bins: Sequence[helpers.RangeSelector], path: Path) -> None:
    # Validation
    path.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting Lund Plane hists")
    _plot_lund_plane_hist(results = results,
                          retrieve_values_func=_retrieve_lund_plane,
                          axes=[bh.axis.Regular(25, 0, 5),
                                bh.axis.Regular(25, -5.0, 5.0)],
                          jet_pt = jet_pt, jet_pt_bins = jet_pt_bins,
                          path = path)
    logger.info("Plotting Lund Plane scatter plots")
    _plot_lund_plane_scatter(results = results,
                             retrieve_values_func=_retrieve_lund_plane,
                             jet_pt = jet_pt, jet_pt_bins = jet_pt_bins,
                             path = path)
    # TODO: What about plotting _all_ of the splittings. Woudl I ever even want to do that??
