#!/usr/bin/env python3

""" Plotting for the jet substructure analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path
from typing import Sequence

import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pachyderm.plot
from pachyderm import histogram

from jet_substructure.base import helpers
from jet_substructure.analysis import substructure

pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True

def kt(results: Sequence[substructure.SubstructureResult], jet_pt: substructure.T_Array, jet_pt_bins: Sequence[helpers.RangeSelector], path: Path) -> None:
    # Validation
    path.mkdir(parents=True, exist_ok=True)

    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    for jet_pt_bin in jet_pt_bins:
        jet_pt_mask = jet_pt_bin.mask_array(jet_pt)
        # Number of jets includes untagged.
        n_jets = len(jet_pt[jet_pt_mask])
        for result in results:
            print(f"Processing {jet_pt_bin}, {result.title}")
            # To get the jet pt mask to match the result, we need to apply the indices.
            # However, we can't apply them directly because the indices are jagged. By taking
            # those which have a count, we'll match the structure.
            jet_pt_substructure_result_mask = jet_pt_mask[result.indices.counts > 0]
            bh_hist = bh.Histogram(bh.axis.Regular(90, 0, 45), storage=bh.storage.Weight())
            bh_hist.fill(result.kt[jet_pt_substructure_result_mask])
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
        ax.legend(frameon=False, loc="lower right")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k_{\text{T}}\:(\text{GeV}/c)$")
        ax.set_ylabel(r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$")
        fig.tight_layout()

        # Store and reset
        fig.savefig(path / f"kt_jetPt_{jet_pt_bin.min}_{jet_pt_bin.max}.pdf")
        ax.clear()

    plt.close()

