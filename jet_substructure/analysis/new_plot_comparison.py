""" New methods for plotting substructure comparison from the skim output.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
from pachyderm import binned_data

import jet_substructure.analysis.plot_base as pb
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


def plot_compare_grooming_methods_for_attribute(
    hists: Mapping[str, bh.Histogram],
    grooming_methods: Sequence[str],
    attr_name: str,
    prefix: str,
    jet_pt_bin: helpers.JetPtRange,
    set_zero_to_nan: bool,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    fig: Optional[matplotlib.figure.Figure] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    plot_png: Optional[bool] = False,
) -> str:
    logger.info(f"Plotting grooming method comparison for {attr_name}")

    passed_mpl_fig = True
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        passed_mpl_fig = False

    grooming_styling = pb.define_grooming_styles()

    for grooming_method in grooming_methods:
        # Setup
        style = grooming_styling[grooming_method]

        # Axes: jet_pt, attr_name
        logger.debug(f"Looking at hist: {grooming_method}_{prefix}_{attr_name}")
        bh_hist = hists[f"{grooming_method}_{prefix}_{attr_name}"]
        # Need to project to just the attr of interest.
        h = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(jet_pt_bin.min) : bh.loc(jet_pt_bin.max) : bh.sum, :]  # noqa: E203
        )

        # Normalize
        # Normalize by the sum of the values to get the n_jets values.
        # Then, we still need to normalize by the bin widths.
        h /= np.sum(h.values)
        h /= h.axes[0].bin_widths

        # Set 0s to NaN (for example, in z_g where have a good portion of the range cut off).
        if set_zero_to_nan:
            h.errors[h.values == 0] = np.nan
            h.values[h.values == 0] = np.nan

        # Plot options
        kwargs = {
            "markerfacecolor": "white" if style.fillstyle == "none" else style.color,
            "alpha": 1 if style.fillstyle == "none" else 0.8,
        }
        if style.fillstyle != "none":
            kwargs["markeredgewidth"] = 0

        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            color=style.color,
            marker=style.marker,
            fillstyle=style.fillstyle,
            linestyle="",
            label=style.label,
            zorder=style.zorder,
            **kwargs,
        )

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)

    # Store and cleanup
    # It's expected that the attr_name is already included in the `plot_config.name`.
    # Sanity check to make sure we don't get that wrong!
    # if attr_name not in plot_config.name:
    #    raise ValueError(
    #        f"PlotConfig name must contain the attr name! attr_name: {attr_name}, name: {plot_config.name}"
    #    )

    filename = f"{plot_config.name}_{jet_pt_bin}_iterative_splittings"
    fig.savefig(output_dir / f"{filename}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{filename}.png")

    if not passed_mpl_fig:
        plt.close(fig)

    return filename
