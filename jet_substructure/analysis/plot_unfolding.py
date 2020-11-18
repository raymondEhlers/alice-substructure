""" Plot unfolding.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

import attr
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import seaborn as sns
import uproot
from pachyderm import binned_data

from jet_substructure.analysis import plot_base as pb
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)

pachyderm.plot.configure()


def _efficiency_substructure_variable(
    hists: Mapping[str, binned_data.BinnedData], true_jet_pt_range: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Efficiency for the substructure variable.

    Note:
        Since we need a set of hists, we just pass all of them.

    Args:
        hists: Input histograms.
        true_jet_pt_range: True jet pt range over which we will integrate.
    Returns:
        Efficiency hist for the substructure variable.
    """
    # Assign them for convenience
    bh_cut_efficiency = hists["true"].to_boost_histogram()
    bh_full_efficiency = hists["truef"].to_boost_histogram()

    # Select true pt range.
    selection = slice(bh.loc(true_jet_pt_range.min), bh.loc(true_jet_pt_range.max), bh.sum)
    cut = binned_data.BinnedData.from_existing_data(bh_cut_efficiency[:, selection])
    full = binned_data.BinnedData.from_existing_data(bh_full_efficiency[:, selection])

    return cut / full


def _project_substructure_variable(
    input_hist: binned_data.BinnedData, jet_pt_range: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Project the hist to the substructure variable.

    Args:
        input_hist: Hist to be projected.
        jet_pt_range: True jet pt range over which we will integrate.
    Returns:
        The input hist projected onto the the substructure variable axis.
    """
    # For convenience
    bh_hist = input_hist.to_boost_histogram()

    selection = slice(bh.loc(jet_pt_range.min), bh.loc(jet_pt_range.max), bh.sum)
    return binned_data.BinnedData.from_existing_data(bh_hist[:, selection])


def _efficiency_pt(
    hists: Mapping[str, binned_data.BinnedData], true_substructure_variable_range: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Efficiency for the jet pt.

    Note:
        Since we need a set of hists, we just pass all of them.

    Args:
        hists: Input histograms.
        true_substructure_variable_range: True substructure variable range over which we will integrate.
    Returns:
        Efficiency hist for jet pt.
    """
    # For convenience
    bh_cut_efficiency = hists["true"].to_boost_histogram()
    bh_full_efficiency = hists["truef"].to_boost_histogram()

    # Select true pt range.
    selection = slice(
        bh.loc(true_substructure_variable_range.min), bh.loc(true_substructure_variable_range.max), bh.sum
    )
    cut = binned_data.BinnedData.from_existing_data(bh_cut_efficiency[selection, :])
    full = binned_data.BinnedData.from_existing_data(bh_full_efficiency[selection, :])

    return cut / full


def _project_jet_pt(
    input_hist: binned_data.BinnedData, substructure_variable_bin: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Project the hist to the jet pt.

    Args:
        input_hist: Hist to be projected.
        substructure_variable_range: True substructure variable range over which we will integrate.
    Returns:
        The input hist projected onto the the jet pt axis.
    """
    bh_hist = input_hist.to_boost_histogram()

    selection = slice(bh.loc(substructure_variable_bin.min), bh.loc(substructure_variable_bin.max), bh.sum)
    return binned_data.BinnedData.from_existing_data(bh_hist[selection, :])


def _normalize_unfolded(hist: binned_data.BinnedData, efficiency: binned_data.BinnedData) -> binned_data.BinnedData:
    """Normalized unfolded hist.

    This involves applying the efficiency and then normalizing by the integral and the bin width.

    Args:
        hist: Histogram to be normalized.
        efficiency: Efficiency histogram with the same binning as the input hist.
    Returns:
        The normalized histogram.
    """
    # Apply the efficiency.
    hist /= efficiency
    # Then normalize by the integral (sum) and bin width.
    hist /= np.sum(hist.values)
    hist /= hist.axes[0].bin_widths
    return hist


def _normalize_refolded(hist: binned_data.BinnedData) -> binned_data.BinnedData:
    """Normalize refolded hist.

    This involves normalizing by the integral and the bin width.

    Args:
        hist: Histogram to be normalized.
    Returns:
        The normalized histogram.
    """
    hist /= np.sum(hist.values)
    hist /= hist.axes[0].bin_widths
    return hist


def _smeared(
    hists: Mapping[str, binned_data.BinnedData],
    hist_name: str,
    projection_func: Callable[[binned_data.BinnedData, helpers.RangeSelector], binned_data.BinnedData],
    smeared_range_to_integrate_over: helpers.RangeSelector,
) -> binned_data.BinnedData:
    """Helper function to get a smeared hist along a desired axis.

    Args:
        hists: Input hists.
        hist_name: Name of the smeared histogram to retrieve.
        projection_func: Function to project the histogram along the desired axis.
        smeared_range_to_integrate_over: Smeared range over which we will integrate.
    Returns:
        The desired smeared histogram.
    """
    hist = projection_func(hists[hist_name], smeared_range_to_integrate_over)
    return _normalize_refolded(hist=hist)


def _unfolded(
    hists: Mapping[str, binned_data.BinnedData],
    hist_name: str,
    projection_func: Callable[[binned_data.BinnedData, helpers.RangeSelector], binned_data.BinnedData],
    efficiency_func: Callable[[Mapping[str, binned_data.BinnedData], helpers.RangeSelector], binned_data.BinnedData],
    true_range_to_integrate_over: helpers.RangeSelector,
) -> binned_data.BinnedData:
    """Helper function to get an unfolded hist along a desired axis.

    Args:
        hists: Input hists.
        hist_name: Name of the unfolded histogram to retrieve.
        projection_func: Function to project the histogram along the desired axis.
        true_range_to_integrate_over: True range over which we will integrate.
    Returns:
        The desired unfolded histogram.
    """
    # efficiency = efficiency_func(hists, true_bin)
    ## For convenience in normalizing.
    # _normalize_hist = functools.partial(_normalize_unfolded, efficiency=efficiency)
    hist = projection_func(hists[hist_name], true_range_to_integrate_over)
    # hist = _normalize_hist(hist)
    efficiency = efficiency_func(hists, true_range_to_integrate_over)
    return _normalize_unfolded(hist=hist, efficiency=efficiency)


@attr.s
class UnfoldingOutput:
    substructure_variable: str = attr.ib()
    grooming_method: str = attr.ib()
    smeared_var_range: helpers.RangeSelector = attr.ib()
    smeared_untagged_var: helpers.RangeSelector = attr.ib()
    smeared_jet_pt_range: helpers.JetPtRange = attr.ib()
    collision_system: str = attr.ib()
    base_dir: Path = attr.ib(converter=Path)
    smeared_input: bool = attr.ib(default=False)
    pure_matches: bool = attr.ib(default=False)
    suffix: str = attr.ib(default="")
    n_iter_compare: int = attr.ib(default=4)
    raw_hist_name: str = attr.ib(default="raw")
    smeared_hist_name: str = attr.ib(default="smeared")
    true_hist_name: str = attr.ib(default="true")
    hists: MutableMapping[str, binned_data.BinnedData] = attr.ib(factory=dict)

    def __attrs_post_init__(self) -> None:
        # Fully setup base dir.
        # NOTE: Added "parsl" for the newer output results.
        self.base_dir = self.base_dir / self.collision_system / "unfolding" / "parsl"

        # Initialize the file if the histograms aren't specified.
        if not self.hists:
            f = uproot.open(self.input_filename)
            for k in f.keys():
                hist_key = k.decode("utf-8")
                hist_key = hist_key[: hist_key.find(";")]
                self.hists[hist_key] = binned_data.BinnedData.from_existing_data(f[k])

    @property
    def identifier(self) -> str:
        name = f"{self.substructure_variable}_grooming_method_{self.grooming_method}"
        name += f"_smeared_{self.smeared_var_range}"
        # TEMP until fixed in the unfolding code...
        name += f"_untagged_{self.smeared_untagged_var}"
        # name += f"_untagged_{self.smeared_untagged_var.min}_{self.smeared_untagged_var.max}"
        # TEMP until fixed in the unfolding code...
        name += f"_smeared_{self.smeared_jet_pt_range}"
        # name += f"_smeared_jet_pt_{self.smeared_jet_pt_range.min}_{self.smeared_jet_pt_range.max}"
        # if self.smeared_input:
        #    name += "_hybrid_as_input"
        if self.pure_matches:
            name += "_pure_matches"
        if self.suffix:
            name += f"_{self.suffix}"
        return name

    @property
    def max_n_iter(self) -> int:
        try:
            return self._max_n_iter
        except AttributeError:
            n = 1
            for hist_name in self.hists:
                # We could equally use the unfolded.
                if "bayesian_folded_iter_" in hist_name:
                    # We add a +1 so we can use it easily with range(...).
                    n = max(n, int(hist_name.split("_")[-1]) + 1)
            self._max_n_iter: int = n
        return self._max_n_iter

    @property
    def input_filename(self) -> Path:
        return self.base_dir / f"unfolding_{self.identifier}.root"

    @property
    def output_dir(self) -> Path:
        p = self.base_dir / self.substructure_variable / self.identifier
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir_png(self) -> Path:
        return self.output_dir / "png"

    def unfolded_substructure(self, n_iter: int, true_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Helper to retrieve the unfolded substructure directly """
        return self.true_substructure(
            hist_name=f"bayesian_unfolded_iter_{n_iter}",
            true_jet_pt_range=true_jet_pt_range,
        )

    def true_substructure(self, hist_name: str, true_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Retrieve a true level substructure hist. """
        return _unfolded(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_substructure_variable,
            efficiency_func=_efficiency_substructure_variable,
            true_range_to_integrate_over=true_jet_pt_range,
        )

    def unfolded_jet_pt(
        self, n_iter: int, true_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        return self.true_jet_pt(
            hist_name=f"bayesian_unfolded_iter_{n_iter}",
            true_substructure_variable_range=true_substructure_variable_range,
        )

    def true_jet_pt(
        self, hist_name: str, true_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        return _unfolded(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_jet_pt,
            efficiency_func=_efficiency_pt,
            true_range_to_integrate_over=true_substructure_variable_range,
        )

    def refolded_substructure(self, n_iter: int, smeared_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Helper to retrieve the refolded substructure directly. """
        return self.smeared_substructure(
            hist_name=f"bayesian_folded_iter_{n_iter}",
            smeared_jet_pt_range=smeared_jet_pt_range,
        )

    def smeared_substructure(self, hist_name: str, smeared_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Retrieve a smeared substructure hist. """
        return _smeared(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_substructure_variable,
            smeared_range_to_integrate_over=smeared_jet_pt_range,
        )

    def refolded_jet_pt(
        self, n_iter: int, smeared_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        """ Helper to retrieve the refolded jet pt directly. """
        return self.smeared_jet_pt(
            hist_name=f"bayesian_folded_iter_{n_iter}",
            smeared_substructure_variable_range=smeared_substructure_variable_range,
        )

    def smeared_jet_pt(
        self, hist_name: str, smeared_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        """ Retrieve a smeared jet pt hist. """
        return _smeared(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_jet_pt,
            smeared_range_to_integrate_over=smeared_substructure_variable_range,
        )


@attr.s
class SingleResult:
    """ Container for a single unfolding result. """

    data: binned_data.BinnedData = attr.ib()
    n_iter: int = attr.ib()
    ranges: Sequence[helpers.RangeSelector] = attr.ib(factory=list)


def plot_relative_individual_systematics(
    unfolded: SingleResult,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    """Plot relative individual systematic errors."""
    import mplhep as hep

    # Setup
    logger.debug("Plotting systematic relative errors.")
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for name, systematic in unfolded.data.metadata["y_systematic"].items():
        hep.histplot(
            H=np.maximum(systematic.low, systematic.high) / unfolded.data.values,
            bins=unfolded.data.axes[0].bin_edges,
            # color=style.color,
            label=name.replace("_", " "),
            alpha=0.8,
        )

    # For comparison, add the statistical too
    hep.histplot(
        H=unfolded.data.errors / unfolded.data.values,
        bins=unfolded.data.axes[0].bin_edges,
        # color=style.color,
        label="Statistical (for comparison)",
        # marker="o",
        # linestyle="",
        alpha=0.8,
    )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    logger.info(f"Writing plot to {output_dir / figure_name}.pdf")
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)


def plot_systematic(
    unfolded: SingleResult,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    """Plot systematic

    Initial version to play around with.
    """
    # Setup
    logger.debug("Plotting systematic.")
    fig, ax = plt.subplots(figsize=(10, 8))
    # fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    # ax_upper, ax_ratio_iter, ax_ratio_true = axes

    ax.errorbar(
        unfolded.data.axes[0].bin_centers,
        unfolded.data.values,
        yerr=unfolded.data.errors,
        xerr=unfolded.data.axes[0].bin_widths / 2,
        # color=style.color,
        marker="o",
        linestyle="",
    )
    # Systematic
    pachyderm.plot.error_boxes(
        ax=ax,
        x_data=unfolded.data.axes[0].bin_centers,
        y_data=unfolded.data.values,
        x_errors=unfolded.data.axes[0].bin_widths / 2,
        y_errors=np.array(
            [
                unfolded.data.metadata["y_systematic"]["quadrature"].low,
                unfolded.data.metadata["y_systematic"]["quadrature"].high,
            ]
        ),
        # y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
        # color=style.color,
        # color=p[0].get_color(),
        linewidth=0,
        color="red",
    )

    # This isn't really right, but it's a first pass. Let's see...
    # Rmax06
    # pachyderm.plot.error_boxes(
    #    ax=ax,
    #    x_data=unfolded.data.axes[0].bin_centers,
    #    y_data=unfolded.data.values,
    #    x_errors=unfolded.data.axes[0].bin_widths / 2,
    #    y_errors=unfolded.data.metadata["y_systematic_Rmax06"],
    #    #y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
    #    #color=style.color,
    #    #color=p[0].get_color(),
    #    linewidth=0,
    #    label="RMax06",
    #    color="red",
    # )
    ## Tracking efficiency
    # pachyderm.plot.error_boxes(
    #    ax=ax,
    #    x_data=unfolded.data.axes[0].bin_centers,
    #    y_data=unfolded.data.values,
    #    x_errors=unfolded.data.axes[0].bin_widths / 2,
    #    y_errors=unfolded.data.metadata["y_systematic_tracking_efficiency"],
    #    #y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
    #    #color=style.color,
    #    #color=p[0].get_color(),
    #    linewidth=0,
    #    label="Tracking Eff.",
    #    color="green",
    # )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    logger.info(f"Writing plot to {output_dir / figure_name}.pdf")
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)


def plot_unfolded(
    unfolding_output: UnfoldingOutput,
    hist_true: binned_data.BinnedData,
    hist_n_iter_compare: binned_data.BinnedData,
    unfolded_hists: Sequence[binned_data.BinnedData],
    plot_config: pb.PlotConfig,
    plot_png: bool = False,
) -> None:
    """Plot unfolded."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 12),
        gridspec_kw={"height_ratios": [4, 1, 1]},
        sharex=True,
    )
    ax_upper, ax_ratio_iter, ax_ratio_true = axes

    for i, hist in enumerate(unfolded_hists, start=1):
        ax_upper.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            yerr=hist.errors,
            label=f"Bayes {i}",
            marker="o",
            linestyle="",
            alpha=0.8,
        )

        # Plot ratio with selected iter (in principle could also be with true, but now it's
        # not necessary because we have another panel with the true).
        ratio = hist / hist_n_iter_compare
        ax_ratio_iter.errorbar(
            ratio.axes[0].bin_centers,
            ratio.values,
            xerr=ratio.axes[0].bin_widths / 2,
            yerr=ratio.errors,
            marker="o",
            linestyle="",
            alpha=0.8,
        )

        # Plot ratio with true
        ratio_true = hist / hist_true
        ax_ratio_true.errorbar(
            ratio_true.axes[0].bin_centers,
            ratio_true.values,
            xerr=ratio_true.axes[0].bin_widths / 2,
            yerr=ratio_true.errors,
            marker="o",
            linestyle="",
            alpha=0.8,
        )

    # Cross check.
    # Plot truth
    ax_upper.errorbar(
        hist_true.axes[0].bin_centers,
        hist_true.values,
        xerr=hist_true.axes[0].bin_widths / 2,
        yerr=hist_true.errors,
        label="True",
        marker="o",
        linestyle="",
        color="black",
        alpha=0.8,
    )
    ## And the ratio too
    # ratio = hist_true / h_ratio_denominator
    # ax_lower.errorbar(
    #    ratio.axes[0].bin_centers,
    #    ratio.values,
    #    xerr=ratio.axes[0].bin_widths / 2,
    #    yerr=ratio.errors,
    #    marker="o",
    #    linestyle="",
    #    color="black",
    #    alpha=0.8,
    # )

    # Plot truth and compare to the full efficient truth.
    ## Compare to the full efficiency to make sure that have the right shape...
    # full_eff_true = projection_func(hists["truef"], true_bin)
    ## Then normalize by the integral (sum) and bin width.
    ## Don't need to correct for the kinematic efficiency here because it's already fully efficient.
    # full_eff_true /= np.sum(full_eff_true.values)
    # full_eff_true /= full_eff_true.axes[0].bin_widths
    # ax_upper.errorbar(full_eff_true.axes[0].bin_centers, full_eff_true.values, xerr=full_eff_true.axes[0].bin_widths / 2, yerr=full_eff_true.errors, label = "True fully eff",
    #                  marker="o", linestyle="", alpha=0.8)
    ## Add ratio...
    # ratio = hist_true / full_eff_true
    # ax_lower.errorbar(
    #    ratio.axes[0].bin_centers,
    #    ratio.values,
    #    xerr=ratio.axes[0].bin_widths / 2,
    #    yerr=ratio.errors,
    #    marker="o",
    #    linestyle="",
    #    alpha=0.8,
    #    color="black",
    # )

    # Draw reference line for ratio
    ax_ratio_iter.axhline(y=1, color="black", linestyle="dashed", zorder=1)
    ax_ratio_true.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Label and layout
    plot_config.apply(fig=fig, axes=[ax_upper, ax_ratio_iter, ax_ratio_true])

    figure_name = f"{plot_config.name}"
    logger.info(f"Writing plot to {unfolding_output.output_dir / figure_name}.pdf")
    fig.savefig(unfolding_output.output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = unfolding_output.output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)


def plot_refolded(
    unfolding_output: UnfoldingOutput,
    hist_raw: binned_data.BinnedData,
    hist_smeared: binned_data.BinnedData,
    refolded_hists: Sequence[binned_data.BinnedData],
    plot_config: pb.PlotConfig,
    plot_png: bool = False,
) -> None:
    """Plot refolded."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax_upper, ax_lower = axes

    # Raw
    # Only plot if there's something meaningful to plot
    if hist_raw.values.any():
        ax_upper.errorbar(
            hist_raw.axes[0].bin_centers,
            hist_raw.values,
            xerr=hist_raw.axes[0].bin_widths / 2,
            yerr=hist_raw.errors,
            label="Raw",
            marker="o",
            linestyle="",
            color="red",
        )

    # Smeared
    ax_upper.errorbar(
        hist_smeared.axes[0].bin_centers,
        hist_smeared.values,
        xerr=hist_smeared.axes[0].bin_widths / 2,
        yerr=hist_smeared.errors,
        label="Smeared",
        marker="o",
        linestyle="",
        color="green",
    )

    ratio_denominator = hist_smeared if unfolding_output.smeared_input else hist_raw
    for i, hist in enumerate(refolded_hists, start=1):
        ax_upper.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            yerr=hist.errors,
            label=f"Bayes {i}",
            marker="o",
            linestyle="",
            alpha=0.8,
        )

        ratio = hist / ratio_denominator
        ax_lower.errorbar(
            ratio.axes[0].bin_centers,
            ratio.values,
            xerr=ratio.axes[0].bin_widths / 2,
            yerr=ratio.errors,
            marker="o",
            linestyle="",
            alpha=0.8,
        )

    # Add smeared ratio in the right circumstances.
    if not unfolding_output.smeared_input:
        r = hist_smeared / ratio_denominator
        ax_lower.errorbar(
            r.axes[0].bin_centers,
            r.values,
            xerr=r.axes[0].bin_widths / 2,
            yerr=r.errors,
            marker="o",
            linestyle="",
            color="green",
            alpha=0.8,
        )

    # Draw reference line for ratio
    ax_lower.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Label and layout
    plot_config.apply(fig=fig, axes=[ax_upper, ax_lower])

    figure_name = f"{plot_config.name}"
    # if tag:
    #    figure_name = f"{tag}_{figure_name}"
    fig.savefig(unfolding_output.output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = unfolding_output.output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)


def plot_response(
    hists: Mapping[str, binned_data.BinnedData],
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    # Setup
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")

    h = binned_data.BinnedData.from_existing_data(hists["h2_substructure_variable"])

    # Normalize the response.
    normalization_values = h.values.sum(axis=0, keepdims=True)
    h.values = np.divide(h.values, normalization_values, out=np.zeros_like(h.values), where=normalization_values != 0)

    # Finish setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        # "vmin": h_proj.values[h_proj.values > 0].min(),
        "vmin": max(1e-4, h.values[h.values > 0].min()),
        # "vmax": h.values.max(),
        "vmax": 1,
    }

    # Plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T,
        h.axes[1].bin_edges.T,
        h.values.T,
        norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")
    plt.close(fig)


def plot_jet_pt_vs_substructure(
    hists: Mapping[str, binned_data.BinnedData],
    hist_name: str,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    # Setup
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")

    h = binned_data.BinnedData.from_existing_data(hists[hist_name])

    # Finish setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        "vmin": h.values[h.values > 0].min(),
        "vmax": h.values.max(),
    }

    # Plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T,
        h.axes[1].bin_edges.T,
        h.values.T,
        norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")
    plt.close(fig)


def plot_efficiency(
    hists: Mapping[str, binned_data.BinnedData],
    efficiency_func: Callable[[Mapping[str, binned_data.BinnedData], helpers.RangeSelector], binned_data.BinnedData],
    true_bins: Sequence[helpers.RangeSelector],
    true_bin_label: str,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    """Plot kinematic efficiency."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    for true_bin in true_bins:
        # Project
        # We need the efficiency in the true bin that we actually want to measure.
        hist = efficiency_func(hists, true_bin)

        # Plot
        ax.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            yerr=hist.errors,
            label=fr"${true_bin.min} < {true_bin_label}_{{\text{{T,jet}}}}^{{\text{{true}}}} < {true_bin.max}$",
            marker="o",
            linestyle="",
            alpha=0.8,
        )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    fig.savefig(output_dir / f"{plot_config.name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{plot_config.name}.png")
    plt.close(fig)


def plot_select_iteration(
    hists: Mapping[str, binned_data.BinnedData],
    projection_func: Callable[[binned_data.BinnedData, helpers.RangeSelector], binned_data.BinnedData],
    max_iter: int,
    true_bin: helpers.RangeSelector,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    """Plot selected iteration."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # -2 because we go two above, and then -2 because we start at 2
    n_bins = max_iter - 2 - 2
    hist_reg = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )
    hist_stat = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )
    hist_total = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )

    for i, iter in enumerate(range(2, max_iter - 2)):
        # Current iteration
        hist_name = f"bayesian_unfolded_iter_{iter}"
        current_iter_hist = projection_func(hists[hist_name], true_bin)
        # Previous iter hist
        hist_name = f"bayesian_unfolded_iter_{iter-1}"
        previous_iter_hist = projection_func(hists[hist_name], true_bin)
        # Iter + 2 hist
        hist_name = f"bayesian_unfolded_iter_{iter+2}"
        forward_iter_hist = projection_func(hists[hist_name], true_bin)

        # hist = _normalize_hist(hist)
        # Calculate and store regularization error
        regularization_value = np.sum(
            (
                np.maximum(
                    np.abs(previous_iter_hist.values - current_iter_hist.values),
                    np.abs(forward_iter_hist.values - current_iter_hist.values),
                )
                # TEMP: Try excluding the untagged bin.
                # / current_iter_hist.values)[1:]
                / current_iter_hist.values
            )
        )
        hist_reg.values[i] = regularization_value
        # Calculate and store stat error
        stat_value = np.sum(current_iter_hist.errors / current_iter_hist.values)
        hist_stat.values[i] = stat_value

        # Total
        hist_total.values[i] = np.sqrt(regularization_value ** 2 + stat_value ** 2)

    # Plot the total errors
    ax.errorbar(
        hist_total.axes[0].bin_centers,
        hist_total.values,
        xerr=hist_total.axes[0].bin_widths / 2,
        label="Total",
        marker="o",
        linestyle="",
    )
    # The regularization errors
    ax.errorbar(
        hist_reg.axes[0].bin_centers,
        hist_reg.values,
        xerr=hist_reg.axes[0].bin_widths / 2,
        label="Regularization",
        marker="o",
        linestyle="",
    )
    # Plot the stat errors
    ax.errorbar(
        hist_stat.axes[0].bin_centers,
        hist_stat.values,
        xerr=hist_stat.axes[0].bin_widths / 2,
        label="Statistical",
        marker="o",
        linestyle="",
    )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)
    # Additional tweaks
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=2.0))

    figure_name = f"{plot_config.name}"
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")
    plt.close(fig)


def plot_kt_unfolding(unfolding_output: UnfoldingOutput, plot_png: bool = False) -> Path:
    # with sns.color_palette("GnBu_d", n_colors=11):
    with sns.color_palette("Paired", n_colors=unfolding_output.max_n_iter):
        # Main unfolded plot.
        true_jet_pt_range = helpers.JetPtRange(60, 80)
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_substructure(
                unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_substructure(
                unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
            ),
            unfolded_hists=[
                unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
                for n_iter in range(1, unfolding_output.max_n_iter)
            ],
            plot_config=pb.PlotConfig(
                name=f"unfolded_{unfolding_output.substructure_variable}_true_{str(true_jet_pt_range)}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"$\text{{d}}N/\text{{d}}k_{{\text{{T}}}}\:(\text{{GeV}}/c)^{{-1}}$",  # noqa: F541
                                log=True,
                            )
                        ],
                        legend=pb.LegendConfig(location="lower left"),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(-0.5, 15)),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to true",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )
        # Check a broader true jet pt range: 40-120
        true_jet_pt_range = helpers.JetPtRange(40, 120)
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_substructure(
                unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_substructure(
                unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
            ),
            unfolded_hists=[
                unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
                for n_iter in range(1, unfolding_output.max_n_iter)
            ],
            plot_config=pb.PlotConfig(
                name=f"unfolded_{unfolding_output.substructure_variable}_true_{str(true_jet_pt_range)}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"$\text{{d}}N/\text{{d}}k_{{\text{{T}}}}\:(\text{{GeV}}/c)^{{-1}}$",  # noqa: F541
                                log=True,
                            )
                        ],
                        legend=pb.LegendConfig(location="lower left"),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(-0.5, 15)),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to true",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )
        # Unfolded jet pt
        true_substructure_variable_range = helpers.KtRange(-1, 100)
        text = f"${true_substructure_variable_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_jet_pt(
                unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
                unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
            ),
            unfolded_hists=[
                unfolding_output.unfolded_jet_pt(
                    n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
                )
                for n_iter in range(1, unfolding_output.max_n_iter)
            ],
            plot_config=pb.PlotConfig(
                name="unfolded_pt",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                        ],
                        legend=pb.LegendConfig(location="lower left"),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to true",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )

        # Now, on to the refolded.
        text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='data')}$"
        plot_refolded(
            unfolding_output=unfolding_output,
            hist_raw=unfolding_output.smeared_substructure(
                hist_name=unfolding_output.raw_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
            ),
            hist_smeared=unfolding_output.smeared_substructure(
                hist_name=unfolding_output.smeared_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
            ),
            refolded_hists=[
                unfolding_output.refolded_substructure(
                    n_iter=n_iter, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
                )
                for n_iter in range(1, unfolding_output.max_n_iter)
            ],
            plot_config=pb.PlotConfig(
                name=f"refolded_{unfolding_output.substructure_variable}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                        ],
                        legend=pb.LegendConfig(location="lower left"),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                            # y label is set in the function.
                            pb.AxisConfig(
                                "y",
                                label="Ratio to smeared" if unfolding_output.smeared_input else "Ratio to data",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )
        # Jet pt
        text = f"${unfolding_output.smeared_var_range.display_str(label='data')}$"
        plot_refolded(
            unfolding_output=unfolding_output,
            hist_raw=unfolding_output.smeared_jet_pt(
                hist_name=unfolding_output.raw_hist_name,
                smeared_substructure_variable_range=unfolding_output.smeared_var_range,
            ),
            hist_smeared=unfolding_output.smeared_jet_pt(
                hist_name=unfolding_output.smeared_hist_name,
                smeared_substructure_variable_range=unfolding_output.smeared_var_range,
            ),
            refolded_hists=[
                unfolding_output.refolded_jet_pt(
                    n_iter=n_iter, smeared_substructure_variable_range=unfolding_output.smeared_var_range
                )
                for n_iter in range(1, unfolding_output.max_n_iter)
            ],
            plot_config=pb.PlotConfig(
                name="refolded_pt",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                        ],
                        legend=pb.LegendConfig(location="lower left"),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                            # y label is set in the function.
                            pb.AxisConfig(
                                "y",
                                label="Ratio to smeared"
                                if (unfolding_output.smeared_input or unfolding_output.suffix == "closure2")
                                else "Ratio to data",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )

    # Plot the response
    if "h2_substructure_variable" in unfolding_output.hists:
        text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='hybrid')}$"
        plot_response(
            hists=unfolding_output.hists,
            plot_config=pb.PlotConfig(
                name=f"response_{unfolding_output.substructure_variable}_hybrid_{unfolding_output.smeared_jet_pt_range}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                        # Use the smeared variable max value as a proxy for the max true value of interest.
                        pb.AxisConfig(
                            "y",
                            label=r"$k_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$",
                            range=(0, unfolding_output.smeared_var_range.max),
                        ),
                    ],
                    text=pb.TextConfig(text, 0.97, 0.03),
                ),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
        )

    # Plot kt vs jet pt
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="smeared",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_hybrid",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    # True
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="true",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_true",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$", range=(None, 20)),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    # Select the n_iter iteration
    true_jet_pt_range = helpers.JetPtRange(60, 80)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_select_iteration(
        hists=unfolding_output.hists,
        projection_func=_project_substructure_variable,
        max_iter=unfolding_output.max_n_iter,
        true_bin=true_jet_pt_range,
        plot_config=pb.PlotConfig(
            name=f"select_iteration_{unfolding_output.substructure_variable}_true_pt_60_80",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label="Iteration"),
                    pb.AxisConfig("y", label="Summed Error", range=(0, None)),
                ],
                legend=pb.LegendConfig(location="center right"),
                text=pb.TextConfig(text, 0.03, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    # Efficiency
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=_efficiency_substructure_variable,
        true_bins=[
            helpers.JetPtRange(40, 120),
            helpers.JetPtRange(40, 60),
            helpers.JetPtRange(60, 80),
            helpers.JetPtRange(80, 120),
        ],
        true_bin_label="p",
        plot_config=pb.PlotConfig(
            name=f"efficiency_{unfolding_output.substructure_variable}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", log=True),
                    pb.AxisConfig("y", label=r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$"),
                ],
                legend=pb.LegendConfig(location="lower left"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=_efficiency_pt,
        true_bins=[
            unfolding_output.smeared_var_range,
            # helpers.RangeSelector(unfolding_output.smeared_var_range.min, unfolding_output.smeared_var_range.max),
            # helpers.RangeSelector(1, 15),
            # helpers.RangeSelector(2, 13),
            # helpers.RangeSelector(2, 15),
        ],
        true_bin_label="k",
        plot_config=pb.PlotConfig(
            name="efficiency_pt",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                    pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$"),
                ],
                legend=pb.LegendConfig(location="lower right"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    # plot_spectra_comparison(hists, output_dir)
    # plot_spectra_comparison_fine_binned(hists, output_dir)
    # plot_response_matrix(hists["responseUnscaled"], "response", output_dir)

    return unfolding_output.output_dir


def run(collision_system: str) -> None:
    for input_file in [
        ###################### kt smeared = 2-10 ##########################
        ## 2-10, 1-2, 30-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        #    smeared_input=True,
        # ),
        ## 2-10, 1-2, 40-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ## 2-10, 10-13, 30-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        #    smeared_input=True,
        # ),
        ## 2-10, 10-13, 40-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ###################### kt smeared = 3-10 ##########################
        ## 3-10, 2-3, 30-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    smeared_input=True,
        # ),
        # 3-10, 2-3, 40-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ## 3-10, 10-13, 30-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    smeared_input=True,
        # ),
        ## 3-10, 10-13, 40-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        # 3-10, 2-3, 40-120, pure matches
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    pure_matches=True,
        #    n_iter_compare=11,
        #    max_iter=15,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    pure_matches=True,
        #    n_iter_compare=11,
        #    max_iter=15,
        #    smeared_input=True,
        # ),
        ###################### kt smeared = 3-10, broad true bins ##########################
        ## 3-10, 2-3, 30-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    suffix="broadTrueBins",
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    suffix="broadTrueBins",
        #    smeared_input=True,
        # ),
        ## 3-10, 2-3, 40-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    suffix="broadTrueBins",
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    suffix="broadTrueBins",
        #    smeared_input=True,
        # ),
        ## 3-11, 30-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    smeared_input=True,
        # ),
        ## 3-11, 40-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    smeared_input=True,
        # ),
        ## 3-15, 30-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    smeared_input=True,
        # ),
        ## 3-15, 40-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    pure_matches=True,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    pure_matches=True,
        #    smeared_input=True,
        # ),
        ###################### kt smeared = 5-15 ##########################
        ## 4-15, 3-4, 40-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(4, 15),
        #    smeared_untagged_var=helpers.KtRange(3, 4),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(4, 15),
        #    smeared_untagged_var=helpers.KtRange(3, 4),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ## 5-15, 4-5, 40-120
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(5, 15),
        #    smeared_untagged_var=helpers.KtRange(4, 5),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # InputFile(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(5, 15),
        #    smeared_untagged_var=helpers.KtRange(4, 5),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ####### Dynamical kt ##########
        # 3-15, 2-3, 40-120
        InputFile(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
        ),
        InputFile(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            smeared_input=True,
        ),
        # 2-15, 1-2, 30-120
        InputFile(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(2, 15),
            smeared_untagged_var=helpers.KtRange(1, 2),
            smeared_jet_pt_range=helpers.JetPtRange(30, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
        ),
        InputFile(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(2, 15),
            smeared_untagged_var=helpers.KtRange(1, 2),
            smeared_jet_pt_range=helpers.JetPtRange(30, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            smeared_input=True,
        ),
        ####### Dynamical time ##########
        # 3-15, 2-3, 40-120
        InputFile(
            "kt",
            "dynamical_time",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
        ),
        InputFile(
            "kt",
            "dynamical_time",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            smeared_input=True,
        ),
        ####### Leading kt ##########
        # 3-15, 2-3, 40-120
        InputFile(
            "kt",
            "leading_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
        ),
        InputFile(
            "kt",
            "leading_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            smeared_input=True,
        ),
    ]:
        plot_kt_unfolding(input_file=input_file, collision_system=collision_system)


def run_delta_R(collision_system: str) -> None:
    for input_file in [
        InputFile(
            "delta_R",
            "leading_kt_z_cut_02",
            # Hack until the labeling is fixed...
            smeared_var_range=helpers.RgRange(0, 350),
            smeared_untagged_var=helpers.RgRange(-50, 0),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        ),
        InputFile(
            "delta_R",
            "leading_kt_z_cut_02",
            # Hack until the labeling is fixed...
            smeared_var_range=helpers.RgRange(0, 350),
            smeared_untagged_var=helpers.RgRange(-50, 0),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            smeared_input=True,
        ),
    ]:
        # Setup
        hists, output_dir = setup(input_file=input_file, collision_system=collision_system)

        tag = ""
        if input_file.smeared_input:
            tag = "hybridAsInput"

        # with sns.color_palette("GnBu_d", n_colors=11):
        with sns.color_palette("Paired", n_colors=input_file.max_iter):
            n_iter_for_ratio = 6
            jet_pt_for_text = helpers.RangeSelector(60, 80)
            text = f"${jet_pt_for_text.display_str(label='true')}$"
            plot_unfolded(
                hists=hists,
                projection_func=_project_substructure_variable,
                efficiency_func=_efficiency_substructure_variable,
                n_iter_for_ratio=n_iter_for_ratio,
                max_iter=input_file.max_iter,
                true_bin=helpers.RangeSelector(60, 80),
                tag=tag,
                plot_config=pb.PlotConfig(
                    name=f"unfolded_{input_file.substructure_variable}_true_pt_60_80",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=r"$\text{d}N/\text{d}\Delta R$",
                                    log=True,
                                )
                            ],
                            # legend=pb.LegendConfig(location="lower left"),
                            legend=pb.LegendConfig(location="center right"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[pb.AxisConfig("y", label=fr"Ratio to iter {n_iter_for_ratio}", range=(0.5, 1.5))],
                        ),
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$\Delta R$"),
                                pb.AxisConfig(
                                    "y",
                                    label="Ratio to true",
                                    range=(0.5, 1.5),
                                ),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )
            # 40-120 true pt.
            jet_pt_for_text = helpers.RangeSelector(40, 120)
            text = f"${jet_pt_for_text.display_str(label='true')}$"
            plot_unfolded(
                hists=hists,
                projection_func=_project_substructure_variable,
                efficiency_func=_efficiency_substructure_variable,
                n_iter_for_ratio=n_iter_for_ratio,
                max_iter=input_file.max_iter,
                true_bin=helpers.RangeSelector(40, 120),
                tag=tag,
                plot_config=pb.PlotConfig(
                    name=f"unfolded_{input_file.substructure_variable}_true_pt_40_120",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=fr"$\text{{d}}N/\text{{d}}k_{{\text{{T}}}}\:(\text{{GeV}}/c)^{{-1}}$",
                                    log=True,
                                )
                            ],
                            legend=pb.LegendConfig(location="center right"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[pb.AxisConfig("y", label=fr"Ratio to iter {n_iter_for_ratio}", range=(0.5, 1.5))],
                        ),
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$\Delta R$"),
                                pb.AxisConfig(
                                    "y",
                                    label="Ratio to true",
                                    range=(0.5, 1.5),
                                ),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )
            text = ""
            plot_unfolded(
                hists=hists,
                projection_func=_project_jet_pt,
                efficiency_func=_efficiency_pt,
                n_iter_for_ratio=n_iter_for_ratio,
                max_iter=input_file.max_iter,
                true_bin=helpers.RangeSelector(0, 0.6),
                tag=tag,
                plot_config=pb.PlotConfig(
                    name="unfolded_pt",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True
                                )
                            ],
                            legend=pb.LegendConfig(location="lower left"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[pb.AxisConfig("y", label=fr"Ratio to iter {n_iter_for_ratio}", range=(0.5, 1.5))],
                        ),
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                                pb.AxisConfig(
                                    "y",
                                    label="Ratio to true",
                                    range=(0.5, 1.5),
                                ),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )
            jet_pt_for_text = helpers.RangeSelector(40, 120)
            text = f"${jet_pt_for_text.display_str(label='data')}$"
            plot_refolded(
                hists=hists,
                projection_func=_project_substructure_variable,
                smeared_input=input_file.smeared_input,
                max_iter=input_file.max_iter,
                measured_bin=helpers.RangeSelector(40, 120),
                tag=tag,
                plot_config=pb.PlotConfig(
                    name=f"refolded_{input_file.substructure_variable}",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[pb.AxisConfig("y", label=r"$\text{d}N/\text{d}\Delta R$", log=True)],
                            legend=pb.LegendConfig(location="lower left"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$\Delta R$"),
                                # y label is set in the function.
                                pb.AxisConfig(
                                    "y",
                                    label="Ratio to smeared" if input_file.smeared_input else "Ratio to data",
                                    range=(0.5, 1.5),
                                ),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )
            text = ""
            plot_refolded(
                hists=hists,
                projection_func=_project_jet_pt,
                smeared_input=input_file.smeared_input,
                max_iter=input_file.max_iter,
                measured_bin=helpers.RangeSelector(0, 0.35),
                tag=tag,
                plot_config=pb.PlotConfig(
                    name="refolded_pt",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True
                                )
                            ],
                            legend=pb.LegendConfig(location="lower left"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                                # y label is set in the function.
                                pb.AxisConfig(
                                    "y",
                                    label="Ratio to smeared" if input_file.smeared_input else "Ratio to data",
                                    range=(0.5, 1.5),
                                ),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )

        # Plot the response
        if "h2_substructure_variable" in hists:
            jet_pt_for_text = helpers.JetPtRange(40, 120)
            text = f"${jet_pt_for_text.display_str(label='hybrid')}$"
            plot_response(
                hists=hists,
                tag=tag,
                plot_config=pb.PlotConfig(
                    name=f"response_{input_file.substructure_variable}_hybrid_40_120",
                    panels=pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$\Delta R^{\text{hybrid}}\:(\text{GeV}/c)$"),
                            pb.AxisConfig("y", label=r"$\Delta R^{\text{true}}\:(\text{GeV}/c)$", range=(0, 0.4)),
                        ],
                        text=pb.TextConfig(text, 0.97, 0.03),
                    ),
                    figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.10}),
                ),
                output_dir=output_dir,
            )

        # Select the n_iter iteration
        jet_pt_for_text = helpers.JetPtRange(60, 80)
        text = f"${jet_pt_for_text.display_str(label='true')}$"
        plot_select_iteration(
            hists=hists,
            projection_func=_project_substructure_variable,
            max_iter=19,
            true_bin=helpers.JetPtRange(60, 80),
            tag=tag,
            plot_config=pb.PlotConfig(
                name=f"select_iteration_{input_file.substructure_variable}_true_pt_60_80",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label="Iteration"),
                        pb.AxisConfig("y", label="Summed Error", range=(0, None)),
                    ],
                    legend=pb.LegendConfig(location="center right"),
                    text=pb.TextConfig(text, 0.03, 0.03),
                ),
            ),
            output_dir=output_dir,
        )


if __name__ == "__main__":
    # Setup
    helpers.setup_logging()
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
    collision_system = "PbPb"

    # Enable ticks on all sides
    # Unfortunately, some of this is overriding the pachyderm plotting style.
    # That will have to be updated eventually...
    # matplotlib.rcParams["xtick.top"] = True
    # matplotlib.rcParams["xtick.minor.top"] = True
    # matplotlib.rcParams["ytick.right"] = True
    # matplotlib.rcParams["ytick.minor.right"] = True

    run(collision_system=collision_system)
    # run_delta_R(collision_system=collision_system)
