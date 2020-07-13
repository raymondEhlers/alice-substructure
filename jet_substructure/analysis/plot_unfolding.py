""" Plot unfolding.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
import logging
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple

import attr
import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import uproot
from pachyderm import binned_data

from jet_substructure.analysis import plot_base as pb
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


def _efficiency_kt(
    hists: Mapping[str, binned_data.BinnedData], true_jet_pt_bin: helpers.RangeSelector
) -> binned_data.BinnedData:
    """

    """
    # For convenience
    bh_cut_efficiency = hists["true"].to_boost_histogram()
    bh_full_efficiency = hists["truef"].to_boost_histogram()

    # Select true pt range.
    selection = slice(bh.loc(true_jet_pt_bin.min), bh.loc(true_jet_pt_bin.max), bh.sum)
    cut = binned_data.BinnedData.from_existing_data(bh_cut_efficiency[:, selection])
    full = binned_data.BinnedData.from_existing_data(bh_full_efficiency[:, selection])

    return cut / full


def _project_kt(input_hist: binned_data.BinnedData, true_jet_pt_bin: helpers.RangeSelector) -> binned_data.BinnedData:
    # For convenience
    bh_hist = input_hist.to_boost_histogram()

    selection = slice(bh.loc(true_jet_pt_bin.min), bh.loc(true_jet_pt_bin.max), bh.sum)
    return binned_data.BinnedData.from_existing_data(bh_hist[:, selection])


def _efficiency_pt(
    hists: Mapping[str, binned_data.BinnedData], true_kt_bin: helpers.RangeSelector
) -> binned_data.BinnedData:
    """

    """
    # For convenience
    bh_cut_efficiency = hists["true"].to_boost_histogram()
    bh_full_efficiency = hists["truef"].to_boost_histogram()

    # Select true pt range.
    selection = slice(bh.loc(true_kt_bin.min), bh.loc(true_kt_bin.max), bh.sum)
    cut = binned_data.BinnedData.from_existing_data(bh_cut_efficiency[selection, :])
    full = binned_data.BinnedData.from_existing_data(bh_full_efficiency[selection, :])

    return cut / full


def _project_pt(input_hist: binned_data.BinnedData, true_kt_bin: helpers.RangeSelector) -> binned_data.BinnedData:
    """

    """
    bh_hist = input_hist.to_boost_histogram()

    selection = slice(bh.loc(true_kt_bin.min), bh.loc(true_kt_bin.max), bh.sum)
    return binned_data.BinnedData.from_existing_data(bh_hist[selection, :])


def _normalize_unfolded(hist: binned_data.BinnedData, efficiency: binned_data.BinnedData) -> binned_data.BinnedData:
    # Apply the efficiency.
    hist /= efficiency
    # Then normalize by the integral (sum) and bin width.
    hist /= np.sum(hist.values)
    hist /= hist.axes[0].bin_widths
    return hist


def plot_unfolded(
    hists: Mapping[str, binned_data.BinnedData],
    projection_func: Callable[[binned_data.BinnedData, helpers.RangeSelector], binned_data.BinnedData],
    efficiency_func: Callable[[Mapping[str, binned_data.BinnedData], helpers.RangeSelector], binned_data.BinnedData],
    n_iter_for_ratio: int,
    true_bin: helpers.RangeSelector,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """ Plot unfolded.

    """
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    ax_upper, ax_lower = axes

    # We need the efficiency in the true bin that we actually want to measure.
    efficiency = efficiency_func(hists, true_bin)
    # For convenience in normalizing.
    _normalize_hist = functools.partial(_normalize_unfolded, efficiency=efficiency)

    # True
    # Project true onto the kt axis. We use boost histogram for the convince.
    hist_true = projection_func(hists["true"], true_bin)
    # Normalize
    hist_true = _normalize_hist(hist_true)

    # Determine ratio denominator.
    if n_iter_for_ratio > 0:
        selected_n_iter_hist = projection_func(hists[f"Bayesian_Unfoldediter{n_iter_for_ratio}"], true_bin)
        selected_n_iter_hist = _normalize_hist(selected_n_iter_hist)
        h_ratio_denominator = selected_n_iter_hist
    else:
        h_ratio_denominator = hist_true

    for i in range(1, 10):
        # Retrieve the hist and normalize it properly.
        hist = projection_func(hists[f"Bayesian_Unfoldediter{i}"], true_bin)
        hist = _normalize_hist(hist)

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

        ratio = hist / h_ratio_denominator
        ax_lower.errorbar(
            ratio.axes[0].bin_centers,
            ratio.values,
            xerr=ratio.axes[0].bin_widths / 2,
            yerr=ratio.errors,
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
    ax_lower.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Label and layout
    plot_config.apply(fig=fig, axes=[ax_upper, ax_lower])

    fig.savefig(output_dir / f"{plot_config.name}.pdf")
    plt.close(fig)


def _normalize_refolded(hist: binned_data.BinnedData) -> binned_data.BinnedData:
    """

    """
    hist /= np.sum(hist.values)
    hist /= hist.axes[0].bin_widths
    return hist


def plot_refolded(
    hists: Mapping[str, binned_data.BinnedData],
    projection_func: Callable[[binned_data.BinnedData, helpers.RangeSelector], binned_data.BinnedData],
    smeared_input: bool,
    measured_bin: helpers.RangeSelector,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """ Plot refolded.

    """
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    ax_upper, ax_lower = axes

    # Raw
    hist_raw = projection_func(hists["raw"], measured_bin)
    # Normalize
    hist_raw = _normalize_refolded(hist_raw)
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
    hist_smeared = projection_func(hists["smeared"], measured_bin)
    hist_smeared = _normalize_refolded(hist_smeared)
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

    ratio_denominator = hist_smeared if smeared_input else hist_raw
    for i in range(1, 10):
        # Convert
        hist = projection_func(hists[f"Bayesian_Foldediter{i}"], measured_bin)
        hist = _normalize_refolded(hist)
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
    if not smeared_input:
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

    fig.savefig(output_dir / f"{plot_config.name}.pdf")
    plt.close(fig)


def plot_efficiency(
    hists: Mapping[str, binned_data.BinnedData],
    efficiency_func: Callable[[Mapping[str, binned_data.BinnedData], helpers.RangeSelector], binned_data.BinnedData],
    true_bins: Sequence[helpers.RangeSelector],
    true_bin_label: str,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """ Plot kinematic efficiency.

    """
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
    plt.close(fig)


@attr.s
class InputFile:
    substructure_variable: str = attr.ib()
    grooming_method: str = attr.ib()
    var_range: helpers.RangeSelector = attr.ib()
    pt_range: helpers.RangeSelector = attr.ib()
    smeared_input: bool = attr.ib(default=False)
    pure_matches: bool = attr.ib(default=False)
    suffix: str = attr.ib(default="")

    @property
    def identifier(self) -> str:
        name = f"{self.substructure_variable}_grooming_method_{self.grooming_method}"
        name += f"_{self.var_range}"
        name += f"_{self.pt_range}"
        if self.smeared_input:
            name += "_hybrid_as_input"
        if self.pure_matches:
            name += "_pureMatches"
        if self.suffix:
            name += f"_{self.suffix}"
        return name

    @property
    def filename(self) -> str:
        return f"unfolding_{self.identifier}.root"


def setup(input_file: InputFile, collision_system: str) -> Tuple[Dict[str, binned_data.BinnedData], Path]:
    base_dir = Path("output") / collision_system / "unfolding"
    input_filename = base_dir / input_file.filename
    output_dir = base_dir / input_file.substructure_variable / input_file.identifier
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing file {input_filename}")
    logger.info(f"Output dir: {output_dir}")

    # Extract with uproot and convert to BinnedData
    hists = {}
    f = uproot.open(input_filename)
    for k in f.keys():
        hist_key = k.decode("utf-8")
        hist_key = hist_key[: hist_key.find(";")]
        hists[hist_key] = binned_data.BinnedData.from_existing_data(f[k])

    # Hists:
    # [
    #     "correff20-40", "correff40-60", "correff60-80", "correff80-120",
    #     "raw", "smeared", "trueptd", "true", "truef",
    #     "Bayesian_Unfoldediter1", "Bayesian_Foldediter1", "Bayesian_Unfoldediter2", "Bayesian_Foldediter2", "Bayesian_Unfoldediter3", "Bayesian_Foldediter3",
    #     "Bayesian_Unfoldediter4", "Bayesian_Foldediter4", "Bayesian_Unfoldediter5", "Bayesian_Foldediter5", "Bayesian_Unfoldediter6", "Bayesian_Foldediter6",
    #     "Bayesian_Unfoldediter7", "Bayesian_Foldediter7", "Bayesian_Unfoldediter8", "Bayesian_Foldediter8", "Bayesian_Unfoldediter9", "Bayesian_Foldediter9",
    #     "pearsonmatrix_iter8_binshape0", "pearsonmatrix_iter8_binshape1", "pearsonmatrix_iter8_binshape2", "pearsonmatrix_iter8_binshape3",
    #     "pearsonmatrix_iter8_binpt0", "pearsonmatrix_iter8_binpt1", "pearsonmatrix_iter8_binpt2", "pearsonmatrix_iter8_binpt3", "pearsonmatrix_iter8_binpt4",
    #     "pearsonmatrix_iter8_binpt5", "pearsonmatrix_iter8_binpt6", "pearsonmatrix_iter8_binpt7",
    # ]

    return hists, output_dir


def run(collision_system: str) -> None:
    for input_file in [
        InputFile("kt", "leading_kt_z_cut_02", var_range=helpers.KtRange(3, 10), pt_range=helpers.JetPtRange(30, 120)),
        InputFile(
            "kt",
            "leading_kt_z_cut_02",
            var_range=helpers.KtRange(3, 10),
            pt_range=helpers.JetPtRange(30, 120),
            smeared_input=True,
        ),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="kt3to10"),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="kt3to10", smeared_input=True),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="kt3to12"),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="kt3to12", smeared_input=True),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="minKt2"),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="minKt2", smeared_input=True),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="minKt3"),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="minKt3", smeared_input=True),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="kt2to13"),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="kt2to13", smeared_input=True),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="kt3to13"),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="kt3to13", smeared_input=True),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="test", pure_matches=True),
        # InputFile("kt", "leading_kt_z_cut_02", suffix="test", pure_matches=True, smeared_input=True),
        # InputFile("kt", "leading_kt_z_cut_04", suffix="test"),
        # InputFile("kt", "leading_kt_z_cut_04", suffix="test", smeared_input=True),
        # InputFile("kt", "leading_kt_z_cut_04", suffix="test", pure_matches=True),
        # InputFile("kt", "leading_kt_z_cut_04", suffix="test", pure_matches=True, smeared_input=True),
    ]:
        hists, output_dir = setup(input_file=input_file, collision_system=collision_system)

        # with sns.color_palette("GnBu_d", n_colors=11):
        with sns.color_palette("Paired", n_colors=11):
            n_iter_for_ratio = -1
            jet_pt_for_text = helpers.RangeSelector(60, 80)
            text = f"${jet_pt_for_text.display_str(label='true')}$"
            plot_unfolded(
                hists=hists,
                projection_func=_project_kt,
                efficiency_func=_efficiency_kt,
                n_iter_for_ratio=n_iter_for_ratio,
                true_bin=helpers.RangeSelector(60, 80),
                plot_config=pb.PlotConfig(
                    name=f"unfolded_{input_file.substructure_variable}_true_pt_60_80",
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
                            legend=pb.LegendConfig(location="lower left"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0, 15)),
                                pb.AxisConfig(
                                    "y",
                                    label=fr"Ratio to iter {n_iter_for_ratio}"
                                    if n_iter_for_ratio > 0
                                    else "Ratio to true",
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
                projection_func=_project_kt,
                efficiency_func=_efficiency_kt,
                n_iter_for_ratio=n_iter_for_ratio,
                true_bin=helpers.RangeSelector(40, 120),
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
                            legend=pb.LegendConfig(location="lower left"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0, 15)),
                                pb.AxisConfig(
                                    "y",
                                    label=fr"Ratio to iter {n_iter_for_ratio}"
                                    if n_iter_for_ratio > 0
                                    else "Ratio to true",
                                    range=(0.5, 1.5),
                                ),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )
            text = ""
            n_iter_for_ratio = 6
            plot_unfolded(
                hists=hists,
                projection_func=_project_pt,
                efficiency_func=_efficiency_pt,
                n_iter_for_ratio=n_iter_for_ratio,
                true_bin=helpers.RangeSelector(0, 25),
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
                            axes=[
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                                pb.AxisConfig(
                                    "y",
                                    label=fr"Ratio to iter {n_iter_for_ratio}"
                                    if n_iter_for_ratio > 0
                                    else "Ratio to true",
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
                projection_func=_project_kt,
                smeared_input=input_file.smeared_input,
                measured_bin=helpers.RangeSelector(40, 120),
                plot_config=pb.PlotConfig(
                    name=f"refolded_{input_file.substructure_variable}",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y", label=r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True
                                )
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
                projection_func=_project_pt,
                smeared_input=input_file.smeared_input,
                measured_bin=helpers.RangeSelector(1, 15),
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

        plot_efficiency(
            hists=hists,
            efficiency_func=_efficiency_kt,
            true_bins=[
                helpers.RangeSelector(40, 120),
                helpers.RangeSelector(40, 60),
                helpers.RangeSelector(60, 80),
                helpers.RangeSelector(80, 120),
            ],
            true_bin_label="p",
            plot_config=pb.PlotConfig(
                name=f"efficiency_{input_file.substructure_variable}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", log=True),
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$"),
                    ],
                    legend=pb.LegendConfig(location="lower left"),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
            ),
            output_dir=output_dir,
        )
        plot_efficiency(
            hists=hists,
            efficiency_func=_efficiency_pt,
            true_bins=[
                helpers.RangeSelector(1, 13),
                helpers.RangeSelector(1, 15),
                helpers.RangeSelector(2, 13),
                helpers.RangeSelector(2, 15),
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
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
            ),
            output_dir=output_dir,
        )

        # plot_spectra_comparison(hists, output_dir)
        # plot_spectra_comparison_fine_binned(hists, output_dir)
        # plot_response_matrix(hists["responseUnscaled"], "response", output_dir)


def run_delta_R(collision_system: str) -> None:
    for input_file in [
        InputFile(
            "delta_R",
            "leading_kt_z_cut_04",
            var_range=helpers.RgRange(0, 0.4),
            pt_range=helpers.JetPtRange(40, 120),
            suffix="test",
        ),
        InputFile(
            "delta_R",
            "leading_kt_z_cut_04",
            var_range=helpers.RgRange(0, 0.4),
            pt_range=helpers.JetPtRange(40, 120),
            suffix="test",
            smeared_input=True,
        ),
    ]:
        hists, output_dir = setup(input_file=input_file, collision_system=collision_system)

        # with sns.color_palette("GnBu_d", n_colors=11):
        with sns.color_palette("Paired", n_colors=11):
            n_iter_for_ratio = 6
            jet_pt_for_text = helpers.RangeSelector(60, 80)
            text = f"${jet_pt_for_text.display_str(label='true')}$"
            plot_unfolded(
                hists=hists,
                projection_func=_project_kt,
                efficiency_func=_efficiency_kt,
                n_iter_for_ratio=n_iter_for_ratio,
                true_bin=helpers.RangeSelector(60, 80),
                plot_config=pb.PlotConfig(
                    name=f"unfolded_{input_file.substructure_variable}_true_pt_60_80",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[pb.AxisConfig("y", label=r"$\text{d}N/\text{d}\Delta R$", log=True,)],
                            # legend=pb.LegendConfig(location="lower left"),
                            legend=pb.LegendConfig(location="center right"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$\Delta R$"),
                                pb.AxisConfig("y", label=fr"Ratio to iter {n_iter_for_ratio}", range=(0.5, 1.5)),
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
                projection_func=_project_kt,
                efficiency_func=_efficiency_kt,
                n_iter_for_ratio=n_iter_for_ratio,
                true_bin=helpers.RangeSelector(40, 120),
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
                            axes=[
                                pb.AxisConfig("x", label=r"$\Delta R$"),
                                pb.AxisConfig("y", label=fr"Ratio to iter {n_iter_for_ratio}", range=(0.5, 1.5)),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )
            text = ""
            plot_unfolded(
                hists=hists,
                projection_func=_project_pt,
                efficiency_func=_efficiency_pt,
                n_iter_for_ratio=n_iter_for_ratio,
                true_bin=helpers.RangeSelector(0, 0.6),
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
                            axes=[
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                                pb.AxisConfig("y", label=fr"Ratio to iter {n_iter_for_ratio}", range=(0.5, 1.5)),
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
                projection_func=_project_kt,
                smeared_input=input_file.smeared_input,
                measured_bin=helpers.RangeSelector(40, 120),
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
                projection_func=_project_pt,
                smeared_input=input_file.smeared_input,
                measured_bin=helpers.RangeSelector(0, 0.35),
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
