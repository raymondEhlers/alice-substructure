""" Plot unfolding.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
import logging
from pathlib import Path
from typing import Callable, Mapping

import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
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

    return binned_data.BinnedData.from_existing_data(
        bh_hist[:, bh.loc(true_jet_pt_bin.min) : bh.loc(true_jet_pt_bin.max) : bh.sum]  # noqa: E203
    )


def _efficiency_pt(
    hists: Mapping[str, binned_data.BinnedData], true_kt_bin: helpers.RangeSelector
) -> binned_data.BinnedData:
    """

    """
    # For convenience
    bh_cut_efficiency = hists["true"].to_boost_histogram()
    bh_full_efficiency = hists["truef"].to_boost_histogram()

    # Select true pt range.
    # selection = slice(bh.loc(true_kt_bin.min), bh.loc(true_kt_bin.max), bh.sum)
    cut = binned_data.BinnedData.from_existing_data(bh_cut_efficiency[:: bh.sum, :])
    full = binned_data.BinnedData.from_existing_data(bh_full_efficiency[:: bh.sum, :])

    return cut / full


def _project_pt(input_hist: binned_data.BinnedData, true_kt_bin: helpers.RangeSelector) -> binned_data.BinnedData:
    """

    """
    bh_hist = input_hist.to_boost_histogram()

    return binned_data.BinnedData.from_existing_data(bh_hist[:: bh.sum, :])


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
    h_ratio_denominator = projection_func(hists[f"Bayesian_Unfoldediter{n_iter_for_ratio}"], true_bin)
    h_ratio_denominator = _normalize_hist(h_ratio_denominator)

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
    ## Plot truth and compare to the full efficient truth.
    ## Plot truth
    # ax_upper.errorbar(hist_true.axes[0].bin_centers, hist_true.values, xerr=hist_true.axes[0].bin_widths / 2, yerr=hist_true.errors, label = "True",
    #                  marker="o", linestyle="", color="black", alpha=0.8)

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

    # Draw reference line for ratio
    ax_lower.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Label and layout
    plot_config.apply(fig=fig, axes=[ax_upper, ax_lower])

    fig.savefig(output_dir / f"{plot_config.name}.pdf")
    plt.close(fig)


def run() -> None:
    # for val, smeared_input in [("leading_kt_z_cut_04_test", False)]:
    for val, smeared_input in [
        ("hybrid_as_input", True),
        ("leading_kt_test", False),
        # ("leading_kt_z_cut_04_test", False),
    ]:
        output_dir = Path("output") / "unfolding" / val
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(f"unfolding_{val}.root")
        logger.info(f"Processing file {filename}")
        logger.info(f"Output dir: {output_dir}")

        # Extract with uproot and convert to BinnedData
        hists = {}
        f = uproot.open(filename)
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

        import seaborn as sns

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
                    name="unfolded_kt_true_pt_60_80",
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
                            # legend=pb.LegendConfig(location="lower left"),
                            legend=pb.LegendConfig(location="center right"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                                pb.AxisConfig("y", label=fr"Ratio to iter {n_iter_for_ratio}", range=(0, 2)),
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
                    name="unfolded_kt_true_pt_40_120",
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
                            # legend=pb.LegendConfig(location="lower left"),
                            legend=pb.LegendConfig(location="center right"),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                                pb.AxisConfig("y", label=fr"Ratio to iter {n_iter_for_ratio}", range=(0, 2)),
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
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c$)"),
                                pb.AxisConfig("y", label=fr"Ratio to iter {n_iter_for_ratio}", range=(0, 2)),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )
            plot_refolded(
                hists=hists,
                projection_func=_project_kt,
                smeared_input=smeared_input,
                measured_bin=helpers.RangeSelector(40, 120),
                plot_config=pb.PlotConfig(
                    name="refolded_kt",
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
                                pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c$)"),
                                # y label is set in the function.
                                pb.AxisConfig(
                                    "y", label="Ratio to smeared" if smeared_input else "Ratio to data", range=(0, 2)
                                ),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )
            plot_refolded(
                hists=hists,
                projection_func=_project_pt,
                smeared_input=smeared_input,
                measured_bin=helpers.RangeSelector(0, 15),
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
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c$)"),
                                # y label is set in the function.
                                pb.AxisConfig(
                                    "y", label="Ratio to smeared" if smeared_input else "Ratio to data", range=(0, 2)
                                ),
                            ],
                        ),
                    ],
                ),
                output_dir=output_dir,
            )

        # TODO: Plot efficiency
        # plot_spectra_comparison(hists, output_dir)
        # plot_spectra_comparison_fine_binned(hists, output_dir)
        # plot_efficiency(hists, output_dir)
        # plot_response_matrix(hists["responseUnscaled"], "response", output_dir)


if __name__ == "__main__":
    # Setup
    helpers.setup_logging()
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)

    # Enable ticks on all sides
    # Unfortunately, some of this is overriding the pachyderm plotting style.
    # That will have to be updated eventually...
    # matplotlib.rcParams["xtick.top"] = True
    # matplotlib.rcParams["xtick.minor.top"] = True
    # matplotlib.rcParams["ytick.right"] = True
    # matplotlib.rcParams["ytick.minor.right"] = True

    run()
