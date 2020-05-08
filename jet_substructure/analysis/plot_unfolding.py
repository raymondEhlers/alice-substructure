""" Plot unfolding.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Mapping

import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import uproot
from pachyderm import binned_data

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)

pachyderm.plot.configure()


def plot_unfolded_kt(hists: Mapping[str, binned_data.BinnedData], output_dir: Path) -> None:
    """ Plot unfolded.

    """
    logger.debug("Plotting unfolded kt")
    # Setup
    # True jet pt range
    true_jet_pt_range = helpers.RangeSelector(60, 80)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    ax_upper, ax_lower = axes

    # We need the efficiency in the jet pt range that we actually want to measure.
    efficiency = hists[f"correff{true_jet_pt_range.min:g}-{true_jet_pt_range.max:g}"]

    # True
    # Project true onto the kt axis. We use boost histogram for the convince.
    bh_true_2d = hists["true"].to_boost_histogram()
    hist_true = binned_data.BinnedData.from_existing_data(bh_true_2d[:, bh.loc(60) : bh.loc(80) : bh.sum])
    # Apply the efficiency.
    hist_true /= efficiency
    # Then normalize by the integral (sum) and bin width.
    hist_true /= np.sum(hist_true.values)
    hist_true /= hist_true.axes[0].bin_widths

    # Determine ratio denominator.
    n_iter = 6
    # Retrieve the hist and normalize it properly.
    bh_ratio_denominator_2d = hists[f"Bayesian_Unfoldediter{n_iter}"].to_boost_histogram()
    # TODO: What is the right jet pt range here??
    h_ratio_denominator = binned_data.BinnedData.from_existing_data(
        bh_ratio_denominator_2d[:, bh.loc(60) : bh.loc(80) : bh.sum]
    )
    h_ratio_denominator /= efficiency
    h_ratio_denominator /= np.sum(h_ratio_denominator.values)
    h_ratio_denominator /= h_ratio_denominator.axes[0].bin_widths

    for i in range(1, 10):
        # Retrieve the hist and normalize it properly.
        bh_2d = hists[f"Bayesian_Unfoldediter{i}"].to_boost_histogram()
        # TODO: What is the right jet pt range here??
        hist = binned_data.BinnedData.from_existing_data(bh_2d[:, bh.loc(60) : bh.loc(80) : bh.sum])
        hist /= efficiency
        hist /= np.sum(hist.values)
        hist /= hist.axes[0].bin_widths

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

    # Plot truth
    # ax_upper.errorbar(hist_true.axes[0].bin_centers, hist_true.values, xerr=hist_true.axes[0].bin_widths / 2, yerr=hist_true.errors, label = "True",
    #            marker="o", linestyle="", color="red")

    # Draw reference line for ratio
    ax_lower.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # trueptd, which comes from the fully efficient truth and therefore doesn't need the kinematic efficiency correction
    # hist_true = histogram.Histogram1D.from_existing_hist(hists["trueptd"])
    ## Don't need to correct for the kinematic efficiency here.
    # value, error = hist_true.counts_in_interval(min_bin=0, max_bin=len(hist_true.x))
    # hist_true /= value
    # hist_true /= hist_true.bin_widths
    # ax.errorbar(hist_true.x, hist_true.y, xerr=hist_true.bin_widths / 2, yerr=hist_true.errors, label = "True (ptd)",
    #            marker="o", linestyle="")

    # Finalize presentation for upper panel
    ax_upper.set_yscale("log")
    ax_upper.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100))
    ax_upper.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax_upper.legend(loc="upper right", frameon=False)
    ax_upper.set_ylabel(r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$")

    # Finalize presentation for lower panel
    ax_lower.set_xlabel(r"$k_{\text{T}}\:(\text{GeV}/c$)")
    ax_lower.set_ylabel(fr"Ratio to iter {n_iter}")
    ax_lower.set_ylim([0, 2])
    fig.align_ylabels()

    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        wspace=0,
        hspace=0,
        # Reduce external spacing
        left=0.12,
        right=0.98,
        top=0.98,
        bottom=0.07,
    )
    fig.savefig(output_dir / "unfolded_kt.pdf")
    plt.close(fig)


def plot_unfolded_pt(hists: Mapping[str, binned_data.BinnedData], output_dir: Path) -> None:
    """ Plot unfolded.

    """
    logger.debug("Plotting unfolded pt")
    # Setup
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    ax_upper, ax_lower = axes

    # NOTE: We don't output the pt efficiency (?)
    # TODO: Do we need to restrict the kt range here?

    # True
    # Project true onto the pt axis.
    bh_true_2d = hists["true"].to_boost_histogram()
    hist_true = binned_data.BinnedData.from_existing_data(bh_true_2d[:: bh.sum, :])
    hist_true /= np.sum(hist_true.values)
    hist_true /= hist_true.axes[0].bin_widths

    # Determine ratio denominator.
    n_iter = 4
    # Retrieve the hist and normalize it properly.
    bh_ratio_denominator_2d = hists[f"Bayesian_Unfoldediter{n_iter}"].to_boost_histogram()
    h_ratio_denominator = binned_data.BinnedData.from_existing_data(bh_ratio_denominator_2d[:: bh.sum, :])
    h_ratio_denominator /= np.sum(h_ratio_denominator.values)
    h_ratio_denominator /= h_ratio_denominator.axes[0].bin_widths

    for i in range(1, 10):
        # Retrieve the hist and normalize it properly.
        bh_2d = hists[f"Bayesian_Unfoldediter{i}"].to_boost_histogram()
        hist = binned_data.BinnedData.from_existing_data(bh_2d[:: bh.sum, :])
        hist /= np.sum(hist.values)
        hist /= hist.axes[0].bin_widths

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

    # Plot truth
    # ax_upper.errorbar(hist_true.axes[0].bin_centers, hist_true.values, xerr=hist_true.axes[0].bin_widths / 2, yerr=hist_true.errors, label = "True",
    #            marker="o", linestyle="", color="red")

    # Draw reference line for ratio
    ax_lower.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # trueptd, which comes from the fully efficient truth and therefore doesn't need the kinematic efficiency correction
    # hist_true = histogram.Histogram1D.from_existing_hist(hists["trueptd"])
    ## Don't need to correct for the kinematic efficiency here.
    # value, error = hist_true.counts_in_interval(min_bin=0, max_bin=len(hist_true.x))
    # hist_true /= value
    # hist_true /= hist_true.bin_widths
    # ax.errorbar(hist_true.x, hist_true.y, xerr=hist_true.bin_widths / 2, yerr=hist_true.errors, label = "True (ptd)",
    #            marker="o", linestyle="")

    # Finalize presentation for upper panel
    ax_upper.set_yscale("log")
    ax_upper.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100))
    ax_upper.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax_upper.legend(loc="upper right", frameon=False)
    ax_upper.set_ylabel(r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$")

    # Finalize presentation for lower panel
    ax_lower.set_xlabel(r"$p_{\text{T}}\:(\text{GeV}/c$)")
    ax_lower.set_ylabel(fr"Ratio to iter {n_iter}")
    ax_lower.set_ylim([0, 2])
    fig.align_ylabels()

    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        wspace=0,
        hspace=0,
        # Reduce external spacing
        left=0.12,
        right=0.98,
        top=0.98,
        bottom=0.07,
    )
    fig.savefig(output_dir / "unfolded_pt.pdf")
    plt.close(fig)


def plot_refolded_kt(hists: Mapping[str, binned_data.BinnedData], smeared_input: bool, output_dir: Path) -> None:
    """

    """
    logger.debug("Plotting refolded kt")
    # Setup
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    ax_upper, ax_lower = axes

    # Raw
    bh_hist_raw_2d = hists["raw"].to_boost_histogram()
    # Project
    hist_raw = binned_data.BinnedData.from_existing_data(bh_hist_raw_2d[:, bh.loc(40) : bh.loc(120) : bh.sum])
    # Normalize
    hist_raw /= np.sum(hist_raw.values)
    hist_raw /= hist_raw.axes[0].bin_widths
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
    bh_hist_smeared_2d = hists["smeared"].to_boost_histogram()
    hist_smeared = binned_data.BinnedData.from_existing_data(bh_hist_smeared_2d[:, bh.loc(40) : bh.loc(120) : bh.sum])
    hist_smeared /= np.sum(hist_smeared.values)
    hist_smeared /= hist_smeared.axes[0].bin_widths
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
        bh_hist = hists[f"Bayesian_Foldediter{i}"].to_boost_histogram()
        hist = binned_data.BinnedData.from_existing_data(bh_hist[:, bh.loc(40) : bh.loc(120) : bh.sum])
        hist /= np.sum(hist.values)
        hist /= hist.axes[0].bin_widths
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

    # Finalize presentation for upper panel
    ax_upper.set_yscale("log")
    ax_upper.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100))
    ax_upper.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax_upper.legend(loc="upper right", frameon=False)
    ax_upper.set_ylabel(r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$")

    # Finalize presentation for lower panel
    ax_lower.set_xlabel(r"$k_{\text{T}}\:(\text{GeV}/c)$")
    ax_lower.set_ylabel(r"Ratio to smeared" if smeared_input else "Ratio to data")
    ax_lower.set_ylim([0, 2])
    fig.align_ylabels()

    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        wspace=0,
        hspace=0,
        # Reduce external spacing
        left=0.12,
        right=0.98,
        top=0.98,
        bottom=0.07,
    )
    ax_upper.legend(frameon=False, loc="upper right")
    ax_upper.set_yscale("log")

    fig.savefig(output_dir / "refolded_kt.pdf")
    plt.close(fig)


def plot_refolded_pt(hists: Mapping[str, binned_data.BinnedData], smeared_input: bool, output_dir: Path) -> None:
    """

    """
    logger.debug("Plotting refolded pt")
    # Setup
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    ax_upper, ax_lower = axes

    # Raw
    bh_hist_raw_2d = hists["raw"].to_boost_histogram()
    # Project
    hist_raw = binned_data.BinnedData.from_existing_data(bh_hist_raw_2d[:: bh.sum, :])
    # Normalize
    hist_raw /= np.sum(hist_raw.values)
    hist_raw /= hist_raw.axes[0].bin_widths
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
    bh_hist_smeared_2d = hists["smeared"].to_boost_histogram()
    hist_smeared = binned_data.BinnedData.from_existing_data(bh_hist_smeared_2d[:: bh.sum, :])
    hist_smeared /= np.sum(hist_smeared.values)
    hist_smeared /= hist_smeared.axes[0].bin_widths
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
        bh_hist = hists[f"Bayesian_Foldediter{i}"].to_boost_histogram()
        hist = binned_data.BinnedData.from_existing_data(bh_hist[:: bh.sum, :])
        hist /= np.sum(hist.values)
        hist /= hist.axes[0].bin_widths
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

    # Finalize presentation for upper panel
    ax_upper.set_yscale("log")
    ax_upper.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100))
    ax_upper.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax_upper.legend(loc="upper right", frameon=False)
    ax_upper.set_ylabel(r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$")

    # Finalize presentation for lower panel
    ax_lower.set_xlabel(r"$p_{\text{T}}\:(\text{GeV}/c)$")
    ax_lower.set_ylabel(r"Ratio to smeared" if smeared_input else "Ratio to data")
    ax_lower.set_ylim([0, 2])
    fig.align_ylabels()

    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        wspace=0,
        hspace=0,
        # Reduce external spacing
        left=0.12,
        right=0.98,
        top=0.98,
        bottom=0.07,
    )
    ax_upper.legend(frameon=False, loc="upper right")
    ax_upper.set_yscale("log")

    fig.savefig(output_dir / "refolded_pt.pdf")
    plt.close(fig)


def run() -> None:
    # for val, smeared_input in [("leading_kt_z_cut_04_test", False)]:
    for val, smeared_input in [
        ("hybrid_as_input", True),
        ("leading_kt_test", False),
        ("leading_kt_z_cut_04_test", False),
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
            plot_unfolded_kt(hists, output_dir)
            plot_unfolded_pt(hists, output_dir)
            plot_refolded_kt(hists, smeared_input, output_dir)
            plot_refolded_pt(hists, smeared_input, output_dir)
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
    matplotlib.rcParams["xtick.top"] = True
    matplotlib.rcParams["xtick.minor.top"] = True
    matplotlib.rcParams["ytick.right"] = True
    matplotlib.rcParams["ytick.minor.right"] = True

    run()
