"""Comparison between standard analysis and track skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBNL/UCB
"""

import logging
import pprint
from pathlib import Path

import attr
import awkward as ak
import boost_histogram as bh
import mammoth.helpers
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import uproot
from pachyderm import binned_data


pachyderm.plot.configure()


logger = logging.getLogger(__name__)


@attr.s
class Input:
    name: str = attr.ib()
    arrays: ak.Array = attr.ib()
    attribute: str = attr.ib()


def arrays_to_hist(
    arrays: ak.Array, attribute: str, axis: bh.axis.Regular = bh.axis.Regular(30, 0, 150)
) -> binned_data.BinnedData:
    bh_hist = bh.Histogram(axis, storage=bh.storage.Weight())
    bh_hist.fill(ak.flatten(arrays[attribute], axis=None))

    return binned_data.BinnedData.from_existing_data(bh_hist)


def plot_attribute_compare(
    other: Input,
    mine: Input,
    output_name: str,
    output_dir: Path,
    axis: bh.axis.Regular = bh.axis.Regular(30, 0, 150),
    log_y: bool = False,
    normalize: bool = False,
) -> None:
    # Plot
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    other_hist = arrays_to_hist(arrays=other.arrays, attribute=other.attribute, axis=axis)
    mine_hist = arrays_to_hist(arrays=mine.arrays, attribute=mine.attribute, axis=axis)
    # Normalize
    if normalize:
        other_hist /= np.sum(other_hist.values)
        mine_hist /= np.sum(mine_hist.values)

    ax.errorbar(
        other_hist.axes[0].bin_centers,
        other_hist.values,
        xerr=other_hist.axes[0].bin_widths / 2,
        yerr=other_hist.errors,
        label=other.name,
        linestyle="",
        alpha=0.8,
    )
    ax.errorbar(
        mine_hist.axes[0].bin_centers,
        mine_hist.values,
        xerr=mine_hist.axes[0].bin_widths / 2,
        yerr=mine_hist.errors,
        label=mine.name,
        linestyle="",
        alpha=0.8,
    )

    ratio = mine_hist / other_hist
    ax_ratio.errorbar(
        ratio.axes[0].bin_centers, ratio.values, xerr=ratio.axes[0].bin_widths / 2, yerr=ratio.errors, linestyle=""
    )
    print(f"ratio sum: {np.sum(ratio.values)}")
    print(f"other: {np.sum(other_hist.values)}")
    print(f"mine: {np.sum(mine_hist.values)}")

    ax.set_ylabel("Prob.")
    if log_y:
        ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right")
    ax_ratio.set_ylabel(f"{mine.name}/{other.name}")
    # ax_ratio.set_xlabel(r"$p_{\text{T, det}}$")
    ax_ratio.set_ylim([0.6, 1.4])
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.12,
        bottom=0.105,
        right=0.98,
        top=0.98,
    )
    fig.savefig(output_dir / f"{output_name}.pdf")
    plt.close(fig)


def compare(collision_system: str, standard_filename: Path, track_skim_filename: Path) -> None:
    #standard_tree_name = "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl"
    #if collision_system == "pp":
    #    standard_tree_name = "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_scheme_RawTree_Data_NoSub_Incl"
    #if collision_system == "pythia":
    #    standard_tree_name = "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeTree_PythiaDef_NoSub_Incl"
    standard_tree_name = "tree"
    standard = uproot.open(standard_filename)[standard_tree_name].arrays()
    track_skim = uproot.open(track_skim_filename)["tree"].arrays()
    print(f"standard.type: {standard.type}")
    print(f"track_skim.type: {track_skim.type}")

    output_dir = Path("comparison") / "trackSkim" / collision_system
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_attribute_compare(
        other=Input(arrays=standard, attribute="data_jet_pt", name="Standard"),
        mine=Input(arrays=track_skim, attribute="data_jet_pt", name="Track skim"),
        output_name="jet_pt",
        log_y=True,
        normalize=True,
        axis=bh.axis.Regular(50, 0, 100),
        output_dir=output_dir,
    )
    standard_jet_pt = standard["data_jet_pt"]
    track_skim_jet_pt = track_skim["data_jet_pt"]

    #logger.info("jet pt")
    #_arr = ak.zip({"s": standard_jet_pt, "t": track_skim_jet_pt})
    #logger.info(pprint.pformat(_arr.to_list()))

    for grooming_method in ["dynamical_kt", "soft_drop_z_cut_02"]:
        logger.info(f"Plotting method \"{grooming_method}\"")
        plot_attribute_compare(
            other=Input(arrays=standard, attribute=f"{grooming_method}_data_kt", name="Standard"),
            mine=Input(arrays=track_skim, attribute=f"{grooming_method}_data_kt", name="Track skim"),
            output_name=f"{grooming_method}_data_kt",
            log_y=True,
            normalize=True,
            axis=bh.axis.Regular(50, 0, 10),
            output_dir=output_dir,
        )
        standard_kt = standard[f"{grooming_method}_data_kt"]
        track_skim_kt = track_skim[f"{grooming_method}_data_kt"]

        logger.info(f"standard_kt: {standard_kt}")
        logger.info(f"track_skim_kt: {track_skim_kt}")

        plot_attribute_compare(
            other=Input(arrays=standard, attribute=f"{grooming_method}_data_delta_R", name="Standard"),
            mine=Input(arrays=track_skim, attribute=f"{grooming_method}_data_delta_R", name="Track skim"),
            output_name=f"{grooming_method}_data_delta_R",
            normalize=True,
            axis=bh.axis.Regular(50, 0, 0.6),
            output_dir=output_dir,
        )
        standard_rg = standard[f"{grooming_method}_data_delta_R"]
        track_skim_rg = track_skim[f"{grooming_method}_data_delta_R"]

        #logger.info("delta_R")
        #_arr = ak.zip({"s": standard_rg, "t": track_skim_rg})
        #logger.info(pprint.pformat(_arr.to_list()))

        #import IPython; IPython.embed()


        #logger.info(f"standard_rg: {standard_rg}")
        #logger.info(f"track_skim_rg: {track_skim_rg}")

        plot_attribute_compare(
            other=Input(arrays=standard, attribute=f"{grooming_method}_data_z", name="Standard"),
            mine=Input(arrays=track_skim, attribute=f"{grooming_method}_data_z", name="Track skim"),
            output_name=f"{grooming_method}_data_z",
            normalize=True,
            axis=bh.axis.Regular(50, 0, 0.5),
            output_dir=output_dir,
        )


def run(collision_system: str) -> None:
    mammoth.helpers.setup_logging()
    path_to_mammoth = Path("/Users/re239/code/alice/mammoth")
    compare(
        collision_system=collision_system,
        standard_filename=path_to_mammoth / f"projects/framework/{collision_system}/1/skim/AnalysisResults.repaired.00_iterative_splittings.root",
        track_skim_filename=path_to_mammoth / f"projects/framework/{collision_system}/1/skim/skim_output.root",
    )


if __name__ == "__main__":
    run(collision_system="pythia")
