from pathlib import Path

import attr
import awkward as ak
import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import uproot
from pachyderm import binned_data


pachyderm.plot.configure()


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
    fig.savefig(Path("comparison/trackSkim") / f"{output_name}.pdf")
    plt.close(fig)


def compare(standard_filename: Path, track_skim_filename: Path) -> None:
    standard = uproot.open(standard_filename)[
        "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl"
    ].arrays()
    track_skim = uproot.open(track_skim_filename)["tree"].arrays()
    print(f"standard.type: {standard.type}")
    print(f"track_skim.type: {track_skim.type}")

    plot_attribute_compare(
        other=Input(arrays=standard, attribute="data_jet_pt", name="Standard"),
        mine=Input(arrays=track_skim, attribute="data_jet_pt", name="Track skim"),
        output_name="jet_pt",
        log_y=True,
        # normalize=True,
        axis=bh.axis.Regular(100, 0, 100),
    )

    plot_attribute_compare(
        other=Input(arrays=standard, attribute="dynamical_kt_data_kt", name="Standard"),
        mine=Input(arrays=track_skim, attribute="dynamical_kt_data_kt", name="Track skim"),
        output_name="dyg_kt_data_kt",
        log_y=True,
        normalize=True,
        axis=bh.axis.Regular(20, 0, 10),
    )

    plot_attribute_compare(
        other=Input(arrays=standard, attribute="dynamical_kt_data_delta_R", name="Standard"),
        mine=Input(arrays=track_skim, attribute="dynamical_kt_data_delta_R", name="Track skim"),
        output_name="dyg_kt_data_delta_R",
        normalize=True,
        axis=bh.axis.Regular(20, 0, 0.6),
    )

    plot_attribute_compare(
        other=Input(arrays=standard, attribute="dynamical_kt_data_z", name="Standard"),
        mine=Input(arrays=track_skim, attribute="dynamical_kt_data_z", name="Track skim"),
        output_name="dyg_kt_data_z",
        normalize=True,
        axis=bh.axis.Regular(20, 0, 0.5),
    )


if __name__ == "__main__":
    compare(
        # TODO: Do the proper skim for comparison...
        standard_filename=Path("/software/rehlers/dev/mammoth/projects/framework/PbPb/AnalysisResults.root"),
        track_skim_filename=Path("/software/rehlers/dev/mammoth/projects/framework/PbPb/skim/skim_output.root"),
    )
