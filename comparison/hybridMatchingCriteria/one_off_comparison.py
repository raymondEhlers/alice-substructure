#!/usr/bin/env python

"""

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import attr
import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import pandas as pd
import uproot3
from pachyderm import binned_data

pachyderm.plot.configure()


@attr.s
class Input:
    name: str = attr.ib()
    df: pd.DataFrame = attr.ib()
    attribute: str = attr.ib()


def df_to_hist(
    df: pd.DataFrame, attribute: str, axis: bh.axis.Regular = bh.axis.Regular(30, 0, 150)
) -> binned_data.BinnedData:
    bh_hist = bh.Histogram(axis, storage=bh.storage.Weight())
    bh_hist.fill(df[attribute].to_numpy())

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
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    other_hist = df_to_hist(df=other.df, attribute=other.attribute, axis=axis)
    mine_hist = df_to_hist(df=mine.df, attribute=mine.attribute, axis=axis)
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
    ax_ratio.set_ylim([0.9, 1.1])
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
    fig.savefig(f"{output_name}.pdf")
    plt.close(fig)


pt_hard_to_train_number = {
    14: (6011, 6016),
    20: (6012, 6017),
}

pt_hard_bin = 14
dm_train_number, lm_train_number = pt_hard_to_train_number[pt_hard_bin]

distance_matching_file = uproot3.open(f"../../trains/embedPythia/{dm_train_number}/AnalysisResults.18q.root")
df_dm = distance_matching_file[
    "AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"
].pandas.df()
# Double counting cut
df_dm = df_dm[df_dm["det_level_leading_track_pt"] >= df_dm["data_leading_track_pt"]]
print("Done loading distance matching (dm)!")
label_matching_file = uproot3.open(f"../../trains/embedPythia/{lm_train_number}/AnalysisResults.18q.root")
df_lm = label_matching_file[
    "AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"
].pandas.df()
# Double counting cut
df_lm = df_lm[df_lm["det_level_leading_track_pt"] >= df_lm["data_leading_track_pt"]]
print("Done loading label matching (lm)!")

plot_attribute_compare(
    other=Input(df=df_dm, attribute="leading_kt_z_cut_02_hybrid_det_level_matching_leading", name="Distance"),
    mine=Input(df=df_lm, attribute="leading_kt_z_cut_02_hybrid_det_level_matching_leading", name="Label"),
    output_name=f"hybrid_det_level_leading_{pt_hard_bin}",
    log_y=True,
    normalize=True,
    axis=bh.axis.Regular(5, -1.5, 3.5),
)
plot_attribute_compare(
    other=Input(df=df_dm, attribute="leading_kt_z_cut_02_hybrid_det_level_matching_subleading", name="Distance"),
    mine=Input(df=df_lm, attribute="leading_kt_z_cut_02_hybrid_det_level_matching_subleading", name="Label"),
    output_name=f"hybrid_det_level_subleading_{pt_hard_bin}",
    log_y=True,
    normalize=True,
    axis=bh.axis.Regular(5, -1.5, 3.5),
)
plot_attribute_compare(
    other=Input(df=df_dm, attribute="leading_kt_z_cut_02_det_level_true_matching_leading", name="Distance"),
    mine=Input(df=df_lm, attribute="leading_kt_z_cut_02_det_level_true_matching_leading", name="Label"),
    output_name=f"det_level_true_leading_{pt_hard_bin}",
    log_y=True,
    normalize=True,
    axis=bh.axis.Regular(5, -1.5, 3.5),
)
plot_attribute_compare(
    other=Input(df=df_dm, attribute="leading_kt_z_cut_02_det_level_true_matching_subleading", name="Distance"),
    mine=Input(df=df_lm, attribute="leading_kt_z_cut_02_det_level_true_matching_subleading", name="Label"),
    output_name=f"det_level_true_subleading_{pt_hard_bin}",
    log_y=True,
    normalize=True,
    axis=bh.axis.Regular(5, -1.5, 3.5),
)
