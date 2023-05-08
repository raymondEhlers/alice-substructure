#!/usr/bin/env python

"""

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import pickle
from pathlib import Path
from typing import Optional

import attr
import boost_histogram as bh
import IPython
import matplotlib
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
    laura: Input,
    mine: Input,
    output_name: str,
    axis: bh.axis.Regular = bh.axis.Regular(30, 0, 150),
    log_y: bool = False,
    external_ref: Optional[binned_data.BinnedData] = None,
) -> None:
    # Plot
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    laura_hist = df_to_hist(df=laura.df, attribute=laura.attribute, axis=axis)
    mine_hist = df_to_hist(df=mine.df, attribute=mine.attribute, axis=axis)
    # Normalize
    laura_hist /= np.sum(laura_hist.values)
    mine_hist /= np.sum(mine_hist.values)
    if external_ref:
        external_ref /= np.sum(external_ref.values)

    ax.errorbar(
        laura_hist.axes[0].bin_centers,
        laura_hist.values,
        xerr=laura_hist.axes[0].bin_widths / 2,
        yerr=laura_hist.errors,
        label=laura.name,
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
    if external_ref:
        ax.errorbar(
            external_ref.axes[0].bin_centers,
            external_ref.values,
            xerr=external_ref.axes[0].bin_widths / 2,
            yerr=external_ref.errors,
            label="Pre-skim",
            linestyle="",
            alpha=0.8,
        )

    ratio = mine_hist / laura_hist
    ax_ratio.errorbar(
        ratio.axes[0].bin_centers, ratio.values, xerr=ratio.axes[0].bin_widths / 2, yerr=ratio.errors, linestyle=""
    )
    if external_ref:
        ratio = external_ref / laura_hist
        ax_ratio.errorbar(
            ratio.axes[0].bin_centers, ratio.values, xerr=ratio.axes[0].bin_widths / 2, yerr=ratio.errors, linestyle=""
        )

    ax.set_ylabel("Prob.")
    if log_y:
        ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right")
    ax_ratio.set_ylabel(f"{mine.name}/{laura.name}")
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


def plot_centrality_hist(hist: binned_data.BinnedData, output_name: str) -> None:
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        "vmin": hist.values[hist.values > 0].min(),
        # "vmin": 1e-4,
        "vmax": hist.values.max(),
    }

    # Plot
    mesh = ax.pcolormesh(
        hist.axes[0].bin_edges.T,
        hist.axes[1].bin_edges.T,
        hist.values.T,
        norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)
    ax.set_xlim([25, 55])
    ax.set_ylim([0, 4000])

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
    fig.savefig(f"centrality_{output_name}.pdf")
    plt.close(fig)


laura_hists = uproot3.open("laura/AnalysisResults_PbPbMC_pThard13.root")[
    "JetSubstructure_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_TCRaw_EventSub_Incl"
]
print(laura_hists)  # noqa: T201
# This is wayyyyyy too hard to select...
laura_centrality = binned_data.BinnedData.from_existing_data(
    [h for h in laura_hists if hasattr(h, "name") and h.name == b"MultCentCorrelation_selected"][0]
)
mine_hists = uproot3.open("mine/hists/hists.root")
print(mine_hists.keys())  # noqa: T201
# Mine is easier, for some reason.
mine_centrality = binned_data.BinnedData.from_existing_data(mine_hists[b"MultCentCorrelation_selected"])
# Jet pt hists from before the skim
with Path("hists.pkl").open("rb") as f:
    mine_pre_skim_input = pickle.load(f)
    # Convert to BinnedData
    mine_pre_skim = {}
    for k, v in mine_pre_skim_input.items():
        mine_pre_skim[k] = binned_data.BinnedData.from_existing_data(v[:: bh.rebin(5)])

# Plot the centrality hists...
plot_centrality_hist(hist=laura_centrality, output_name="laura")
plot_centrality_hist(hist=mine_centrality, output_name="mine")

print("Loading Laura's...")
# laura_df = uproot3.open("laura/AnalysisResults_PbPbMC_pThard13.root")["JetSubstructure_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_TCRawTree_EventSub_Incl"].pandas.df()
laura_df = uproot3.open("laura/5570_AnalysisResults.18q.root")[
    "JetSubstructure_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_TCRawTree_EventSub_Incl"
].pandas.df()
print(laura_df.columns)
laura_dc_mask = (laura_df["LeadingTrackPtMatch"] >= laura_df["LeadingTrackPt"]) & (
    laura_df["LeadingTrackPtDet"] >= laura_df["LeadingTrackPt"]
)
print(f"Laura: double counting cut is cutting {1 - (np.count_nonzero(laura_dc_mask) / len(laura_df))}")
laura_df = laura_df[laura_dc_mask]

IPython.embed()  # type: ignore[no-untyped-call]
print("Loading mine...")
mine_iter = uproot3.pandas.iterate("mine/AnalysisResults.18q.01.chunk*_iterative_splittings.root", "tree")
mine_df = pd.concat(list(mine_iter))
print(mine_df.keys())
# leticia_iter = uproot3.pandas.iterate("leticia/Ev_13_*.root", "JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRawTree_PythiaDef_NoSub_Incl")
# leticia_df = pd.concat([df for df in leticia_iter])
# print(leticia_df)
# Apply the double counting cut, but between hybrid and true to match Laura's.
double_counting_mask = (mine_df["leading_track_true"] >= mine_df["leading_track_hybrid"]) & (
    mine_df["leading_track_det_level"] >= mine_df["leading_track_hybrid"]
)
print(f"Mine: double counting cut is cutting {1 - (np.count_nonzero(double_counting_mask) / len(mine_df))}")
mine_df = mine_df[double_counting_mask]

# Plot true pt
plot_attribute_compare(
    laura=Input(df=laura_df, attribute="ptJetMatch", name="Laura"),
    mine=Input(df=mine_df, attribute="jet_pt_true", name="Mine"),
    output_name="compare_true",
    log_y=True,
    external_ref=mine_pre_skim["jet_pt_true"],
)
# Plot hybrid pt
plot_attribute_compare(
    laura=Input(df=laura_df, attribute="ptJet", name="Laura"),
    mine=Input(df=mine_df, attribute="jet_pt_hybrid", name="Mine"),
    output_name="compare_hybrid",
    log_y=True,
    external_ref=mine_pre_skim["jet_pt_hybrid"],
)
# Plot det level pt
plot_attribute_compare(
    laura=Input(df=laura_df, attribute="ptJetDet", name="Laura"),
    mine=Input(df=mine_df, attribute="jet_pt_det_level", name="Mine"),
    output_name="compare_det",
    log_y=True,
    external_ref=mine_pre_skim["jet_pt_det_level"],
)
# rg.
# Shared pt cuts
# Selection for Laura
mask = laura_df["zg"] < 0.2
laura_df_copy = laura_df.copy()
laura_df_copy.loc[mask, "rg"] = -0.005
mask_match = laura_df["zgMatch"] < 0.2
laura_df_copy.loc[mask_match, "rgMatch"] = -0.005
mask_det_level = laura_df["zgDet"] < 0.2
laura_df_copy.loc[mask_det_level, "rgDet"] = -0.005
# True
plot_attribute_compare(
    laura=Input(df=laura_df_copy[laura_df_copy["ptJetMatch"] > 20], attribute="rgMatch", name="Laura"),
    mine=Input(df=mine_df[mine_df["jet_pt_true"] > 20], attribute="soft_drop_z_cut_02_true_delta_R", name="Mine"),
    output_name="compare_true_rg",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# Hybrid
plot_attribute_compare(
    laura=Input(df=laura_df_copy[laura_df_copy["ptJet"] > 20], attribute="rg", name="Laura"),
    mine=Input(df=mine_df[mine_df["jet_pt_hybrid"] > 20], attribute="soft_drop_z_cut_02_hybrid_delta_R", name="Mine"),
    output_name="compare_hybrid_rg",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# Det Level
plot_attribute_compare(
    laura=Input(df=laura_df_copy[laura_df_copy["ptJetDet"] > 20], attribute="rgDet", name="Laura"),
    mine=Input(
        df=mine_df[mine_df["jet_pt_det_level"] > 20], attribute="soft_drop_z_cut_02_det_level_delta_R", name="Mine"
    ),
    output_name="compare_det_rg",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)

# rg z cut 04.
# Selection for Leticia
# mask = leticia_df["zg"] < 0.4
# leticia_df_copy = leticia_df.copy()
# leticia_df_copy.loc[mask, "rg"] = -0.005
# mask_true = leticia_df["zgMatch"] < 0.4
# leticia_df_copy.loc[mask_true, "rgMatch"] = -0.005
# plot_attribute_compare(laura=Input(df=leticia_df_copy[leticia_df_copy["ptJetMatch"] > 20], attribute="rgMatch", name="Leticia"), mine=Input(df=mine_df[mine_df["jet_pt_true"] > 20], attribute="soft_drop_z_cut_04_true_delta_R", name="Mine"), output_name="compare_true_rg_z_cut_04", axis=bh.axis.Regular(21, -0.02, 0.4))
# plot_attribute_compare(laura=Input(df=leticia_df_copy[leticia_df_copy["ptJet"] > 20], attribute="rg", name="Leticia"), mine=Input(df=mine_df[mine_df["jet_pt_hybrid"] > 20], attribute="soft_drop_z_cut_04_hybrid_delta_R", name="Mine"), output_name="compare_det_rg_z_cut_04", axis=bh.axis.Regular(21, -0.02, 0.4))
#
## Leticia
## Plot true pt
# plot_attribute_compare(laura=Input(df=leticia_df, attribute="ptJetMatch", name="Leticia"), mine=Input(df=mine_df, attribute="jet_pt_matched", name="Mine"), output_name="compare_true_leticia", log_y=True)
## Plot det pt
# plot_attribute_compare(laura=Input(df=leticia_df, attribute="ptJet", name="Leticia"), mine=Input(df=mine_df, attribute="jet_pt_data", name="Mine"), output_name="compare_det_leticia", log_y=True)


# np.testing.assert_allclose(mine_df["soft_drop_z_cut_02_data_delta_R"], mine_df["soft_drop_z_cut_04_data_delta_R"])
# print("all close")
