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
    laura: Input,
    mine: Input,
    output_name: str,
    axis: bh.axis.Regular = bh.axis.Regular(30, 0, 150),
    log_y: bool = False,
) -> None:
    # Plot
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    laura_hist = df_to_hist(df=laura.df, attribute=laura.attribute, axis=axis)
    mine_hist = df_to_hist(df=mine.df, attribute=mine.attribute, axis=axis)
    # Normalize
    laura_hist /= np.sum(laura_hist.values)
    mine_hist /= np.sum(mine_hist.values)

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

    ratio = mine_hist / laura_hist
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


print("Loading Laura's...")
print(uproot3.open("laura/AnalysisResults.5403.18q.root").keys())
laura_df = uproot3.open("laura/AnalysisResults.5403.18q.root")[
    "JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_TCRawTree_Data_ConstSub_Incl"
].pandas.df()
print(laura_df.columns)
print("Loading mine...")
mine_iter = uproot3.pandas.iterate("mine/AnalysisResults.18*.root", "tree")
dfs = [df for df in mine_iter]
mine_df = pd.concat(dfs)
print(f"Number of files: {len(dfs)}")
print(mine_df.keys())
# leticia_iter = uproot3.pandas.iterate("leticia/Ev_13_*.root", "JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRawTree_PythiaDef_NoSub_Incl")
# leticia_df = pd.concat([df for df in leticia_iter])
# print(leticia_df)

# Plot jet pt
plot_attribute_compare(
    laura=Input(df=laura_df, attribute="ptJet", name="Laura"),
    mine=Input(df=mine_df, attribute="jet_pt_data", name="Mine"),
    output_name="compare_data",
    log_y=True,
)
# rg.
# Shared pt cuts
# Selection for Laura
mask = laura_df["zg"] < 0.2
laura_df_copy = laura_df.copy()
laura_df_copy.loc[mask, "rg"] = -0.005
# rg
plot_attribute_compare(
    laura=Input(df=laura_df_copy[laura_df_copy["ptJet"] > 40], attribute="rg", name="Laura"),
    mine=Input(df=mine_df[mine_df["jet_pt_data"] > 40], attribute="soft_drop_z_cut_02_data_delta_R", name="Mine"),
    output_name="compare_data_rg",
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
# plot_attribute_compare(laura=Input(df=leticia_df_copy[leticia_df_copy["ptJet"] > 20], attribute="rg", name="Leticia"), mine=Input(df=mine_df[mine_df["jet_pt_data"] > 20], attribute="soft_drop_z_cut_04_data_delta_R", name="Mine"), output_name="compare_det_rg_z_cut_04", axis=bh.axis.Regular(21, -0.02, 0.4))
#
## Leticia
## Plot true pt
# plot_attribute_compare(laura=Input(df=leticia_df, attribute="ptJetMatch", name="Leticia"), mine=Input(df=mine_df, attribute="jet_pt_matched", name="Mine"), output_name="compare_true_leticia", log_y=True)
## Plot det pt
# plot_attribute_compare(laura=Input(df=leticia_df, attribute="ptJet", name="Leticia"), mine=Input(df=mine_df, attribute="jet_pt_data", name="Mine"), output_name="compare_det_leticia", log_y=True)


# np.testing.assert_allclose(mine_df["soft_drop_z_cut_02_data_delta_R"], mine_df["soft_drop_z_cut_04_data_delta_R"])
# print("all close")
