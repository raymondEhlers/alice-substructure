#!/usr/bin/env python

""" One off comparison between my skimming and a simplified cross check task.

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


_default_axis = bh.axis.Regular(30, 0, 150)

def df_to_hist(
    df: pd.DataFrame, attribute: str, axis: bh.axis.Regular = _default_axis
) -> binned_data.BinnedData:
    bh_hist = bh.Histogram(axis, storage=bh.storage.Weight())
    bh_hist.fill(df[attribute].to_numpy())

    return binned_data.BinnedData.from_existing_data(bh_hist)


def plot_attribute_compare(
    other: Input,
    mine: Input,
    output_name: str,
    axis: bh.axis.Regular = _default_axis,
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
    print(f"ratio sum: {np.sum(ratio.values)}")  # noqa: T201
    print(f"other: {np.sum(other_hist.values)}")  # noqa: T201
    print(f"mine: {np.sum(mine_hist.values)}")  # noqa: T201

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


# Same input file for both of ours.
original_skim_file = uproot3.open(
    "../../trains/embedPythia/5966/skim/AnalysisResults.18q.repaired_iterative_splittings.root"
)
# And then there is my separate skim input.
input_skim = uproot3.open(
    "../../trains/embedPythia/5966/skim/AnalysisResults.18q.repaired_iterative_splittings_new.root"
)

print("Loading cross check...")
original_df = original_skim_file["tree"].arrays(original_skim_file["tree"].keys(), library="pd")
print(original_df.columns)
# Apply double counting mask
# import IPython; IPython.embed()
cross_check_dc_mask = original_df["leading_track_det_level"] >= original_df["leading_track_hybrid"]
print(f"Cross check: double counting cut is cutting {1 - (np.count_nonzero(cross_check_dc_mask) / len(original_df))}")
original_df = original_df[cross_check_dc_mask]
# laura_dc_mask = (original_df["LeadingTrackPtMatch"] >= original_df["LeadingTrackPt"]) & (original_df["LeadingTrackPtDet"] >= original_df["LeadingTrackPt"])
# original_df = original_df[laura_dc_mask]
print("Loading skim...")
mine_df = input_skim["tree"].arrays(input_skim["tree"].keys(), library="pd")
print(list(mine_df.keys()))
mine_dc_mask = mine_df["leading_track_det_level"] >= mine_df["leading_track_hybrid"]
mine_df = mine_df[mine_dc_mask]
# leticia_iter = uproot3.pandas.iterate("leticia/Ev_13_*.root", "JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRawTree_PythiaDef_NoSub_Incl")
# leticia_df = pd.concat([df for df in leticia_iter])
# print(leticia_df)
# Apply the double counting cut, but between hybrid and true to match Laura's.
# double_counting_mask = (mine_df["leading_track_true"] >= mine_df["leading_track_hybrid"]) & (mine_df["leading_track_det_level"] >= mine_df["leading_track_hybrid"])
# print(f"Mine: double counting cut is cutting {1 - (np.count_nonzero(double_counting_mask) / len(mine_df))}")
# mine_df = mine_df[double_counting_mask]

# They're run side-by-side, so we expect exact matching, and we can just check the arrays.
# Cross check. Make sure that we have matching columns so we don't miss anything.
assert sorted(original_df.columns) == sorted(mine_df.columns)

# Now let's check the arrays.
for col in original_df.columns:
    # Deal with this below. Some conventions changed, so it's not as trivial.
    if "matching" in col:
        continue
    print(f"Checking column {col}")  # noqa: T201
    try:
        # Types vary a bit, so we need to reduce the absolute tolerance just a little bit.
        np.testing.assert_allclose(
            original_df[col].to_numpy(), mine_df[col].to_numpy(), atol=1e-5,
        )
        print("Success")  # noqa: T201
    except AssertionError as e:
        print(f"cross_check: {original_df[col]}")  # noqa: T201
        print(f"mine: {mine_df[col]}")  # noqa: T201
        print(e)  # noqa: T201
        import IPython

        IPython.embed()  # type: ignore[no-untyped-call]
        raise (e)
        # Continue from here so we can check the rest...

# Now deal with matching

# Check matching
# We can only check 1, 2, and 3 because the conventions changed a bit.
grooming_methods = [
    "dynamical_z",
    "dynamical_kt",
    "dynamical_time",
    "leading_kt",
    "leading_kt_z_cut_02",
    "leading_kt_z_cut_04",
    "soft_drop_z_cut_02",
    "soft_drop_z_cut_04",
]
for grooming_method in grooming_methods:
    for matching_type in ["hybrid_det_level", "det_level_true"]:
        for subjet_type in ["leading", "subleading"]:
            # print(f'cross_check: {original_df[f"leading_kt_z_cut_02_{matching_type}_matching_{subjet_type}"]}')
            # print(f'mine: {mine_df[f"leading_kt_z_cut_02_{matching_type}_matching_{subjet_type}"]}')
            # for value in [1, 2, 3]:
            #    print(f"Grooming method: {grooming_method}, matching type: {matching_type}, subjet_type: {subjet_type}, value: {value}")
            #    try:
            #        np.testing.assert_allclose(
            #            original_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"][
            #                original_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"] == value
            #            ].to_numpy(),
            #            mine_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"][
            #                mine_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"] == value
            #            ].to_numpy(),
            #        )
            #        print("Success")
            #    except AssertionError as e:
            #        print(f'cross_check: {original_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"]}')
            #        print(f'mine: {mine_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"]}')
            #        print(e)
            #        # Continue from here so we can check the rest...
            print(f"Grooming method: {grooming_method}, matching type: {matching_type}, subjet_type: {subjet_type}")
            try:
                original_array = original_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"].to_numpy()
                # Ignore values less than 1
                original_array[original_array < 1] = -10
                # Ignore values less than 1
                mine_array = mine_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"].to_numpy()
                mine_array[mine_array < 1] = -10
                original_array[original_array < 1] = -10
                np.testing.assert_allclose(
                    original_array, mine_array,
                )
                print("Success")
            except AssertionError as e:
                print(f'cross_check: {original_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"]}')
                print(f'mine: {mine_df[f"{grooming_method}_{matching_type}_matching_{subjet_type}"]}')
                print(e)
                # Continue from here so we can check the rest...


# Plot true pt
plot_attribute_compare(
    other=Input(df=original_df, attribute="jet_pt_true", name="Cross check"),
    mine=Input(df=mine_df, attribute="jet_pt_true", name="Mine"),
    output_name="compare_true",
    log_y=True,
)
# Plot hybrid pt
plot_attribute_compare(
    other=Input(df=original_df, attribute="jet_pt_hybrid", name="Cross check"),
    mine=Input(df=mine_df, attribute="jet_pt_hybrid", name="Mine"),
    output_name="compare_hybrid",
    log_y=True,
)
# Plot det level pt
plot_attribute_compare(
    other=Input(df=original_df, attribute="jet_pt_det_level", name="Cross check"),
    mine=Input(df=mine_df, attribute="jet_pt_det_level", name="Mine"),
    output_name="compare_det",
    log_y=True,
)
# rg.
# Shared pt cuts
# Selection for Cross check
# mask = original_df["zg"] < 0.2
# original_df_copy = original_df.copy()
# original_df_copy.loc[mask, "rg"] = -0.005
# mask_match = original_df["zgMatch"] < 0.2
# original_df_copy.loc[mask_match, "rgMatch"] = -0.005
# mask_det_level = original_df["zgDet"] < 0.2
# original_df_copy.loc[mask_det_level, "rgDet"] = -0.005
original_df_copy = original_df
# True
plot_attribute_compare(
    other=Input(df=original_df_copy, attribute="leading_kt_z_cut_02_true_delta_R", name="Cross check",),
    mine=Input(df=mine_df, attribute="leading_kt_z_cut_02_true_delta_R", name="Mine"),
    output_name="compare_true_rg",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# Hybrid
plot_attribute_compare(
    other=Input(df=original_df_copy, attribute="leading_kt_z_cut_02_hybrid_delta_R", name="Cross check",),
    mine=Input(df=mine_df, attribute="leading_kt_z_cut_02_hybrid_delta_R", name="Mine"),
    output_name="compare_hybrid_rg",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# Det Level
plot_attribute_compare(
    other=Input(df=original_df_copy, attribute="leading_kt_z_cut_02_det_level_delta_R", name="Cross check",),
    mine=Input(df=mine_df, attribute="leading_kt_z_cut_02_det_level_delta_R", name="Mine"),
    output_name="compare_det_rg",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# zg
# True
plot_attribute_compare(
    other=Input(df=original_df_copy, attribute="leading_kt_z_cut_02_true_z", name="Cross check",),
    mine=Input(df=mine_df, attribute="leading_kt_z_cut_02_true_z", name="Mine"),
    output_name="compare_true_z",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# Hybrid
plot_attribute_compare(
    other=Input(df=original_df_copy, attribute="leading_kt_z_cut_02_hybrid_z", name="Cross check",),
    mine=Input(df=mine_df, attribute="leading_kt_z_cut_02_hybrid_z", name="Mine"),
    output_name="compare_hybrid_z",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# Det Level
plot_attribute_compare(
    other=Input(df=original_df_copy, attribute="leading_kt_z_cut_02_det_level_z", name="Cross check",),
    mine=Input(df=mine_df, attribute="leading_kt_z_cut_02_det_level_z", name="Mine"),
    output_name="compare_det_z",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# kt
# True
plot_attribute_compare(
    other=Input(df=original_df_copy, attribute="leading_kt_z_cut_02_true_kt", name="Cross check",),
    mine=Input(df=mine_df, attribute="leading_kt_z_cut_02_true_kt", name="Mine"),
    output_name="compare_true_kt",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# Hybrid
plot_attribute_compare(
    other=Input(df=original_df_copy, attribute="leading_kt_z_cut_02_hybrid_kt", name="Cross check",),
    mine=Input(df=mine_df, attribute="leading_kt_z_cut_02_hybrid_kt", name="Mine"),
    output_name="compare_hybrid_kt",
    axis=bh.axis.Regular(21, -0.02, 0.4),
)
# Det Level
plot_attribute_compare(
    other=Input(df=original_df_copy, attribute="leading_kt_z_cut_02_det_level_kt", name="Cross check",),
    mine=Input(df=mine_df, attribute="leading_kt_z_cut_02_det_level_kt", name="Mine"),
    output_name="compare_det_kt",
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


# np.testing.assert_allclose(mine_df["soft_drop_z_cut_02_hybrid_delta_R"], mine_df["soft_drop_z_cut_04_hybrid_delta_R"])
# print("all close")
