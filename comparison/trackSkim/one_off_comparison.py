"""Comparison between standard analysis and track skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBNL/UCB
"""

import logging
import pprint
from pathlib import Path
from typing import Sequence

import attr
import awkward as ak
import boost_histogram as bh
import mammoth.helpers
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import uproot
from pachyderm import binned_data

from jet_substructure.analysis import plot_base as pb


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
    plot_config: pb.PlotConfig,
    output_dir: Path,
    axis: bh.axis.Regular = bh.axis.Regular(30, 0, 150),
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

    # Apply the PlotConfig
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])

    # filename = f"{plot_config.name}_{jet_pt_bin}{grooming_methods_filename_label}_{identifiers}_iterative_splittings"
    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def compare(collision_system: str, prefixes: Sequence[str], standard_filename: Path, track_skim_filename: Path) -> None:
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

    # For whatever reason, the sorting of the jets is inconsistent for embedding compared to all other datasets.
    # So we just apply a mask here to swap the one event where we have two jets.
    # NOTE: AliPhysics is actually the one that gets the sorting wrong here...
    # NOTE: This is a super specialized thing, but better to do it here instead of messing around with
    #       the actual mammoth analysis code.
    if collision_system == "embedPythia":
        # NOTE: I derived this mask by hand. It swaps index -2 and -3 (== swapping index 15 and 16)
        #       It can be double checked by looking at the jet pt. The precision makes
        #       it quite obvious which should go with which.
        reorder_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17]
        track_skim = track_skim[reorder_mask]
        # NOTE: ***********************************************************************************
        #       The canonical file actually did go through the steps of making the order in mammoth
        #       match AliPhysics by turning off sorting. But since we're more likely to be
        #       testing in the future with new files to do validation, it's better that we apply
        #       this remapping here.
        #       ***********************************************************************************

    output_dir = Path("comparison") / "trackSkim" / collision_system
    output_dir.mkdir(parents=True, exist_ok=True)

    for prefix in prefixes:
        logger.info(f"Comparing prefix '{prefix}'")

        text = f"{collision_system.replace('_', ' ')}: {prefix.replace('_', ' ')}"
        plot_attribute_compare(
            other=Input(arrays=standard, attribute=f"{prefix}_jet_pt", name="Standard"),
            mine=Input(arrays=track_skim, attribute=f"{prefix}_jet_pt", name="Track skim"),
            plot_config=pb.PlotConfig(
                name=f"{prefix}_jet_pt",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label="Prob.",
                                log=True,
                                font_size=22,
                            ),
                        ],
                        text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                        legend=pb.LegendConfig(location="center right", anchor=(0.985, 0.52), font_size=22),
                    ),
                    # Data ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "x",
                                label=r"$p_{\text{T,ch jet}}$ (GeV/$c$)",
                                font_size=22,
                            ),
                            pb.AxisConfig(
                                "y",
                                label=r"Track skim/Standard",
                                range=(0.6, 1.4),
                                font_size=22,
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
            ),
            output_dir=output_dir,
            axis=bh.axis.Regular(50, 0, 100),
            normalize=True,
        )
        standard_jet_pt = standard[f"{prefix}_jet_pt"]
        track_skim_jet_pt = track_skim[f"{prefix}_jet_pt"]

        # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessasry
        #logger.info(f"standard_jet_pt: {standard_jet_pt.to_list()}")
        #logger.info(f"track_skim_jet_pt: {track_skim_jet_pt.to_list()}")

        try:
            all_close_jet_pt = np.allclose(ak.to_numpy(standard_jet_pt), ak.to_numpy(track_skim_jet_pt))

            logger.info(f"jet_pt all close? {all_close_jet_pt}")
            #import IPython; IPython.embed()
            if not all_close_jet_pt:
                logger.info("jet pt")
                _arr = ak.zip({"s": standard_jet_pt, "t": track_skim_jet_pt})
                logger.info(pprint.pformat(_arr.to_list()))
                is_not_close_jet_pt = np.where(~np.isclose(ak.to_numpy(standard_jet_pt), ak.to_numpy(track_skim_jet_pt)))
                logger.info(f"Indicies where not close: {is_not_close_jet_pt}")
        except ValueError as e:
            logger.exception(e)

        for grooming_method in ["dynamical_kt", "soft_drop_z_cut_02"]:
            logger.info(f"Plotting method \"{grooming_method}\"")
            plot_attribute_compare(
                other=Input(arrays=standard, attribute=f"{grooming_method}_{prefix}_kt", name="Standard"),
                mine=Input(arrays=track_skim, attribute=f"{grooming_method}_{prefix}_kt", name="Track skim"),
                plot_config=pb.PlotConfig(
                    name=f"{grooming_method}_{prefix}_kt",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label="Prob.",
                                    log=True,
                                    font_size=22,
                                ),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                            legend=pb.LegendConfig(location="center right", anchor=(0.985, 0.52), font_size=22),
                        ),
                        # Data ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$k_{\text{T,g}}$ (GeV/$c$)",
                                    font_size=22,
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label=r"Track skim/Standard",
                                    range=(0.6, 1.4),
                                    font_size=22,
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
                ),
                normalize=True,
                axis=bh.axis.Regular(50, 0, 10),
                output_dir=output_dir,
            )

            standard_kt = standard[f"{grooming_method}_{prefix}_kt"]
            track_skim_kt = track_skim[f"{grooming_method}_{prefix}_kt"]

            # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessasry
            #logger.info(f"standard_kt: {standard_kt.to_list()}")
            #logger.info(f"track_skim_kt: {track_skim_kt.to_list()}")

            try:
                all_close_kt = np.allclose(ak.to_numpy(standard_kt), ak.to_numpy(track_skim_kt), rtol=1e-4)
                logger.info(f"kt all close? {all_close_kt}")
                if not all_close_kt:
                    logger.info("kt")
                    _arr = ak.zip({"s": standard_kt, "t": track_skim_kt})
                    logger.info(pprint.pformat(_arr.to_list()))
                    is_not_close_kt = np.where(~np.isclose(ak.to_numpy(standard_kt), ak.to_numpy(track_skim_kt)))
                    logger.info(f"Indicies where not close: {is_not_close_kt}")
            except ValueError as e:
                logger.exception(e)

            plot_attribute_compare(
                other=Input(arrays=standard, attribute=f"{grooming_method}_{prefix}_delta_R", name="Standard"),
                mine=Input(arrays=track_skim, attribute=f"{grooming_method}_{prefix}_delta_R", name="Track skim"),
                plot_config=pb.PlotConfig(
                    name=f"{grooming_method}_{prefix}_delta_R",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label="Prob.",
                                    log=True,
                                    font_size=22,
                                ),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper left", font_size=22),
                        ),
                        # Data ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$R_{\text{g}}$",
                                    font_size=22,
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label=r"Track skim/Standard",
                                    range=(0.6, 1.4),
                                    font_size=22,
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
                ),
                output_dir=output_dir,
                axis=bh.axis.Regular(50, 0, 0.6),
                normalize=True,
            )
            standard_rg = standard[f"{grooming_method}_{prefix}_delta_R"]
            track_skim_rg = track_skim[f"{grooming_method}_{prefix}_delta_R"]

            # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessasry
            #logger.info(f"standard_zg: {standard_zg.to_list()}")
            #logger.info(f"track_skim_zg: {track_skim_zg.to_list()}")

            try:
                all_close_rg = np.allclose(ak.to_numpy(standard_rg), ak.to_numpy(track_skim_rg), rtol=1e-4)
                logger.info(f"Rg all close? {all_close_rg}")
                if not all_close_rg:
                    logger.info("delta_R")
                    _arr = ak.zip({"s": standard_rg, "t": track_skim_rg})
                    logger.info(pprint.pformat(_arr.to_list()))
                    is_not_close_rg = np.where(~np.isclose(ak.to_numpy(standard_rg), ak.to_numpy(track_skim_rg)))
                    logger.info(f"Indicies where not close: {is_not_close_rg}")
            except ValueError as e:
                logger.exception(e)

            #import IPython; IPython.embed()

            #logger.info(f"standard_rg: {standard_rg.to_list()}")
            #logger.info(f"track_skim_rg: {track_skim_rg.to_list()}")

            plot_attribute_compare(
                other=Input(arrays=standard, attribute=f"{grooming_method}_{prefix}_z", name="Standard"),
                mine=Input(arrays=track_skim, attribute=f"{grooming_method}_{prefix}_z", name="Track skim"),
                plot_config=pb.PlotConfig(
                    name=f"{grooming_method}_{prefix}_z",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label="Prob.",
                                    log=True,
                                    font_size=22,
                                ),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper left", font_size=22),
                        ),
                        # Data ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$z_{\text{g}}$",
                                    font_size=22,
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label=r"Track skim/Standard",
                                    range=(0.6, 1.4),
                                    font_size=22,
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
                ),
                normalize=True,
                axis=bh.axis.Regular(50, 0, 0.5),
                output_dir=output_dir,
            )

            standard_zg = standard[f"{grooming_method}_{prefix}_z"]
            track_skim_zg = track_skim[f"{grooming_method}_{prefix}_z"]

            # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessasry
            #logger.info(f"standard_zg: {standard_zg.to_list()}")
            #logger.info(f"track_skim_zg: {track_skim_zg.to_list()}")

            try:
                all_close_zg = np.allclose(ak.to_numpy(standard_zg), ak.to_numpy(track_skim_zg))
                logger.info(f"zg all close? {all_close_zg}")
                if not all_close_zg:
                    logger.info("z")
                    _arr = ak.zip({"s": standard_zg, "t": track_skim_zg})
                    logger.info(pprint.pformat(_arr.to_list()))
            except ValueError as e:
                logger.exception(e)


def run(collision_system: str, prefixes: Sequence[str] = None) -> None:
    if prefixes is None:
        prefixes = ["data"]
    mammoth.helpers.setup_logging()
    logger.info(f"Running {collision_system} with prefixes {prefixes}")
    path_to_mammoth = Path(mammoth.helpers.__file__).parent.parent
    standard_base_filename = "AnalysisResults"
    if collision_system == "pythia":
        standard_base_filename += ".12"
    compare(
        collision_system=collision_system,
        prefixes=prefixes,
        standard_filename=path_to_mammoth / f"projects/framework/{collision_system}/1/skim/{standard_base_filename}.repaired.00_iterative_splittings.root",
        track_skim_filename=path_to_mammoth / f"projects/framework/{collision_system}/1/skim/skim_output.root",
    )


if __name__ == "__main__":
    collision_system = "embedPythia"

    _prefixes = {
        "pp": ["data"],
        "pythia": ["data", "true"],
        "PbPb": ["data"],
        "embedPythia": ["hybrid", "det_level", "true"],
    }
    run(collision_system=collision_system, prefixes=_prefixes[collision_system])
