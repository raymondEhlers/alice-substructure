#!/usr/bin/env python3

""" Attempt analysis just using data frames.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
import gzip
import itertools
import logging
import pickle
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import boost_histogram as bh
import dill
import enlighten
import IPython
import numpy as np
import pandas as pd
import uproot

from jet_substructure.base import analysis_objects, data_manager, helpers, skim_analysis_objects


logger = logging.getLogger(__name__)


def merge_hists(a: Dict[str, bh.Histogram], b: Dict[str, bh.Histogram]) -> Dict[str, bh.Histogram]:
    """ Merge hists stored in a file.

    """
    for k in b:
        a[k] += b[k]
    return a


def dask_df_from_file() -> None:
    df = uproot.tree.daskframe(
        path=[
            "temp_cache/embedPythia/55*/skim/*_iterative_splittings.root",
            "trains/embedPythia/55*/skim/*_iterative_splittings.root",
        ],
        treepath="tree",
        namedecode="utf-8",
        branches=["scale_factor", "*det_level*", "*hybrid*"],
    )

    IPython.start_ipython(user_ns=locals())


def dask_df_from_delayed() -> None:
    # From: https://stackoverflow.com/q/60189433/12907985
    import dask.dataframe as dd
    from dask import delayed

    @delayed  # type: ignore
    def get_df(file: Path, treepath: str, branches: Sequence[str]) -> pd.DataFrame:
        tree = uproot.open(file)[treepath]
        return tree.pandas.df(branches=branches)

    path_list = data_manager._ensure_and_expand_paths(
        [
            Path("temp_cache/embedPythia/55*/skim/*_iterative_splittings.root"),
            Path("trains/embedPythia/55*/skim/*_iterative_splittings.root"),
        ]
    )

    dfs = [get_df(path, "tree", branches=["scale_factor", "*det_level*", "*hybrid*"]) for path in path_list]
    daskframe = dd.from_delayed(dfs)

    IPython.start_ipython(user_ns=locals())


# def df_from_file(filenames: Sequence[Path], branches: Sequence[str]):
def df_from_file_embedding(path_list: Sequence[Path], path_list_friends: Sequence[Path], output_dir: Optional[Path] = None) -> None:  # noqa: 901
    # It's dumb to reimport, but we need to do  it here for it to be available immediately in IPython.
    from pathlib import Path  # noqa: F401
    # Validation
    if output_dir is None:
        output_dir = Path(f"output/embedPythia/skim")

    data_frames = uproot.pandas.iterate(
        path=path_list,
        treepath="tree",
        namedecode="utf-8",
        # Apparently I forgot to rename the prefixes for eta, data, so I account for that here and when I access the values.
        branches=["scale_factor", "*true*", "*det_level*", "*hybrid*", "jet_eta_data", "jet_phi_data", "jet_eta_detLevel", "jet_phi_detLevel"],
        reportpath=True,
        # Otherwise, we can't really count how many steps it's going to take...
        #entrysteps=float("inf"),
    )
    # NOTE: One needs to be careful here if iterating over many files. The friends may not match up in the entries!!
    #data_frames_friends = uproot.pandas.iterate(
    #    path=path_list_friends,
    #    treepath="tree",
    #    namedecode="utf-8",
    #    branches=["*matched*", "*detLevel*", "*data*"],
    #    reportpath=True,
    #)

    # NOPE! Still too big...
    # df = pd.concat(data_frames, axis=1, copy=False)

    # TODO: Define grooming methods better?
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

    # Define hists.
    response_types = [
        skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="det_level"),
        skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="true"),
        skim_analysis_objects.ResponseType(measured_like="det_level", generator_like="true"),
    ]
    _matching_name_to_axis_value: Dict[str, int] = {
        "all": 0,
        "pure": 1,
        "leading_untagged_subleading_correct": 2,
        "leading_correct_subleading_untagged": 3,
        "leading_untagged_subleading_mistag": 4,
        "leading_mistag_subleading_untagged": 5,
        "swap": 6,
        "both_untagged": 7,
    }
    hists = {}
    for grooming_method in grooming_methods:
        for matching_type in _matching_name_to_axis_value:
            #
            # Residuals by matching type
            #
            # jet_pt residual
            # Axes: hybrid_level_jet_pt, det_level_jet_pt, residual
            hists[f"{grooming_method}_hybrid_det_level_jet_pt_residuals_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(80, -2, 2),
                storage=bh.storage.Weight(),
            )
            # kt residual
            # Axes: hybrid_level_jet_kt, det_level_jet_kt, residual
            hists[f"{grooming_method}_hybrid_det_level_kt_residuals_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(25, 0, 25),
                bh.axis.Regular(25, 0, 25),
                bh.axis.Regular(80, -2, 2),
                storage=bh.storage.Weight(),
            )
            #
            # Responses
            #
            # Example axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt
            # Generally: measured_pt, measured_kt, generator_pt, generator_kt
            for response_type in response_types:
                # kt response
                hists[
                    f"{grooming_method}_{str(response_type)}_kt_response_matching_type_{matching_type}"
                ] = bh.Histogram(
                    bh.axis.Regular(28, 0, 140),
                    bh.axis.Regular(26, -1, 25),
                    bh.axis.Regular(28, 0, 140),
                    bh.axis.Regular(26, -1, 25),
                    storage=bh.storage.Weight(),
                )
                # Delta R response
                # Axes: measured_pt, measured_R, generator_pt, generator_R
                hists[
                    f"{grooming_method}_{str(response_type)}_delta_R_response_matching_type_{matching_type}"
                ] = bh.Histogram(
                    bh.axis.Regular(28, 0, 140),
                    bh.axis.Regular(21, -0.02, 0.4),
                    bh.axis.Regular(28, 0, 140),
                    bh.axis.Regular(21, -0.02, 0.4),
                    storage=bh.storage.Weight(),
                )

            # Where we restrict the matching distance.
            # kt response
            hists[
                f"{grooming_method}_hybrid_det_level_kt_response_matching_type_leading_correct_subleading_untagged_distance_less_than_005"
            ] = bh.Histogram(
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(26, -1, 25),
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(26, -1, 25),
                storage=bh.storage.Weight(),
            )
            # Delta R response
            # Axes: measured_pt, measured_R, generator_pt, generator_R
            hists[
                f"{grooming_method}_hybrid_det_level_delta_R_response_matching_type_leading_correct_subleading_untagged_distance_less_than_005"
            ] = bh.Histogram(
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(21, -0.02, 0.4),
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(21, -0.02, 0.4),
                storage=bh.storage.Weight(),
            )

        #
        # Residuals
        #
        # Jet pt residual mean: JES
        # Error is the width: JER
        # We normalize the width by the true jet pt afterwards, so we have to collect it separately.
        # NOTE: Ideally, we'd extract both values from one profile histogram which is binned in true and
        #       hybrid jet pt, but projecting profiles doesn't seem to work quite as expected with bh, and
        #       it's not worth investigating further at the moment.
        for hybrid_jet_pt_bin in [helpers.RangeSelector(40, 120), helpers.RangeSelector(20, 200)]:
            hists[
                f"{grooming_method}_hybrid_det_level_jet_pt_residual_mean_hybrid_{str(hybrid_jet_pt_bin)}"
            ] = bh.Histogram(bh.axis.Regular(25, 0, 250), storage=bh.storage.WeightedMean(),)
            hists[f"{grooming_method}_hybrid_true_jet_pt_residual_mean_hybrid_{str(hybrid_jet_pt_bin)}"] = bh.Histogram(
                bh.axis.Regular(25, 0, 250), storage=bh.storage.WeightedMean(),
            )
            hists[f"{grooming_method}_det_true_jet_pt_residual_mean_hybrid_{str(hybrid_jet_pt_bin)}"] = bh.Histogram(
                bh.axis.Regular(25, 0, 250), storage=bh.storage.WeightedMean(),
            )
        # Residual so we can see the entire distribution.
        # We intentionally don't select the hybrid jet pt range here.
        hists[f"{grooming_method}_hybrid_det_level_jet_pt_residual_distribution"] = bh.Histogram(
            bh.axis.Regular(15, 0, 150), bh.axis.Regular(150, -1.5, 1.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_hybrid_true_jet_pt_residual_distribution"] = bh.Histogram(
            bh.axis.Regular(15, 0, 150), bh.axis.Regular(150, -1.5, 1.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_det_true_jet_pt_residual_distribution"] = bh.Histogram(
            bh.axis.Regular(15, 0, 150), bh.axis.Regular(150, -1.5, 1.5), storage=bh.storage.Weight(),
        )

        # Distance comparison by matching type
        for matching_type in _matching_name_to_axis_value:
            hists[f"{grooming_method}_hybrid_det_level_distance_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(20, 0, 0.5), storage=bh.storage.Weight(),
            )

    progress_manager = enlighten.Manager()
    # NOTE: Careful with this counter! It may not be correct if we are iterating over chunks in a file.
    with progress_manager.counter(total=len(path_list), desc="Analyzing", unit="tree", leave=True) as tree_counter:
        #for (df_path, df), (df_friend_path, df_friend) in tree_counter(zip(data_frames, data_frames_friends)):
        for df_path, df in tree_counter(data_frames):
            logger.debug(f"Processing df from {df_path}")
            # Merge the friends together.
            # Rename friends columns because I forgot to rename earlier.
            #df_friend = df_friend.rename(
            #    columns=lambda s: s.replace("matched", "true")
            #    .replace("data", "hybrid")
            #    .replace("detLevel", "det_level")
            #)
            #df = pd.concat([df, df_friend], axis=1)

            # Setup
            hybrid_jet_pt_mask = (df["jet_pt_hybrid"] > 40) & (df["jet_pt_hybrid"] < 120)
            # And finally process
            for grooming_method in grooming_methods:

                matching_leading = df[f"{grooming_method}_hybrid_det_level_matching_leading"]
                matching_subleading = df[f"{grooming_method}_hybrid_det_level_matching_subleading"]

                matching_selections = analysis_objects.MatchingSelections(
                    leading=analysis_objects.MatchingResult(
                        properly=(matching_leading == 1),
                        mistag=(matching_leading == 2),
                        failed=(matching_leading == 3),
                    ),
                    subleading=analysis_objects.MatchingResult(
                        properly=(matching_subleading == 1),
                        mistag=(matching_subleading == 2),
                        failed=(matching_subleading == 3),
                    ),
                )

                # Residuals (without caring about matching types, so we just take all)
                mask = matching_selections["all"]
                for temp_hybrid_jet_pt_bin in [helpers.RangeSelector(40, 120), helpers.RangeSelector(20, 200)]:
                    # NOTE: Add temp_ onto the names to avoid interfering with the general hybrid jet pt mask
                    #       which is used all over the place.
                    temp_hybrid_jet_pt_mask = temp_hybrid_jet_pt_bin.mask_array(df["jet_pt_hybrid"])
                    masked_df = df[mask & temp_hybrid_jet_pt_mask]
                    # Residual mean and width
                    # We store the unnnormalized residual so we can use this profile hist to extract both the
                    # mean and the width. Note that it's slightly less accurate because then normalize by the bin center
                    # rather than the exact true jet pt per fill, but it's close enough for our purposes.
                    hists[
                        f"{grooming_method}_hybrid_det_level_jet_pt_residual_mean_hybrid_{str(temp_hybrid_jet_pt_bin)}"
                    ].fill(
                        masked_df["jet_pt_true"].to_numpy(),
                        sample=(masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                    hists[
                        f"{grooming_method}_hybrid_true_jet_pt_residual_mean_hybrid_{str(temp_hybrid_jet_pt_bin)}"
                    ].fill(
                        masked_df["jet_pt_true"].to_numpy(),
                        sample=(masked_df["jet_pt_hybrid"] - masked_df["jet_pt_true"]).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                    hists[f"{grooming_method}_det_true_jet_pt_residual_mean_hybrid_{str(temp_hybrid_jet_pt_bin)}"].fill(
                        masked_df["jet_pt_true"].to_numpy(),
                        sample=(masked_df["jet_pt_det_level"] - masked_df["jet_pt_true"]).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                # Full residual as a function of true pt. We can select the true jet pt range when plotting.
                # NOTE: We intentionally didn't apply a hybrid jet pt cut. And our true jet pt selection will
                #       be applied when plotting.
                masked_df = df[mask]
                hists[f"{grooming_method}_hybrid_det_level_jet_pt_residual_distribution"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    (
                        (masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]) / masked_df["jet_pt_det_level"]
                    ).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_hybrid_true_jet_pt_residual_distribution"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    ((masked_df["jet_pt_hybrid"] - masked_df["jet_pt_true"]) / masked_df["jet_pt_true"]).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_det_true_jet_pt_residual_distribution"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    ((masked_df["jet_pt_det_level"] - masked_df["jet_pt_true"]) / masked_df["jet_pt_true"]).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                # Matching distance
                masked_df = df[hybrid_jet_pt_mask]
                # We convert from pd.Series to ndarray because the pd.Series conversions seem a bit odd at times.
                distances = np.sqrt(
                    (masked_df["jet_eta_data"] - masked_df["jet_eta_detLevel"]) ** 2
                    + (masked_df["jet_phi_data"] - masked_df["jet_phi_detLevel"]) ** 2
                ).to_numpy()
                for matching_type in _matching_name_to_axis_value:
                    mask = matching_selections[matching_type]
                    masked_df = df[mask & hybrid_jet_pt_mask]
                    # Axes: hybrid_level_jet_pt, det_level_jet_pt, residual
                    hists[f"{grooming_method}_hybrid_det_level_jet_pt_residuals_matching_type_{matching_type}"].fill(
                        masked_df["jet_pt_hybrid"].to_numpy(),
                        masked_df["jet_pt_det_level"].to_numpy(),
                        (
                            (masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]) / masked_df["jet_pt_det_level"]
                        ).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                    # Axes: hybrid_level_jet_kt, det_level_jet_kt, residual
                    hists[f"{grooming_method}_hybrid_det_level_kt_residuals_matching_type_{matching_type}"].fill(
                        masked_df[f"{grooming_method}_hybrid_kt"].to_numpy(),
                        masked_df[f"{grooming_method}_det_level_kt"].to_numpy(),
                        (
                            (masked_df[f"{grooming_method}_hybrid_kt"] - masked_df[f"{grooming_method}_det_level_kt"])
                            / masked_df[f"{grooming_method}_det_level_kt"]
                        ).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )

                    # Matching distances
                    # Distances were only calculated for hybrid jets, so we don't need to reapply the hybrid jet pt cut here.
                    # Just apply the matching mask to the distances.
                    hists[f"{grooming_method}_hybrid_det_level_distance_matching_type_{matching_type}"].fill(
                        distances[mask[hybrid_jet_pt_mask]], weight=masked_df["scale_factor"].to_numpy()
                    )

                    for response_type in response_types:
                        # Axes: measured_like_pt, measured_like_kt, generator_like_pt, generator_like_kt
                        hists[f"{grooming_method}_{str(response_type)}_kt_response_matching_type_{matching_type}"].fill(
                            masked_df[f"jet_pt_{response_type.measured_like}"].to_numpy(),
                            masked_df[f"{grooming_method}_{response_type.measured_like}_kt"].to_numpy(),
                            masked_df[f"jet_pt_{response_type.generator_like}"].to_numpy(),
                            masked_df[f"{grooming_method}_{response_type.generator_like}_kt"].to_numpy(),
                            weight=masked_df["scale_factor"].to_numpy(),
                        )
                        # Axes: measured_like_pt, measured_like_R, generator_like_pt, generator_like_R
                        hists[
                            f"{grooming_method}_{str(response_type)}_delta_R_response_matching_type_{matching_type}"
                        ].fill(
                            masked_df[f"jet_pt_{response_type.measured_like}"].to_numpy(),
                            masked_df[f"{grooming_method}_{response_type.measured_like}_delta_R"].to_numpy(),
                            masked_df[f"jet_pt_{response_type.generator_like}"].to_numpy(),
                            masked_df[f"{grooming_method}_{response_type.generator_like}_delta_R"].to_numpy(),
                            weight=masked_df["scale_factor"].to_numpy(),
                        )
                    # Response with matching distance for leading_correct_subleading_untagged
                    if matching_type == "leading_correct_subleading_untagged":
                        distance_mask = distances[mask[hybrid_jet_pt_mask]] < 0.05
                        # kt
                        hists[
                            f"{grooming_method}_hybrid_det_level_kt_response_matching_type_leading_correct_subleading_untagged_distance_less_than_005"
                        ].fill(
                            masked_df[distance_mask][f"jet_pt_{response_type.measured_like}"].to_numpy(),
                            masked_df[distance_mask][f"{grooming_method}_{response_type.measured_like}_kt"].to_numpy(),
                            masked_df[distance_mask][f"jet_pt_{response_type.generator_like}"].to_numpy(),
                            masked_df[distance_mask][f"{grooming_method}_{response_type.generator_like}_kt"].to_numpy(),
                            weight=masked_df[distance_mask]["scale_factor"].to_numpy(),
                        )
                        # Delta R
                        hists[
                            f"{grooming_method}_hybrid_det_level_delta_R_response_matching_type_leading_correct_subleading_untagged_distance_less_than_005"
                        ].fill(
                            masked_df[distance_mask][f"jet_pt_{response_type.measured_like}"].to_numpy(),
                            masked_df[distance_mask][
                                f"{grooming_method}_{response_type.measured_like}_delta_R"
                            ].to_numpy(),
                            masked_df[distance_mask][f"jet_pt_{response_type.generator_like}"].to_numpy(),
                            masked_df[distance_mask][
                                f"{grooming_method}_{response_type.generator_like}_delta_R"
                            ].to_numpy(),
                            weight=masked_df[distance_mask]["scale_factor"].to_numpy(),
                        )

    progress_manager.stop()

    # Write the hists
    output_dir = Path(f"output/embedPythia/skim")
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_filename = output_dir / "embedded.pgz"
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        dill.dump(hists, pkl_file)


def map_reduce_pandas_concat() -> None:
    data_frames = uproot.pandas.iterate(
        path=[
            "temp_cache/embedPythia/55*/skim/*_iterative_splittings.root",
            "trains/embedPythia/55*/skim/*_iterative_splittings.root",
        ],
        treepath="tree",
        namedecode="utf-8",
        branches=["scale_factor", "*det_level*", "*hybrid*"],
    )

    # NOPE! Still too big...
    logger.debug("Reducing")
    df = functools.reduce(lambda x, y: pd.concat([x, y], copy=False, ignore_index=True), data_frames)
    logger.debug("Finished")
    # NOPE! Still too big...
    # logger.debug("Starting concat")
    # df = pd.concat(data_frames, copy=False)

    IPython.start_ipython(user_ns=locals())


def _fill_grooming_hists(
    masked_df: pd.DataFrame,
    grooming_method: str,
    hists: Mapping[str, bh.Histogram],
    prefix: str,
    suffix: Optional[str] = None,
) -> None:
    """ Fill grooming hists using the df.

    This is in a separate function so the DataFrame can be masked.

    Args:
        masked_df: Masked DataFrame to be used for filling.
        grooming_method: Grooming method to be filled.
        hists: Hists to be filled.
        prefix: Prefix specifying the data type, such as "data", "matched", "true", etc.
        suffix: Suffix to additional identify the hists.
    Returns:
        None. The hists stored in the hists dict are filled.
    """
    if suffix is None:
        suffix = ""
    else:
        if not suffix.startswith("_"):
            suffix = f"_{suffix}"

    # Handle the case of the first split.
    # If "_first_split" isn't included in the grooming method, then nothing is replaced.
    grooming_method_for_df = grooming_method.replace("_first_split", "")

    hists[f"{grooming_method}_{prefix}_kt{suffix}"].fill(
        masked_df[f"jet_pt_{prefix}"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_kt"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    hists[f"{grooming_method}_{prefix}_delta_R{suffix}"].fill(
        masked_df[f"jet_pt_{prefix}"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_delta_R"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    hists[f"{grooming_method}_{prefix}_z{suffix}"].fill(
        masked_df[f"jet_pt_{prefix}"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_z"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    hists[f"{grooming_method}_{prefix}_n_to_split{suffix}"].fill(
        masked_df[f"jet_pt_{prefix}"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_n_to_split"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    hists[f"{grooming_method}_{prefix}_n_groomed_to_split{suffix}"].fill(
        masked_df[f"jet_pt_{prefix}"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_n_groomed_to_split"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    # Number of splittings which pass the grooming condition. For SoftDrop, this is n_sd.
    hists[f"{grooming_method}_{prefix}_n_passed_grooming{suffix}"].fill(
        masked_df[f"jet_pt_{prefix}"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_n_passed_grooming"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )


def df_from_file_data(collision_system: str, path_list: Sequence[Path], prefix: str, output_dir: Optional[Path] = None) -> None:  # noqa: 901
    # It's dumb to reimport, but we need to do  it here for it to be available immediately in IPython.
    from pathlib import Path  # noqa: F401
    # Validation
    if output_dir is None:
        output_dir = Path(f"output/{collision_system}/skim")

    # Setup
    jet_R = 0.4
    branches = [f"*{prefix}*"]
    if collision_system in ["pythia", "embedPythia"]:
        branches.append("scale_factor")
    data_frames = uproot.pandas.iterate(
        path=path_list, treepath="tree", namedecode="utf-8", branches=branches, reportpath=True,
    )

    # TODO: Define grooming methods better?
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
    direct_comparison_grooming_methods = [
        "leading_kt_z_cut_02_first_split",
        "leading_kt_z_cut_04_first_split",
    ]

    # Define hists.
    hists = {}
    for grooming_method in itertools.chain(grooming_methods, direct_comparison_grooming_methods):
        # Standard
        jet_pt_axis = bh.axis.Regular(28, 0, 140)
        hists[f"{grooming_method}_{prefix}_kt"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(26, -1, 25), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_delta_R"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(21, -0.02, jet_R), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_theta"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(21, -0.05, 1.0), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_z"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(21, -0.025, 0.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_to_split"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(10, -0.5, 9.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_groomed_to_split"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(10, -0.5, 9.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_passed_grooming"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(10, -0.5, 9.5), storage=bh.storage.Weight(),
        )
        # High kt
        hists[f"{grooming_method}_{prefix}_kt_high_kt"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(26, -1, 25), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_delta_R_high_kt"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(21, -0.02, jet_R), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_theta_high_kt"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(21, -0.05, 1.0), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_z_high_kt"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(21, -0.025, 0.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_to_split_high_kt"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(10, -0.5, 9.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_groomed_to_split_high_kt"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(10, -0.5, 9.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_passed_grooming_high_kt"] = bh.Histogram(
            jet_pt_axis, bh.axis.Regular(10, -0.5, 9.5), storage=bh.storage.Weight(),
        )

    progress_manager = enlighten.Manager()
    with progress_manager.counter(total=len(path_list), desc="Analyzing", unit="tree", leave=True) as tree_counter:
        for df_path, df in tree_counter(data_frames):
            logger.info(f"Processing df from {df_path}")
            # Setup
            # Add scale_factor weight as 1 if it's not included. This way, we can always weight from the
            # scale factor of the masked_df.
            if "scale_factor" not in df:
                df = df.assign(scale_factor=np.ones_like(df[f"jet_pt_{prefix}"]))
            # Jet pt bin
            jet_pt_bin = helpers.RangeSelector(min=40, max=120)
            jet_pt_mask = jet_pt_bin.mask_array(df[f"jet_pt_{prefix}"])

            # Standard grooming method plots
            masked_df = df[jet_pt_mask]
            for grooming_method in grooming_methods:
                _fill_grooming_hists(masked_df=masked_df, grooming_method=grooming_method, hists=hists, prefix=prefix)

            # Direct comparison plots
            mask = jet_pt_mask & (df[f"{grooming_method}_{prefix}_n_passed_grooming"] <= 1)
            masked_df = df[mask]
            for grooming_method in direct_comparison_grooming_methods:
                _fill_grooming_hists(masked_df=masked_df, grooming_method=grooming_method, hists=hists, prefix=prefix)

            # High kt grooming plots.
            for grooming_method in grooming_methods:
                mask = jet_pt_mask & (df[f"{grooming_method}_{prefix}_kt"] > 10)
                _fill_grooming_hists(
                    masked_df=df[mask], grooming_method=grooming_method, hists=hists, prefix=prefix, suffix="_high_kt"
                )

    progress_manager.stop()

    # Write the hists
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_filename = output_dir / f"{collision_system}.pgz"
    logger.info(f"Saving hists to {pkl_filename}")
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        pickle.dump(hists, pkl_file)  # type: ignore


def output_dir_f(output_dir: Path, identifier: str) -> Path:
    """ Format an output_dir path with a given identifier.

    Also ensures that the directory exists.

    Args:
        output_dir: Output dir containing a format identifier, `{identifier}`.
        identifier: Identifier to include in the path. Usually, it's the collision system,
            but it doesn't have to be.
    Returns:
        Output path formatted with the identifier.
    """
    p = Path(str(output_dir).format(identifier=identifier))
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_all() -> None:
    # TODO: Consolidate
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
    direct_comparison_grooming_methods = [
        "leading_kt_z_cut_02_first_split",
        "leading_kt_z_cut_04_first_split",
    ]
    # NOTE: Order is changed here to match from before!!
    _matching_name_to_axis_value: Dict[str, int] = {
        "all": 0,
        "pure": 1,
        "leading_untagged_subleading_correct": 2,
        "swap": 6,
        "leading_untagged_subleading_mistag": 4,
        "leading_correct_subleading_untagged": 3,
        "leading_mistag_subleading_untagged": 5,
        "both_untagged": 7,
    }

    # NOTE: Intentionally skipping the f-string here. We want to format it later!
    base_dir = Path("output/{identifier}/skim")

    logger.info("Loading embedded response data")
    pkl_filename = Path("output") / "embedPythia" / "skim" / "embedded.pgz"
    with gzip.GzipFile(pkl_filename, "r") as pkl_file:
        response_hists = pickle.load(pkl_file)  # type: ignore

    logger.info("Loading embedPythia data")
    pkl_filename = Path("output") / "embedPythia" / "skim" / "embedPythia.pgz"
    with gzip.GzipFile(pkl_filename, "r") as pkl_file:
        embed_pythia_hists = pickle.load(pkl_file)  # type: ignore

    logger.info("Loading PbPb data")
    pkl_filename = Path("output") / "PbPb" / "skim" / "PbPb.pgz"
    with gzip.GzipFile(pkl_filename, "r") as pkl_file:
        PbPb_hists = pickle.load(pkl_file)  # type: ignore

    # Add some helpful imports and definitions
    from importlib import reload  # noqa: F401

    try:
        # May not want to import if developing.
        from jet_substructure.analysis import plot_from_skim  # noqa: F401
    except SyntaxError:
        logger.info("Couldn't load plot_from_skim due to syntax error. You need to load it.")

    user_ns = locals()
    user_ns.update({"output_dir_f": output_dir_f})
    IPython.start_ipython(user_ns=user_ns)

    # Plotting
    # plot_from_skim.plot_residuals_by_matching_type(
    #     hists=hists, grooming_methods=grooming_methods, matching_types=list(_matching_name_to_axis_value.keys()), output_dir=output_dir
    # )
    # plot_from_skim.plot_residuals(hists=hists, grooming_methods=grooming_methods, output_dir=output_dir)
    # plot_from_skim.plot_response_by_matching_type(
    #     hists=hists, grooming_methods=grooming_methods, matching_types=list(_matching_name_to_axis_value.keys()), output_dir=output_dir,
    # )
    # plot_from_skim.plot_compare_kt(hists=hists, data_hists=data_hists[0], grooming_methods=grooming_methods, output_dir=output_dir)


def run_embed_pythia(run_response: bool = True) -> None:
    collision_system = "embedPythia"
    path_list = data_manager._ensure_and_expand_paths(
        [
            Path("trains/embedPythia/588*/skim/*_iterative_splittings.root"),
            Path("trains/embedPythia/589*/skim/*_iterative_splittings.root"),
            Path("trains/embedPythia/590*/skim/*_iterative_splittings.root"),
        ]
    )
    path_list_friends = data_manager._ensure_and_expand_paths(
        [
            #Path("temp_cache/embedPythia/55*/skim/*_iterative_splittings_friend.root"),
            #Path("trains/embedPythia/55*/skim/*_iterative_splittings_friend.root"),
        ]
    )
    if run_response:
        for train_number in range(5903, 5904):
            logger.info(f"Processing train number {train_number}")
            path_list = data_manager._ensure_and_expand_paths(
                [
                    #Path("trains/embedPythia/5903/skim/merged/*.root")
                    Path("trains/embedPythia/5903/skim/merged/AnalysisResults.merged.01.root")
                ]
            )
            print(path_list)
            df_from_file_embedding(path_list=path_list, path_list_friends=path_list_friends, output_dir=Path(f"output/{collision_system}/skim/{train_number}"))

        # Marge and write the data hists
        embedding_hists = functools.reduce(merge_hists, [dill.load(gzip.GzipFile(f"output/{collision_system}/skim/{train_number}/embedded.pgz", "r")) for train_number in range(5884, 5904)])
        pkl_filename = Path(f"output/{collision_system}/skim/embedded.pgz")
        logger.info(f"Saving hists to {pkl_filename}")
        with gzip.GzipFile(pkl_filename, "w") as pkl_file:
            dill.dump(embedding_hists, pkl_file)

    for train_number in range(5904, 5904):
        logger.info(f"Processing train number {train_number}")
        path_list = data_manager._ensure_and_expand_paths(
            [
                Path(f"trains/embedPythia/{train_number}/skim/*_iterative_splittings.root"),
            ]
        )
        df_from_file_data(
            collision_system=collision_system, path_list=path_list, prefix="hybrid", output_dir=Path(f"output/{collision_system}/skim/{train_number}")
        )

    # Marge and write the data hists
    hists = functools.reduce(merge_hists, [pickle.load(gzip.GzipFile(f"output/{collision_system}/skim/{train_number}/{collision_system}.pgz", "r")) for train_number in range(5884, 5904)])  # type: ignore
    pkl_filename = Path(f"output/{collision_system}/skim/{collision_system}.pgz")
    logger.info(f"Saving hists to {pkl_filename}")
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        pickle.dump(hists, pkl_file)  # type: ignore


if __name__ == "__main__":
    helpers.setup_logging()
    plot_only = False
    if not plot_only:
        df_from_file_data(
            collision_system="PbPb",
            path_list=data_manager._ensure_and_expand_paths(
                [Path("trains/PbPb/5537/skim/*_iterative_splittings.root")]
            ),
            prefix="data",
        )
        run_embed_pythia(run_response=True)
        # df_from_file_data(collision_system="pythia")
        # dask_df_from_file()
        # dask_df_from_delayed()
        # map_reduce_pandas_concat()

    plot_all()
