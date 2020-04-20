#!/usr/bin/env python3

""" Attempt analysis just using data frames.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
import logging
from pathlib import Path
from typing import Dict, Sequence

import boost_histogram as bh
import enlighten
import IPython
import pandas as pd
import uproot

from jet_substructure.base import analysis_objects, data_manager, helpers


logger = logging.getLogger(__name__)


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
def df_from_file() -> None:
    path_list = data_manager._ensure_and_expand_paths(
        [
            Path("temp_cache/embedPythia/55*/skim/*_iterative_splittings.root"),
            Path("trains/embedPythia/55*/skim/*_iterative_splittings.root"),
        ]
    )
    data_frames = uproot.pandas.iterate(
        path=path_list,
        treepath="tree",
        namedecode="utf-8",
        branches=["scale_factor", "*true*", "*det_level*", "*hybrid*"],
        reportpath=True,
    )
    # for df in data_frames:
    #    IPython.embed()

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
            # Axes: hybrid_level_jet_pt, det_level_jet_pt, residual
            hists[f"{grooming_method}_hybrid_det_jet_pt_residuals_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(80, -2, 2),
                storage=bh.storage.Weight(),
            )
            # Axes: hybrid_level_jet_kt, det_level_jet_kt, residual
            hists[f"{grooming_method}_hybrid_det_kt_residuals_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(25, 0, 25),
                bh.axis.Regular(25, 0, 25),
                bh.axis.Regular(80, -2, 2),
                storage=bh.storage.Weight(),
            )
            # Axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt, residual
            hists[f"{grooming_method}_hybrid_det_kt_response_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(25, 0, 25),
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(25, 0, 25),
                storage=bh.storage.Weight(),
            )
            # Axes: hybrid_pt, hybrid_R, det_level_pt, det_level_R, residual
            hists[f"{grooming_method}_hybrid_det_delta_R_response_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(20, 0, 0.4),
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(20, 0, 0.4),
                storage=bh.storage.Weight(),
            )

        # Residual mean
        hists[f"{grooming_method}_hybrid_det_jet_pt_residual_mean"] = bh.Histogram(
            bh.axis.Regular(25, 0, 250), bh.axis.Regular(15, 0, 150), storage=bh.storage.WeightedMean(),
        )
        hists[f"{grooming_method}_hybrid_true_jet_pt_residual_mean"] = bh.Histogram(
            bh.axis.Regular(25, 0, 250), bh.axis.Regular(15, 0, 150), storage=bh.storage.WeightedMean(),
        )
        hists[f"{grooming_method}_det_true_jet_pt_residual_mean"] = bh.Histogram(
            bh.axis.Regular(25, 0, 250), bh.axis.Regular(15, 0, 150), storage=bh.storage.WeightedMean(),
        )
        # Residual mean width
        hists[f"{grooming_method}_hybrid_det_jet_pt_residual_width"] = bh.Histogram(
            bh.axis.Regular(25, 0, 250), bh.axis.Regular(15, 0, 150), storage=bh.storage.WeightedMean(),
        )
        hists[f"{grooming_method}_hybrid_true_jet_pt_residual_width"] = bh.Histogram(
            bh.axis.Regular(25, 0, 250), bh.axis.Regular(15, 0, 150), storage=bh.storage.WeightedMean(),
        )
        hists[f"{grooming_method}_det_true_jet_pt_residual_width"] = bh.Histogram(
            bh.axis.Regular(25, 0, 250), bh.axis.Regular(15, 0, 150), storage=bh.storage.WeightedMean(),
        )
        # Residual
        hists[f"{grooming_method}_hybrid_det_jet_pt_residual"] = bh.Histogram(
            bh.axis.Regular(15, 0, 150), bh.axis.Regular(150, -1.5, 1.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_hybrid_true_jet_pt_residual"] = bh.Histogram(
            bh.axis.Regular(15, 0, 150), bh.axis.Regular(150, -1.5, 1.5), storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_det_true_jet_pt_residual"] = bh.Histogram(
            bh.axis.Regular(15, 0, 150), bh.axis.Regular(150, -1.5, 1.5), storage=bh.storage.Weight(),
        )

    progress_manager = enlighten.Manager()
    # TODO: Figure out how to get 72 from uproot.iterate
    with progress_manager.counter(total=72, desc="Analyzing", unit="tree", leave=True) as tree_counter:
        for df_path, df in tree_counter(data_frames):
            logger.debug(f"Processing df from {df_path}")
            hybrid_jet_pt_mask = (df["jet_pt_hybrid"] > 40) & (df["jet_pt_hybrid"] < 120)
            for grooming_method in grooming_methods:

                matching_leading = df[f"{grooming_method}_hybrid_detector_matching_leading"]
                matching_subleading = df[f"{grooming_method}_hybrid_detector_matching_subleading"]

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
                masked_df = df[mask]
                # Residual mean
                hists[f"{grooming_method}_hybrid_det_jet_pt_residual_mean"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    masked_df["jet_pt_hybrid"].to_numpy(),
                    sample=(
                        (masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]) / masked_df["jet_pt_det_level"]
                    ).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_hybrid_true_jet_pt_residual_mean"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    masked_df["jet_pt_hybrid"].to_numpy(),
                    sample=(
                        (masked_df["jet_pt_hybrid"] - masked_df["jet_pt_true"]) / masked_df["jet_pt_true"]
                    ).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_det_true_jet_pt_residual_mean"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    masked_df["jet_pt_hybrid"].to_numpy(),
                    sample=(
                        (masked_df["jet_pt_det_level"] - masked_df["jet_pt_true"]) / masked_df["jet_pt_true"]
                    ).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                # Residual width
                hists[f"{grooming_method}_hybrid_det_jet_pt_residual_width"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    masked_df["jet_pt_hybrid"].to_numpy(),
                    sample=(masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_hybrid_true_jet_pt_residual_width"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    masked_df["jet_pt_hybrid"].to_numpy(),
                    sample=(masked_df["jet_pt_hybrid"] - masked_df["jet_pt_true"]).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_det_true_jet_pt_residual_width"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    masked_df["jet_pt_hybrid"].to_numpy(),
                    sample=(masked_df["jet_pt_det_level"] - masked_df["jet_pt_true"]).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                # Full residual as a function of true pt. We can select the true jet pt range when plotting.
                # NOTE: We intentionally didn't apply a hybrid jet pt cut.
                masked_df = df[mask]
                hists[f"{grooming_method}_hybrid_det_jet_pt_residual"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    (
                        (masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]) / masked_df["jet_pt_det_level"]
                    ).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_hybrid_true_jet_pt_residual"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    ((masked_df["jet_pt_hybrid"] - masked_df["jet_pt_true"]) / masked_df["jet_pt_true"]).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_det_true_jet_pt_residual"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    ((masked_df["jet_pt_det_level"] - masked_df["jet_pt_true"]) / masked_df["jet_pt_true"]).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )

                for matching_type in _matching_name_to_axis_value:
                    mask = matching_selections[matching_type]
                    masked_df = df[mask & hybrid_jet_pt_mask]
                    # Axes: hybrid_level_jet_pt, det_level_jet_pt, residual
                    hists[f"{grooming_method}_hybrid_det_jet_pt_residuals_matching_type_{matching_type}"].fill(
                        masked_df["jet_pt_hybrid"].to_numpy(),
                        masked_df["jet_pt_det_level"].to_numpy(),
                        (
                            (masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]) / masked_df["jet_pt_det_level"]
                        ).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                    # Axes: hybrid_level_jet_kt, det_level_jet_kt, residual
                    hists[f"{grooming_method}_hybrid_det_kt_residuals_matching_type_{matching_type}"].fill(
                        masked_df[f"{grooming_method}_hybrid_kt"].to_numpy(),
                        masked_df[f"{grooming_method}_det_level_kt"].to_numpy(),
                        (
                            (masked_df[f"{grooming_method}_hybrid_kt"] - masked_df[f"{grooming_method}_det_level_kt"])
                            / masked_df[f"{grooming_method}_det_level_kt"]
                        ).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                    # Axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt
                    hists[f"{grooming_method}_hybrid_det_kt_response_matching_type_{matching_type}"].fill(
                        masked_df["jet_pt_hybrid"].to_numpy(),
                        masked_df[f"{grooming_method}_hybrid_kt"].to_numpy(),
                        masked_df["jet_pt_det_level"].to_numpy(),
                        masked_df[f"{grooming_method}_det_level_kt"].to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                    # Axes: hybrid_pt, hybrid_R, det_level_pt, det_level_R, residual
                    hists[f"{grooming_method}_hybrid_det_delta_R_response_matching_type_{matching_type}"].fill(
                        masked_df["jet_pt_hybrid"].to_numpy(),
                        masked_df[f"{grooming_method}_hybrid_delta_R"].to_numpy(),
                        masked_df["jet_pt_det_level"].to_numpy(),
                        masked_df[f"{grooming_method}_det_level_delta_R"].to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )

    progress_manager.stop()

    # Add some helpful imports and definitions
    from importlib import reload  # noqa: F401

    try:
        # May not want to import if developing.
        from jet_substructure.analysis import plot_from_skim  # noqa: F401
    except SyntaxError:
        logger.info("Couldn't load plot_from_skim due to syntax error. You need to load it.")

    output_dir = Path("output/embedPythia/skim")

    IPython.start_ipython(user_ns=locals())

    # Plotting
    # plot_from_skim.plot_residuals_by_matching_type(
    #     hists=hists, grooming_methods=grooming_methods, matching_types=list(_matching_name_to_axis_value.keys()), output_dir=output_dir
    # )
    # plot_from_skim.plot_residuals(hists=hists, grooming_methods=grooming_methods, output_dir=output_dir)
    # plot_from_skim.plot_response_by_matching_type(
    #     hists=hists, grooming_methods=grooming_methods, matching_types=list(_matching_name_to_axis_value.keys()), output_dir=output_dir,
    # )


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


if __name__ == "__main__":
    helpers.setup_logging()
    df_from_file()
    # dask_df_from_file()
    # dask_df_from_delayed()
    # map_reduce_pandas_concat()
