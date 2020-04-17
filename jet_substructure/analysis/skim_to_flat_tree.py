#!/usr/bin/env python3

""" Skim train output to a flat tree.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple, Type

import attr
import numpy as np
import uproot

from jet_substructure.analysis import analyze_tree
from jet_substructure.base import analysis_objects, data_manager, helpers, substructure_methods


logger = logging.getLogger(__name__)


@attr.s
class GroomingResultForTree:
    grooming_method: str = attr.ib()
    delta_R: np.ndarray = attr.ib()
    z: np.ndarray = attr.ib()
    kt: np.ndarray = attr.ib()
    n: np.ndarray = attr.ib()

    def asdict(self, prefix: str) -> Iterable[Tuple[str, np.ndarray]]:
        for k, v in attr.asdict(self, recurse=False).items():
            # Skip the label
            if isinstance(v, str):
                continue
            yield "_".join([prefix, self.grooming_method, k]), v


# def define_splitting_branches(prefix: str, grooming_method: str) -> Dict[str, np.dtype]:
#    return {
#        f"{prefix}_{grooming_method}_R": np.float32,
#        f"{prefix}_{grooming_method}_z": np.float32,
#        f"{prefix}_{grooming_method}_kt": np.float32,
#        f"{prefix}_{grooming_method}_n": np.float32,
#    }
#
# def branches_for_prefix(prefix: str, grooming_methods: Sequence[str]) -> Dict[str, np.dtype]:
#    branches = {
#        f"{prefix}_jet_pt",
#    }
#    for grooming_method in grooming_methods:
#        branches.update(define_splitting_branches(prefix=prefix, grooming_method=grooming_method))
#
#    return branches


def calculate_grooming_methods(
    dataset: analysis_objects.Dataset,
    prefix: str,
    jets: substructure_methods.SubstructureJetArray,
    splittings: substructure_methods.JetSplittingArray,
) -> Dict[str, np.ndarray]:
    # Setup
    # TODO: Do this more cleanly.
    (
        inclusive_func,
        dynamical_z_func,
        dynamical_kt_func,
        dynamical_time_func,
        leading_kt_func,
        leading_kt_hard_cutoff_func,
    ) = analyze_tree._define_calculation_funcs(dataset)

    func_map = {
        "dynamical_z": dynamical_z_func,
        "dynamical_kt": dynamical_kt_func,
        "dynamical_time": dynamical_time_func,
        "leading_kt": leading_kt_func,
        "leading_kt_z_cut_02": leading_kt_hard_cutoff_func,
    }

    # Extract the relevant branches.
    results = {}
    results[f"{prefix}_jet_pt"] = jets.jet_pt
    for name, func in func_map.items():
        values, indices = func(splittings)
        groomed_splittings = splittings[indices]

        # TODO: Properly extract the number of splittings...
        grooming_result = GroomingResultForTree(
            grooming_method=name,
            delta_R=groomed_splittings.delta_R.pad(1).fillna(-0.05).flatten(),
            z=groomed_splittings.z.pad(1).fillna(-0.05).flatten(),
            kt=groomed_splittings.kt.pad(1).fillna(-0.05).flatten(),
            n=(groomed_splittings.kt.pad(1).fillna(-0.05).ones_like() * 1).flatten(),
        )

        results.update(grooming_result.asdict(prefix=prefix))

    return results


def calculate_and_skim_embedding(tree: data_manager.Tree, dataset: analysis_objects.Dataset,) -> bool:
    """ Determine the response and prong matching for jets substructure techniques.

    Why combine them together? Because then we only have to open and process a tree once.
    At a future date (beyond the start of April 2020), it would be better to refactor them more separately,
    such that we can enable or disable the different options and still have appropriate return values.
    But for now, we don't worry about it.
    """
    # Setup
    # Perhaps make these into arguments?
    prefixes = ["data", "matched", "detLevel"]
    # grooming_methods = ["dynamical_z", "dynamical_kt", "dynamical_time", "leading_kt", "leading_kt_z_cut_02", "leading_kt_z_cut_04"]
    iterative_splittings = True
    iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
    # TODO: Maybe convert to hdf5? But maybe not because of compression?
    output_filename = f"{tree.filename}_{iterative_splittings_label}_splittings.root"

    # Actual setup.
    logger.info(f"Skimming tree from file {tree.filename}")
    successfully_accessed_data, all_jets = analyze_tree.load_jets_from_tree(tree=tree, prefixes=prefixes)
    hybrid_jets, true_jets, det_level_jets = all_jets
    if not successfully_accessed_data:
        return False

    ## Do the calculations
    mask = (
        (hybrid_jets.constituents.counts > 1)
        & (true_jets.constituents.counts > 1)
        & (det_level_jets.constituents.counts > 1)
    )
    # Require that we have jets that aren't dominated by hybrid jets.
    # It's super important to be ">=". That allows the leading jet in the hybrid to be the same
    # as the leading jet in the true (which would be good - we've probably found the right jet).
    mask = mask & (true_jets.constituents.max_pt >= hybrid_jets.constituents.max_pt)

    # TODO: Scale factor!!

    grooming_results = {}
    # branches = {}
    # for prefix in prefixes:
    #    branches_for_prefix(prefix=prefix, grooming_methods=grooming_methods)

    # Then restrict our jets.
    for prefix, jets in zip(prefixes, all_jets):
        restricted_jets, restricted_splittings = analyze_tree._select_and_retrieve_splittings(
            jets, mask, iterative_splittings
        )
        res = calculate_grooming_methods(
            dataset=dataset, prefix=prefix, jets=restricted_jets, splittings=restricted_splittings
        )
        # import IPython;
        # IPython.start_ipython(user_ns=locals())
        grooming_results.update(res)

    branches = {k: v.dtype for k, v in grooming_results.items()}
    with uproot.recreate(output_filename) as output_file:
        output_file["tree"] = uproot.newtree(branches)
        # Write all of the calculations
        output_file["tree"].extend(grooming_results)

    logger.info(f"Finished processing tree from file {tree.filename}")
    return True


if __name__ == "__main__":
    helpers.setup_logging()
    settings_class_map: Mapping[str, Type[analysis_objects.AnalysisSettings]] = {
        "embedPythia": analysis_objects.PtHardAnalysisSettings,
    }
    dataset = analysis_objects.Dataset.from_config_file(
        collision_system="embedPythia",
        config_filename=Path("config") / "datasets.yaml",
        override_filenames=None,
        hists_filename_stem="embedding_hists",
        output_base=Path("output"),
        settings_class=settings_class_map.get("embedPythia", analysis_objects.AnalysisSettings),
        z_cutoff=0.2,
    )

    # Setup dataset
    dm = data_manager.IterateTrees(
        filenames=[Path("trains/embedPythia/5536/AnalysisResults.18q.1.chunk1.root")],
        tree_name=dataset.tree_name,
        # Mypy is getting confused by Sequence[str] because str is an iterable, so we ignore the type...
        branches=dataset.branches,  # type: ignore
    )
    logger.info("Setup complete. Beginning processing of trees.")

    for tree in dm:
        calculate_and_skim_embedding(tree=tree, dataset=dataset)
