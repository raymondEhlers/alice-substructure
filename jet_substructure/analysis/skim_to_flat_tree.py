#!/usr/bin/env python3

""" Skim train output to a flat tree.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>
"""

from __future__ import annotations

import functools
import logging
import operator
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, cast

import attr
import enlighten
import IPython
import numpy as np
import uproot
from pathos.multiprocessing import ProcessingPool as Pool

from jet_substructure.analysis import analyze_tree
from jet_substructure.base import analysis_objects, data_manager, helpers, substructure_methods
from jet_substructure.base.helpers import UprootArray


logger = logging.getLogger(__name__)


@attr.s
class Calculation:
    """ Similar to `FillHistogramInput`, but adds the splittings indices.

    Note:
        The splitting indices are the overall indices of the input splittings within
        the entire splittings array. The indices are those of the splittings selected
        by the calculation.
    """

    input_jets: substructure_methods.SubstructureJetArray = attr.ib()
    input_splittings: substructure_methods.JetSplittingArray = attr.ib()
    input_splittings_indices: UprootArray[int] = attr.ib()
    values: UprootArray[float] = attr.ib()
    indices: UprootArray[int] = attr.ib()

    @property
    def splittings(self) -> substructure_methods.JetSplittingArray:
        try:
            return self._restricted_splittings
        except AttributeError:
            self._restricted_splittings: substructure_methods.JetSplittingArray = self.input_splittings[self.indices]
        return self._restricted_splittings

    @property
    def absolute_splittings_index(self) -> UprootArray[int]:
        try:
            return self._absolute_splittings_index
        except AttributeError:
            self._absolute_splittings_index: UprootArray[int] = self.input_splittings_indices[self.indices]
        return self._restricted_splittings

    @property
    def n_jets(self) -> int:
        """ Number of jets.

        Need to determine all jets which are accepted in the jet pt range.
        Otherwise, those which may fail (such as with a z_cutoff) may not get
        the proper normalization.
        """
        return len(self.input_jets)

    def __getitem__(self, mask: np.ndarray) -> Calculation:
        """ Mask the stored values, returning a new object. """
        # Validation
        if len(self.input_jets) != len(mask):
            raise ValueError(
                f"Mask length is different than array lengths. mask length: {len(mask)}, array lengths: {len(self.input_jets)}"
            )

        # Return the masked arrays in a new object.
        return type(self)(
            # NOTE: It's super important to use the input variables. Otherwise, we'll try to apply the indices twice
            #       (which won't work for the masked object).
            input_jets=self.input_jets[mask],
            input_splittings=self.input_splittings[mask],
            input_splittings_indices=self.input_splittings_indices[mask],
            values=self.values[mask],
            indices=self.indices[mask],
        )


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
            yield "_".join([self.grooming_method, prefix, k]), v


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


def _soft_drop_wrapper(
    splittings: substructure_methods.JetSplittingArray, z_cutoff: float
) -> Tuple[UprootArray[float], UprootArray[int]]:
    """ Wrap SD calculation to drop n_sd so it fits in the same ouput as the other calculations. """
    values, n_sd, indices = splittings.soft_drop(z_cutoff=z_cutoff)
    return values, indices


def _define_calculation_funcs(
    dataset: analysis_objects.Dataset, iterative_splittings: bool,
) -> Dict[str, functools.partial[Tuple[UprootArray[float], UprootArray[int]]]]:
    """ Define the calculation functions of interest.

    Note:
        The type of the inclusive is different, but it takes and returns the same sets of arguments
        as the other functions.

    Args:
        dataset: Dataset properties necessary to fully specify the calculations.
    Returns:
        dynamical_z, dynamical_kt, dynamical_time, leading_kt, leading_kt z>0.2, leading_kt z>0.4, SD z>0.2, SD z>0.4,
    """
    functions = {
        "dynamical_z": functools.partial(substructure_methods.JetSplittingArray.dynamical_z, R=dataset.settings.jet_R),
        "dynamical_kt": functools.partial(
            substructure_methods.JetSplittingArray.dynamical_kt, R=dataset.settings.jet_R
        ),
        "dynamical_time": functools.partial(
            substructure_methods.JetSplittingArray.dynamical_time, R=dataset.settings.jet_R
        ),
        "leading_kt": functools.partial(substructure_methods.JetSplittingArray.leading_kt,),
        "leading_kt_z_cut_02": functools.partial(substructure_methods.JetSplittingArray.leading_kt, z_cutoff=0.2),
        "leading_kt_z_cut_04": functools.partial(substructure_methods.JetSplittingArray.leading_kt, z_cutoff=0.4),
    }
    # TODO: This currently only works for iterative splittings...
    #       Calculating recursive is way harder in any array-like manner.
    if iterative_splittings:
        functions["soft_drop_z_cut_02"] = functools.partial(_soft_drop_wrapper, z_cutoff=0.2)
        functions["soft_drop_z_cut_04"] = functools.partial(_soft_drop_wrapper, z_cutoff=0.4)
    return functions


def _select_and_retrieve_splittings(
    jets: substructure_methods.SubstructureJetArray, mask: UprootArray[bool], iterative_splittings: bool
) -> Tuple[substructure_methods.SubstructureJetArray, substructure_methods.JetSplittingArray, UprootArray[int]]:
    """ Generalization of the function in analyze_tree to add the splitting index.

    """
    restricted_jets, restricted_splittings = analyze_tree._select_and_retrieve_splittings(
        jets, mask, iterative_splittings
    )
    # Add the indices.
    if iterative_splittings:
        restricted_splittings_indices = restricted_jets.subjets.iterative_splitting_index
    else:
        restricted_splittings_indices = restricted_jets.splittings.kt.localindex
    return restricted_jets, restricted_splittings, restricted_splittings_indices


def calculate_splitting_number(
    all_splittings: substructure_methods.JetSplittingArray,
    selected_splittings: substructure_methods.JetSplittingArray,
    restricted_splittings_indices: UprootArray[int],
) -> np.ndarray:
    # logger.debug("Calculating splitting number")
    # Setup
    # We need the parent index of all of the splittings and of those which we have selected.
    # The restricted splittings aren't enough on their own because they may not contain all of
    # the necessary splitting history to reconstruct the splitting.
    all_splittings_parent_index = all_splittings.parent_index
    parent_index = selected_splittings.parent_index
    counts = np.zeros_like(all_splittings_parent_index, dtype=np.int)

    # The general procedure is that we will mask as true all parent_index != -1
    # If those pass all of the cuts (including that it is in the restricted splittings)
    # then we increment the count. Once a parent index gets to -1, then we stop selecting it
    # in our mask, so it stops being updated.
    # NOTE: In general, we don't want to iterative with these type of arrays, but it's
    #       unavoidable here. And I don't think it should loop more than 30-40 times in the
    #       worst case (and often much less).
    while True:
        # First, we need to access if we're done. If so, all parent_index values will be -1.
        mask = parent_index != -1
        # Need two all() calls because the mask is jagged (with dim one of the jagged axis).
        if (mask != True).all().all():  # noqa: E712
            break
        # Need to repeat the parent_index to be the same shape as the restricted splittings so we can
        # check if any are equal. If any are equal, then that splitting is in the restricted group.
        # NOTE: We fill padded values with -2 because that can't possibly be a splitting index.
        parent_repeated_to_be_same_shape = (
            restricted_splittings_indices.ones_like() * parent_index.pad(1).fillna(-2).flatten()
        )
        accept_mask = (parent_repeated_to_be_same_shape == restricted_splittings_indices).any()
        # In the case that the parent_index of our splitting is in the selected splittings,
        # and it hasn't gotten to the origin, we can finally increment our count.
        # NOTE: Need to pad, fill, and flatten to match the shape of the accept_mask (which is just an ndarray mask)
        counts[mask.pad(1).fillna(False).flatten() & accept_mask] += 1
        # We retrieve the parents, and then assign them for those which are not yet at the origin.
        parent_index[mask] = all_splittings_parent_index[parent_index][mask]

    # logger.debug("Finished splitting number calculation")
    return counts


# def calculate_soft_drop(
#    all_splittings: substructure_methods.JetSplittingArray,
#    restricted_splittings_indices: UprootArray[int],
#    z_cutoff: float,
# ) -> Tuple[np.ndarray, UprootArray[int]]:
#    """
#
#    """
#    # TODO: Move to the splittings object.
#    # Start with the origin (NOTE: the relevant origin is 0 because -1 is a dummy node to start the splittings)
#    parent_index = all_splittings.localindex.zeros_like()
#    # Initial value should be outside of the standard range.
#    values = np.ones(len(all_splittings)) * substructure_methods.UNFILLED_VALUE
#    indices = all_splittings.localindex.ones_like() * -1
#    # The idea here is to iterate over the generations, starting at the origin.
#    while True:
#        # Select splittings that we ca
#        splittings_from_parent_mask = (all_splittings.parent_index == parent_index)
#        splittings = all_splittings[splittings_from_parent_mask]
#        pass_cutoff_mask = splittings.kt > z_cutoff
#
#        splittings_indices = all_splittings.localindex[splittings_from_parent_mask]
#        splittings_indices_needed_to_be_the_same_shape = restricted_splittings_indices.ones_like() * splittings_indices
#        restricted_mask = splittings_indices_needed_to_be_the_same_shape == restricted_splittings_indices
#        values[pass_cutoff_mask & restricted_mask] = splittings.z
#
#        parent_index = all_splittings.localindex[splittings.splittings_from_parent_mask].parent_index
#
#    return values, indices

# def calculate_soft_drop(
#    all_splittings: substructure_methods.JetSplittingArray,
#    restricted_splittings_indices: UprootArray[int],
#    z_cutoff: float,
# ) -> Tuple[np.ndarray, UprootArray[int]]:
#    # Start with the origin (NOTE: the relevant origin is 0 because -1 is a dummy node to start the splittings)
#    parent_index = all_splittings.localindex.zeros_like()
#    # Initial value should be outside of the standard range.
#    values = np.ones(len(all_splittings)) * substructure_methods.UNFILLED_VALUE
#    indices = all_splittings.localindex.ones_like() * -2
#    # The idea is to step through the generations of splittings.
#    #while True:
#    #    splittings_contributing_to_parent =
#    #    ...


def calculate_and_skim_embedding(
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    iterative_splittings: bool,
    draw_example_splittings: bool = False,
) -> bool:
    """ Determine the response and prong matching for jets substructure techniques.

    Why combine them together? Because then we only have to open and process a tree once.
    At a future date (beyond the start of April 2020), it would be better to refactor them more separately,
    such that we can enable or disable the different options and still have appropriate return values.
    But for now, we don't worry about it.
    """
    # Validation
    prefixes = ["matched", "detLevel", "data"]
    # Setup
    # Perhaps make these into arguments?
    iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
    # TODO: Maybe convert to hdf5? But maybe not because of compression?
    output_dir = tree.filename.parent / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{tree.filename.stem}_{iterative_splittings_label}_splittings.root"

    train_number = tree.filename.parent.name
    analysis_settings = cast(analysis_objects.PtHardAnalysisSettings, dataset.settings)
    pt_hard_bin = analysis_settings.train_number_to_pt_hard_bin[int(train_number)]
    scale_factor = analysis_settings.scale_factors[pt_hard_bin]

    # Actual setup.
    logger.info(f"Skimming tree from file {tree.filename}")
    successfully_accessed_data, all_jets = analyze_tree.load_jets_from_tree(tree=tree, prefixes=prefixes)
    true_jets, det_level_jets, hybrid_jets = all_jets
    if not successfully_accessed_data:
        return False

    ## Do the calculations
    mask = (
        (true_jets.constituents.counts > 1)
        & (det_level_jets.constituents.counts > 1)
        & (hybrid_jets.constituents.counts > 1)
    )
    # Require that we have jets that aren't dominated by hybrid jets.
    # It's super important to be ">=". That allows the leading jet in the hybrid to be the same
    # as the leading jet in the true (which would be good - we've probably found the right jet).
    # NOTE: We already apply this cut at the analysis level, so it shouldn't really do anything here.
    #       We're just applying it again to be certain.
    mask = mask & (true_jets.constituents.max_pt >= hybrid_jets.constituents.max_pt)

    # Mask the jets
    masked_true_jets, masked_true_jet_splittings, masked_true_jet_splittings_indices = _select_and_retrieve_splittings(
        true_jets, mask, iterative_splittings
    )
    (
        masked_det_level_jets,
        masked_det_level_jet_splittings,
        masked_det_level_jet_splittings_indices,
    ) = _select_and_retrieve_splittings(det_level_jets, mask, iterative_splittings)
    (
        masked_hybrid_jets,
        masked_hybrid_jet_splittings,
        masked_hybrid_jet_splittings_indices,
    ) = _select_and_retrieve_splittings(hybrid_jets, mask, iterative_splittings)

    grooming_results = {}
    grooming_results["scale_factor"] = np.ones_like(true_jets.jet_pt[mask]) * scale_factor
    # Add jet pt for all prefixes.
    grooming_results["jet_pt_true"] = masked_true_jets.jet_pt
    grooming_results["jet_pt_det_level"] = masked_det_level_jets.jet_pt
    grooming_results["jet_pt_hybrid"] = masked_hybrid_jets.jet_pt

    # Perform our calculations.
    functions = _define_calculation_funcs(dataset, iterative_splittings=iterative_splittings)
    for func_name, func in functions.items():
        true_jets_calculation = Calculation(
            masked_true_jets,
            masked_true_jet_splittings,
            masked_true_jet_splittings_indices,
            *func(masked_true_jet_splittings),
        )
        det_level_jets_calculation = Calculation(
            masked_det_level_jets,
            masked_det_level_jet_splittings,
            masked_det_level_jet_splittings_indices,
            *func(masked_det_level_jet_splittings),
        )
        hybrid_jets_calculation = Calculation(
            masked_hybrid_jets,
            masked_hybrid_jet_splittings,
            masked_hybrid_jet_splittings_indices,
            *func(masked_hybrid_jet_splittings),
        )

        for prefix, calculation in [
            ("true", true_jets_calculation),
            ("det_level", det_level_jets_calculation),
            ("hybrid", hybrid_jets_calculation),
        ]:
            groomed_splittings = calculation.splittings
            splitting_number = calculate_splitting_number(
                all_splittings=calculation.input_jets.splittings,
                selected_splittings=groomed_splittings,
                restricted_splittings_indices=calculation.input_splittings_indices,
            )

            # We pad with the UNFILLED_VALUE constant to account for any calculations that don't find a splitting.
            grooming_result = GroomingResultForTree(
                grooming_method=func_name,
                delta_R=groomed_splittings.delta_R.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                z=groomed_splittings.z.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                kt=groomed_splittings.kt.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                # Splitting number is already flattened.
                n=splitting_number,
            )
            grooming_results.update(grooming_result.asdict(prefix=prefix))

        # Need to mask for calculations which have no indices (ie didn't find any that met criteria.
        mask = (det_level_jets_calculation.indices.counts != 0) & (hybrid_jets_calculation.indices.counts != 0)
        try:
            masked_det_level_jets_calculation = det_level_jets_calculation[mask]
            masked_hybrid_jets_calculation = hybrid_jets_calculation[mask]
        except IndexError as e:
            logger.warning(e)
            IPython.start_ipython(user_ns=locals())

        # Matching
        # Perform hybrid-detector level matching.
        logger.info(f"Performing hybrid-det level matching for {func_name}")
        leading_matching, subleading_matching = analyze_tree.determine_matched_jets(
            hybrid_inputs=analysis_objects.FillHistogramInput(
                jets=masked_hybrid_jets_calculation.input_jets,
                splittings=masked_hybrid_jets_calculation.input_splittings,
                values=masked_hybrid_jets_calculation.values,
                indices=masked_hybrid_jets_calculation.indices,
            ),
            matched_inputs=analysis_objects.FillHistogramInput(
                jets=masked_det_level_jets_calculation.input_jets,
                splittings=masked_det_level_jets_calculation.input_splittings,
                values=masked_det_level_jets_calculation.values,
                indices=masked_det_level_jets_calculation.indices,
            ),
        )
        # Store leading, subleading matches
        for label, matching in [("leading", leading_matching), ("subleading", subleading_matching)]:
            # We'll store the output in an array, and then store that in the overall output with a mask
            # We need the additional mask because we can't perform matching for every jet (single particle jets, etc).
            output = np.zeros(len(det_level_jets_calculation.input_jets), dtype=np.int)
            matching_output = np.zeros(len(masked_det_level_jets_calculation.input_jets), dtype=np.int)
            matching_output[matching.properly] = 1
            matching_output[matching.mistag] = 2
            matching_output[matching.failed] = 3
            output[mask] = matching_output
            grooming_results[f"{func_name}_hybrid_detector_matching_{label}"] = output

        # Look for leading kt just because it's easier to understand conceptually.
        if draw_example_splittings and func_name == "leading_kt" and (leading_matching.properly & subleading_matching.failed).any():  # type: ignore
            from jet_substructure.analysis import draw_splitting

            # Find a sufficiently interesting jet (ie high enough pt)
            mask_jets_of_interest = (leading_matching.properly & subleading_matching.failed) & (
                masked_hybrid_jets.jet_pt > 80
            )
            # Look at most the first 5 jets.
            for i, hybrid_jet in enumerate(masked_hybrid_jets[mask_jets_of_interest][:5]):
                # Find the hybrid jet and splitting of interest.
                # hybrid_jet = masked_hybrid_jets[mask_jets_of_interest][0]
                # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                hybrid_jet_selected_splitting_index = hybrid_jets_calculation.indices[mask_jets_of_interest][i][0]  # type: ignore
                # Same for det level.
                det_level_jet = masked_det_level_jets[mask_jets_of_interest][i]
                # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                det_level_jet_selected_splitting_index = det_level_jets_calculation.indices[mask_jets_of_interest][i][0]  # type: ignore

                # Draw the splittings
                draw_splitting.splittings_graph(
                    jet=hybrid_jet,
                    path=dataset.output.parent / "leading_correct_subleading_failed/",
                    filename=f"{i}_hybrid_splittings_jet_pt_{hybrid_jet.jet_pt:.1f}GeV_selected_splitting_index_{hybrid_jet_selected_splitting_index}",
                    show_subjet_pt=True,
                    selected_splitting_index=hybrid_jet_selected_splitting_index,
                )
                draw_splitting.splittings_graph(
                    jet=det_level_jet,
                    path=dataset.output.parent / "leading_correct_subleading_failed/",
                    filename=f"{i}_det_level_splittings_jet_pt_{det_level_jet.jet_pt:.1f}GeV_selected_splitting_index_{det_level_jet_selected_splitting_index}",
                    show_subjet_pt=True,
                    selected_splitting_index=det_level_jet_selected_splitting_index,
                )

    branches = {k: v.dtype for k, v in grooming_results.items()}
    logger.info(f"Writing skim to {output_filename}")
    with uproot.recreate(output_filename) as output_file:
        output_file["tree"] = uproot.newtree(branches)
        # Write all of the calculations
        output_file["tree"].extend(grooming_results)

    logger.info(f"Finished processing tree from file {tree.filename}")
    return True


def calculate_and_skim_data(
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    iterative_splittings: bool,
    prefixes: Optional[Sequence[str]] = None,
) -> bool:
    # Validation
    if prefixes is None:
        prefixes = ["data"]

    # Setup
    # Perhaps make these into arguments?
    iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
    output_dir = tree.filename.parent / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{tree.filename.stem}_{iterative_splittings_label}_splittings.root"

    # Actual setup.
    logger.info(f"Skimming tree from file {tree.filename}")
    successfully_accessed_data, all_jets = analyze_tree.load_jets_from_tree(tree=tree, prefixes=prefixes)
    if not successfully_accessed_data:
        return False
    # Only unpack if we've successfully accessed the data.

    # Dataset wide masks
    # Select everything by default.
    mask = np.ones_like(all_jets[0].jet_pt) > 0

    # Apparently I can get pt hard < 5. Which is bizarre. Filter these out when applicable.
    if dataset.collision_system == "pythia":
        # App
        mask = mask & (tree["ptHard"] >= 5.0)

    masked_jets: Dict[
        str, Tuple[substructure_methods.SubstructureJetArray, substructure_methods.JetSplittingArray, UprootArray[int]]
    ] = {}
    for prefix, input_jets in zip(prefixes, all_jets):
        masked_jets[prefix] = _select_and_retrieve_splittings(
            input_jets, mask, iterative_splittings=iterative_splittings,
        )

    # Results output
    grooming_results: Dict[str, np.ndarray] = {}
    # Add jet pt for all prefixes.
    for prefix, jets in masked_jets.items():
        grooming_results[f"jet_pt_{prefix}"] = jets[0].jet_pt

    # Add scale factors when appropriate (ie for pythia)
    if dataset.collision_system == "pythia":
        # Need to redo the pt hard bin ranges because we apparently get a pt hard larger than 1000!
        # So we do it again here.
        # pt_hard_bin_ranges = np.array([0, 5, 7, 9, 12, 16, 21, 28, 36, 45, 57, 70, 85, 99, 115, 132, 150, 169, 190, 212, 235, 2000])
        # Need to subtract 1 because our first bin is 1-indexed.
        # pt_hard_bins = np.searchsorted(pt_hard_bin_ranges, tree["ptHard"]) - 1
        # print(np.unique(pt_hard_bins))
        analysis_settings = cast(analysis_objects.PtHardAnalysisSettings, dataset.settings)
        scale_factors = dict(analysis_settings.scale_factors)
        # There is apparently a pt hard > 1000 in this dataset! This ends up with an entry in bin 21, which is weird.
        # So we copy the scale factor for pt hard bin 20 to 21 to cover it. It should be more or less correct.
        scale_factors[21] = scale_factors[20]

        pt_hard_bins = tree["ptHardBin"][mask]
        print(np.unique(pt_hard_bins))
        IPython.start_ipython(user_ns=locals())
        grooming_results.update(
            {
                "scale_factor": np.array([analysis_settings.scale_factors[b] for b in pt_hard_bins], dtype=np.float32),
                "pt_hard_bin": pt_hard_bins,
                "pt_hard": tree["ptHard"][mask],
            }
        )

    # Perform our calculations.
    functions = _define_calculation_funcs(dataset, iterative_splittings=iterative_splittings)
    for func_name, func in functions.items():
        for prefix, jets in masked_jets.items():
            calculation = Calculation(jets[0], jets[1], jets[2], *func(jets[1]),)

            groomed_splittings = calculation.splittings
            splitting_number = calculate_splitting_number(
                all_splittings=calculation.input_jets.splittings,
                selected_splittings=groomed_splittings,
                restricted_splittings_indices=calculation.input_splittings_indices,
            )

            # We pad with the UNFILLED_VALUE constant to account for any calculations that don't find a splitting.
            grooming_result = GroomingResultForTree(
                grooming_method=func_name,
                delta_R=groomed_splittings.delta_R.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                z=groomed_splittings.z.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                kt=groomed_splittings.kt.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                # Splitting number is already flattened.
                n=splitting_number,
            )
            grooming_results.update(grooming_result.asdict(prefix=prefix))

    branches = {k: v.dtype for k, v in grooming_results.items()}
    logger.info(f"Writing skim to {output_filename}")
    with uproot.recreate(output_filename) as output_file:
        output_file["tree"] = uproot.newtree(branches)
        # Write all of the calculations
        output_file["tree"].extend(grooming_results)

    logger.info(f"Finished processing tree from file {tree.filename}")
    return True


def run(
    collision_system: str,
    iterative_splittings: bool,
    calculate_and_skim_func: Callable[[data_manager.Tree, analysis_objects.Dataset, bool], bool],
    number_of_cores: int,
    additional_kwargs_for_analysis: Optional[Mapping[str, Any]] = None,
) -> None:
    # Validation
    if additional_kwargs_for_analysis is None:
        additional_kwargs_for_analysis = {}

    # Setup
    settings_class_map: Mapping[str, Type[analysis_objects.AnalysisSettings]] = {
        "pythia": analysis_objects.PtHardAnalysisSettings,
        "embedPythia": analysis_objects.PtHardAnalysisSettings,
    }
    dataset = analysis_objects.Dataset.from_config_file(
        collision_system=collision_system,
        config_filename=Path("config") / "datasets.yaml",
        override_filenames=None,
        hists_filename_stem="IGNORE",
        output_base=Path("output"),
        settings_class=settings_class_map.get(collision_system, analysis_objects.AnalysisSettings),
        # NOTE: This value is irrelevant for the skim...
        z_cutoff=0.2,
    )

    dm = data_manager.IterateTrees(
        filenames=dataset.filenames,
        tree_name=dataset.tree_name,
        # Mypy is getting confused by Sequence[str] because str is an iterable, so we ignore the type...
        branches=dataset.branches,  # type: ignore
    )
    logger.info("Setup complete. Beginning processing of trees.")

    progress_manager = enlighten.get_manager()
    # with progress_manager.counter(total=len(dm), desc="Analyzing", unit="tree") as tree_counter:
    #    for tree in tree_counter(dm):
    #        calculate_and_skim_embedding(tree=tree, dataset=dataset)

    number_of_trees_processed = 0
    dm_iterator = dm.lazy_iteration(fully_lazy=(number_of_cores > 1))
    wrapper = functools.partial(
        calculate_and_skim_func,
        dataset=dataset,
        iterative_splittings=iterative_splittings,
        **additional_kwargs_for_analysis,
    )
    wrapper_multiprocessing = functools.partial(analyze_tree._wrap_multiprocessing, analysis_function=wrapper,)
    with progress_manager.counter(total=len(dm), desc="Skimming", unit="tree") as tree_counter:
        if number_of_cores > 1:
            with Pool(nodes=number_of_cores) as pool:
                number_of_trees_processed = functools.reduce(
                    operator.add, tree_counter(pool.imap(wrapper_multiprocessing, dm_iterator)),
                )
        else:
            number_of_trees_processed = functools.reduce(operator.add, tree_counter(map(wrapper, dm_iterator)),)

    logger.info(f"Processed {number_of_trees_processed} out of {len(dm)} trees!")

    # Cleanup
    progress_manager.stop()


if __name__ == "__main__":
    helpers.setup_logging()
    # Options
    iterative_splittings = True
    number_of_cores = 1

    # Run embedding
    run(
        collision_system="embedPythia",
        iterative_splittings=iterative_splittings,
        calculate_and_skim_func=calculate_and_skim_embedding,
        number_of_cores=number_of_cores,
        additional_kwargs_for_analysis={"draw_example_splittings": False},
    )
    # Run PbPb
    # run(
    #    collision_system="PbPb",
    #    iterative_splittings=iterative_splittings,
    #    calculate_and_skim_func=calculate_and_skim_data,
    #    number_of_cores=number_of_cores,
    # )
    run(
        collision_system="pythia",
        iterative_splittings=iterative_splittings,
        # mypy apparently doesn't handle adding arguments, even with callable protocols...
        # We only get away with this because the prefixes are optional.
        calculate_and_skim_func=calculate_and_skim_data,
        number_of_cores=number_of_cores,
        additional_kwargs_for_analysis={"prefixes": ["data", "matched"]},
    )
