#!/usr/bin/env python3

"""

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple, cast

import attr
import awkward1 as ak
import IPython
import numba as nb
import numpy as np
import uproot as uproot3
from pachyderm import yaml

from jet_substructure.analysis import analyze_tree
from jet_substructure.base import analysis_objects, data_manager, new_methods
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

    input_jets: ak.Array = attr.ib()
    input_splittings: new_methods.JetSplittingArray = attr.ib()
    input_splittings_indices: UprootArray[int] = attr.ib()
    values: UprootArray[float] = attr.ib()
    indices: UprootArray[int] = attr.ib()
    # If there's no additional grooming selection, then this will be identical to input_splittings_indices.
    possible_indices: UprootArray[int] = attr.ib()

    @property
    def splittings(self) -> new_methods.JetSplittingArray:
        try:
            return self._restricted_splittings
        except AttributeError:
            self._restricted_splittings: new_methods.JetSplittingArray = self.input_splittings[self.indices]
        return self._restricted_splittings

    @property
    def n_jets(self) -> int:
        """ Number of jets. """
        # We flatten the splittings because there may be jets (and consequently splittings) which aren't selected
        # at all due to the grooming (such as a z cut). Thus, we use the selected splittings directly.
        return len(self.splittings.flatten())

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
            possible_indices=self.possible_indices[mask],
        )


@attr.s
class GroomingResultForTree:
    grooming_method: str = attr.ib()
    delta_R: np.ndarray = attr.ib()
    z: np.ndarray = attr.ib()
    kt: np.ndarray = attr.ib()
    n_to_split: np.ndarray = attr.ib()
    n_groomed_to_split: np.ndarray = attr.ib()
    # For SoftDrop, this is equivalent to n_sd.
    n_passed_grooming: np.ndarray = attr.ib()

    def asdict(self, prefix: str) -> Iterable[Tuple[str, np.ndarray]]:
        for k, v in attr.asdict(self, recurse=False).items():
            # Skip the label
            if isinstance(v, str):
                continue
            yield "_".join([self.grooming_method, prefix, k]), v


def _define_calculation_functions(
    jet_R: float, iterative_splittings: bool,
) -> Dict[str, functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]]]:
    """ Define the calculation functions of interest.

    Note:
        The type of the inclusive is different, but it takes and returns the same sets of arguments
        as the other functions.

    Args:
        jet_R: Jet resolution parameter.
        iterative_splittings: Whether calculating iterative splittings or not.
    Returns:
        dynamical_z, dynamical_kt, dynamical_time, leading_kt, leading_kt z>0.2, leading_kt z>0.4, SD z>0.2, SD z>0.4
    """
    functions = {
        "dynamical_z": functools.partial(new_methods.JetSplittingArray.dynamical_z, R=jet_R),
        "dynamical_kt": functools.partial(new_methods.JetSplittingArray.dynamical_kt, R=jet_R),
        "dynamical_time": functools.partial(new_methods.JetSplittingArray.dynamical_time, R=jet_R),
        "leading_kt": functools.partial(new_methods.JetSplittingArray.leading_kt,),
        "leading_kt_z_cut_02": functools.partial(new_methods.JetSplittingArray.leading_kt, z_cutoff=0.2),
        "leading_kt_z_cut_04": functools.partial(new_methods.JetSplittingArray.leading_kt, z_cutoff=0.4),
    }
    # TODO: This currently only works for iterative splittings...
    #       Calculating recursive is way harder in any array-like manner.
    if iterative_splittings:
        functions["soft_drop_z_cut_02"] = functools.partial(new_methods.JetSplittingArray.soft_drop, z_cutoff=0.2)
        functions["soft_drop_z_cut_04"] = functools.partial(new_methods.JetSplittingArray.soft_drop, z_cutoff=0.4)
    return functions


def _select_and_retrieve_splittings(
    jets: ak.Array, mask: UprootArray[bool], iterative_splittings: bool
) -> Tuple[ak.Array, new_methods.JetSplittingArray, UprootArray[int]]:
    """ Generalization of the function in analyze_tree to add the splitting index.

    """
    # Ensure that there are sufficient counts
    restricted_jets = jets[mask]

    # Add the splittings and indices.
    if iterative_splittings:
        # Only keep iterative splittings.
        restricted_splittings = restricted_jets.jet_splittings.iterative_splittings(restricted_jets.subjets)

        # Enable this test to determine if we've selected different sets of splittings with the
        # recursive vs iterative selections.
        # if (splittings.counts != restricted_jets.jet_splittings.counts).any():
        #    logger.warning("Disagreement between number of inclusive and recursive splittings (as expected!)")
        #    IPython.embed()
        restricted_splittings_indices = restricted_jets.subjets.iterative_splitting_index
    else:
        restricted_splittings = restricted_jets.jet_splittings
        restricted_splittings_indices = restricted_jets.jet_splittings.kt.layout.localindex()

    return restricted_jets, restricted_splittings, restricted_splittings_indices


# @nb.jit
# def reproduce(
#    selected_splittings: new_methods.JetSplittingArray,
# ) -> np.ndarray:
#    output = np.zeros(len(selected_splittings), np.int16)
#
#    i = 0
#    for selected_splitting in selected_splittings:
#        if i > 28:
#            return
#        print("======== i =", i)
#        parent_indices = selected_splitting.parent_index
#        print("parent_indices", parent_indices)
#        j = 0
#        for p in parent_indices:
#            print(j, ":", p)
#            j += 1
#        i += 1
#        output[i] = 1
#
#    return output


@nb.njit
def calculate_splitting_number(
    all_splittings: new_methods.JetSplittingArray,
    selected_splittings: new_methods.JetSplittingArray,
    restricted_splittings_indices: UprootArray[int],
    debug: bool = False,
) -> np.ndarray:
    # TODO: Optimize the data sizes...
    output = np.zeros(len(selected_splittings), np.int16)

    for i, (selected_splitting, restricted_splitting_indices, available_splittings_parents) in enumerate(
        zip(selected_splittings, restricted_splittings_indices, all_splittings.parent_index)
    ):
        # restricted_splitting_indices = restricted_splittings_indices[i]
        # available_splittings_parents = all_splittings[i].parent_index

        parent_indices = selected_splitting.parent_index
        if len(parent_indices):
            # We have at least one splitting, so we add an entry for it.
            output[i] += 1

            parent_index = parent_indices[0]
            if debug:
                print("parent_index", parent_index, "restricted_splitting_indices", restricted_splitting_indices)
            # print("i", i, "parent_indices", parent_indices, "parent_index", parent_index, "restricted_splitting_indices", restricted_splitting_indices)
            # if i == 27:
            #    print("parent_indices", parent_indices, "parent_index", parent_index, "restricted_splitting_indices", restricted_splitting_indices)
            while parent_index != -1:
                # Apparently contains isn't implemented either. So we just implement by hand.
                # if parent_index in restricted_splitting_indices:
                for index in restricted_splitting_indices:
                    # print("parent_index: {parent_index}, index: {index}".format(parent_index=parent_index, index=index))
                    if debug:
                        print("parent_index", parent_index, "index", index)
                    # print("parent_index, index: %d, %d" % (parent_index, index))
                    # print("i", i, "parent_index", parent_index, "index", index)
                    if parent_index == index:
                        if debug:
                            print("Found parent index:", index)
                        output[i] += 1
                        # import IPython; IPython.embed()
                        parent_index = available_splittings_parents[parent_index]
                        if debug:
                            print("New parent index:", parent_index)
                        # print("Breaking...")
                        break
                else:
                    # We didn't find it, but we need to advance forward.
                    parent_index = available_splittings_parents[parent_index]

            if debug:
                print("output[i]", output[i])

        # i += 1

    return output


def calculate_splitting_number_old(
    all_splittings: new_methods.JetSplittingArray,
    selected_splittings: new_methods.JetSplittingArray,
    restricted_splittings_indices: UprootArray[int],
    debug: bool = False,
) -> np.ndarray:
    # logger.debug("Calculating splitting number")
    # Setup
    # We need the parent index of all of the splittings and of those which we have selected.
    # The restricted splittings aren't enough on their own because they may not contain all of
    # the necessary splitting history to reconstruct the splitting.
    all_splittings_parent_index = all_splittings.parent_index
    parent_index = selected_splittings.parent_index
    counts = all_splittings_parent_index * 0

    return counts

    IPython.embed()

    # First, increment all which have a selected splitting, meaning that if the splitting is at
    # the origin, it is the considered the 1st splitting (so we're reserving 0 for the untagged).
    counts[ak.num(selected_splittings) > 0] = counts[ak.num(selected_splittings) > 0] + 1

    # The general procedure is that we will mask as true all parent_index != -1
    # If those pass all of the cuts (including that it is in the restricted splittings)
    # then we increment the count. Once a parent index gets to -1, then we stop selecting it
    # in our mask, so it stops being updated.
    # NOTE: In general, we don't want to iterative with these type of arrays, but it's
    #       unavoidable here. And I don't think it should loop more than 30-40 times in the
    #       worst case (and often much less).
    while True:
        # First, we need to determine if we're done. If so, all parent_index values will be -1.
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

        if debug:
            IPython.start_ipython(user_ns=locals())
        # In the case that the parent_index of our splitting is in the selected splittings,
        # and it hasn't gotten to the origin, we can finally increment our count.
        # NOTE: Need to pad, fill, and flatten to match the shape of the accept_mask (which is just an ndarray mask)
        counts[ak.flatten(ak.fill_none(ak.pad_none(mask, 1), False)) & accept_mask] += 1
        # We retrieve the parents, and then assign them for those which are not yet at the origin.
        parent_index[mask] = all_splittings_parent_index[parent_index][mask]

    # logger.debug("Finished splitting number calculation")
    return counts


@nb.njit
def _find_contributing_subjets(input_jet, groomed_index: int):
    # subjets = []
    # for sj in input_jet.subjets:
    #    if sj.parent_splitting_index == groomed_index:
    #        subjets.append(sj)
    # return subjets
    return [sj for sj in input_jet.subjets if sj.parent_splitting_index == groomed_index]


@nb.njit
def _sort_subjets(input_jet, input_subjets):
    pts = []
    for sj in input_subjets:
        px = 0
        py = 0
        for constituent_index in sj.constituent_indices:
            constituent = input_jet.jet_constituents[constituent_index]
            px += constituent.pt * np.cos(constituent.phi)
            py += constituent.pt * np.sin(constituent.phi)
        pts.append(np.sqrt(px ** 2 + py ** 2))

    leading = input_subjets[0]
    subleading = input_subjets[1]

    if pts[1] > pts[0]:
        leading, subleading = subleading, leading

    return leading, subleading


@nb.njit
def _subjet_shared_momentum(
    generator_like_subjet,
    generator_like_jet,
    measured_like_subjet,
    measured_like_jet,
    match_using_distance: bool = False,
):
    delta = 0.01
    sum_pt = 0

    for generator_like_constituent_index in generator_like_subjet.constituent_indices:
        generator_like_constituent = generator_like_jet.jet_constituents[generator_like_constituent_index]
        for measured_like_constituent_index in measured_like_subjet.constituent_indices:
            measured_like_constituent = measured_like_jet.jet_constituents[measured_like_constituent_index]
            if match_using_distance:
                if np.abs(measured_like_constituent.eta - generator_like_constituent.eta) > delta:
                    continue
                if np.abs(measured_like_constituent.phi - generator_like_constituent.phi) > delta:
                    continue
            else:
                if generator_like_constituent.id != measured_like_constituent.id:
                    continue

            sum_pt += generator_like_constituent.pt
            # We've matched once - no need to match again.
            # Otherwise, the run the risk of summing a generator-like constituent pt twice.
            break

    return sum_pt


@nb.njit
def subjet_pt(subjet, jet):
    px = 0
    py = 0
    # pt = 0
    for constituent_index in subjet.constituent_indices:
        constituent = jet.jet_constituents[constituent_index]
        px += constituent.pt * np.cos(constituent.phi)
        py += constituent.pt * np.sin(constituent.phi)
        # pt += constituent.pt
    # return pt
    return np.sqrt(px ** 2 + py ** 2)


@nb.njit
def _subjet_contained_in_subjet(
    generator_like_subjet,
    generator_like_jet,
    measured_like_subjet,
    measured_like_jet,
    match_using_distance: bool = False,
):
    return (
        _subjet_shared_momentum(
            generator_like_subjet=generator_like_subjet,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_subjet,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        )
        / subjet_pt(generator_like_subjet, generator_like_jet)
    ) > 0.5


@nb.njit
def determine_matched_jets_numba(
    generator_like_jets,
    generator_like_splittings,
    generator_like_groomed_values,
    generator_like_groomed_indices,
    measured_like_jets,
    measured_like_splittings,
    measured_like_groomed_values,
    measured_like_groomed_indices,
    match_using_distance: bool,
) -> Dict[str, np.ndarray]:
    n_jets = len(measured_like_jets)
    leading_matching = np.ones(n_jets, dtype=np.int16) * -1
    subleading_matching = np.ones(n_jets, dtype=np.int16) * -1

    for (
        i,
        (
            generator_like_jet,
            generator_like_splitting,
            generator_like_groomed_value,
            generator_like_groomed_index_array,
            measured_like_jet,
            measured_like_splitting,
            measured_like_groomed_value,
            measured_like_groomed_index_array,
        ),
    ) in enumerate(
        zip(
            generator_like_jets,
            generator_like_splittings,
            generator_like_groomed_values,
            generator_like_groomed_indices,
            measured_like_jets,
            measured_like_splittings,
            measured_like_groomed_values,
            measured_like_groomed_indices,
        )
    ):
        # Find the selected index if it's available.
        if len(measured_like_groomed_index_array) > 0 and len(generator_like_groomed_index_array) > 0:
            # This is required. If we not, we handle the other cases and continue.
            pass
        elif len(measured_like_groomed_index_array) > 0:
            # Assign 0 for this case and move on.
            leading_matching[i] = 0
            subleading_matching[i] = 0
            continue
        else:
            # Use the default values and continue
            continue

        # We maintain the singles structure per jet so that each index can be applied to each jet (ie. array entry)
        # (this also lets us keep empty cases accounted for). However, we've now already accounted for empty cases,
        # and it's much easier to work with the individual values, so we extract them. We know each one will have only
        # one entry because it's from an argmax call.
        generator_like_groomed_index = generator_like_groomed_index_array[0]
        measured_like_groomed_index = measured_like_groomed_index_array[0]

        # Find the contributing subjets
        generator_like_subjets = _find_contributing_subjets(generator_like_jet, generator_like_groomed_index)
        measured_like_subjets = _find_contributing_subjets(measured_like_jet, measured_like_groomed_index)
        # print(measured_like_subjets)
        # Sort
        generator_like_leading, generator_like_subleading = _sort_subjets(generator_like_jet, generator_like_subjets)
        measured_like_leading, measured_like_subleading = _sort_subjets(measured_like_jet, measured_like_subjets)

        # Compare
        if _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_leading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_leading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            leading_matching[i] = 1
        elif _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_leading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_subleading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            leading_matching[i] = 2
        else:
            leading_matching[i] = 3

        if _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_subleading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_subleading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            subleading_matching[i] = 1
        elif _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_subleading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_leading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            subleading_matching[i] = 2
        else:
            subleading_matching[i] = 3

    return leading_matching, subleading_matching


def prong_matching_numba_wrapper(
    measured_like_jets_calculation: Calculation,
    measured_like_jets_label: str,
    generator_like_jets_calculation: Calculation,
    generator_like_jets_label: str,
    grooming_method: str,
    match_using_distance: bool = False,
) -> Dict[str, np.ndarray]:
    """ Performs prong matching for the provided collections.

    Note:
        0 is there were insufficient constituents to form a splitting, 1 is properly matched, 2 is mistagged
        (leading -> subleading or subleading -> leading), 3 is untagged (failed).

    Args:
        measured_like_jets_calculation: Grooming calculation for measured-like jets (hybrid for hybrid-det level matching).
        measured_like_jets_label: Label for measured jets (hybrid for hybrid-det level matching).
        generator_like_jets_calculation: Grooming calculation for generator-like jets (det level for hybrid-det level matching).
        generator_like_jets_label: Label for generator jets (det_level for hybrid-det level matching).
        grooming_method: Name of the grooming method.
        match_using_distance: If True, match using distance. Otherwise, match using the stored label.
    Returns:
        Matching and subleading matching values.
    """
    ...

    # Matching
    grooming_results = {}
    logger.info(f"Performing {measured_like_jets_label}-{generator_like_jets_label} matching for {grooming_method}")
    leading_matching, subleading_matching = determine_matched_jets_numba(
        generator_like_jets=generator_like_jets_calculation.input_jets,
        generator_like_splittings=generator_like_jets_calculation.input_splittings,
        generator_like_groomed_values=generator_like_jets_calculation.values,
        generator_like_groomed_indices=generator_like_jets_calculation.indices,
        measured_like_jets=measured_like_jets_calculation.input_jets,
        measured_like_splittings=measured_like_jets_calculation.input_splittings,
        measured_like_groomed_values=measured_like_jets_calculation.values,
        measured_like_groomed_indices=measured_like_jets_calculation.indices,
        match_using_distance=match_using_distance,
    )
    # Store leading, subleading matches
    # for label, matching in [("leading", leading_matching), ("subleading", subleading_matching)]:
    #    # We'll store the output in an array, and then store that in the overall output with a mask
    #    # We need the additional mask because we can't perform matching for every jet (single particle jets, etc).
    #    output = np.zeros(ak.num(generator_like_jets_calculation.input_jets), dtype=np.int)
    #    matching_output = np.zeros(len(matching.properly), dtype=np.int)
    #    matching_output[matching.properly] = 1
    #    matching_output[matching.mistag] = 2
    #    matching_output[matching.failed] = 3
    #    output[mask] = matching_output
    #    grooming_results[
    #        f"{grooming_method}_{measured_like_jets_label}_{generator_like_jets_label}_matching_{label}"
    #    ] = output

    for label, matching in [("leading", leading_matching), ("subleading", subleading_matching)]:
        grooming_results[
            f"{grooming_method}_{measured_like_jets_label}_{generator_like_jets_label}_matching_{label}"
        ] = matching

    return grooming_results


def prong_matching(
    measured_like_jets_calculation: Calculation,
    measured_like_jets_label: str,
    generator_like_jets_calculation: Calculation,
    generator_like_jets_label: str,
    grooming_method: str,
    match_using_distance: bool = True,
) -> Dict[str, np.ndarray]:
    """ Performs prong matching for the provided collections.

    Note:
        0 is there were insufficient constituents to form a splitting, 1 is properly matched, 2 is mistagged
        (leading -> subleading or subleading -> leading), 3 is untagged (failed).

    Args:
        measured_like_jets_calculation: Grooming calculation for measured-like jets (hybrid for hybrid-det level matching).
        measured_like_jets_label: Label for measured jets (hybrid for hybrid-det level matching).
        generator_like_jets_calculation: Grooming calculation for generator-like jets (det level for hybrid-det level matching).
        generator_like_jets_label: Label for generator jets (det_level for hybrid-det level matching).
        grooming_method: Name of the grooming method.
        match_using_distance: If True, match using distance. Otherwise, match using the stored label.
    Returns:
        Matching and subleading matching values.
    """
    # We can only perform matching if there are selected splittings.
    # Need to mask for calculations which have no indices (ie didn't find any that met criteria.
    mask = (generator_like_jets_calculation.indices.counts != 0) & (measured_like_jets_calculation.indices.counts != 0)
    try:
        masked_generator_like_jets_calculation = generator_like_jets_calculation[mask]
        masked_measured_like_jets_calculation = measured_like_jets_calculation[mask]
    except IndexError as e:
        logger.warning(e)
        IPython.start_ipython(user_ns=locals())

    # Matching
    grooming_results = {}
    logger.info(f"Performing {measured_like_jets_label}-{generator_like_jets_label} matching for {grooming_method}")
    leading_matching, subleading_matching = analyze_tree.determine_matched_jets(
        hybrid_inputs=analysis_objects.FillHistogramInput(
            jets=masked_measured_like_jets_calculation.input_jets,
            splittings=masked_measured_like_jets_calculation.input_splittings,
            values=masked_measured_like_jets_calculation.values,
            indices=masked_measured_like_jets_calculation.indices,
        ),
        matched_inputs=analysis_objects.FillHistogramInput(
            jets=masked_generator_like_jets_calculation.input_jets,
            splittings=masked_generator_like_jets_calculation.input_splittings,
            values=masked_generator_like_jets_calculation.values,
            indices=masked_generator_like_jets_calculation.indices,
        ),
        match_using_distance=match_using_distance,
    )
    # Store leading, subleading matches
    for label, matching in [("leading", leading_matching), ("subleading", subleading_matching)]:
        # We'll store the output in an array, and then store that in the overall output with a mask
        # We need the additional mask because we can't perform matching for every jet (single particle jets, etc).
        output = np.zeros(len(generator_like_jets_calculation.input_jets), dtype=np.int)
        matching_output = np.zeros(len(matching.properly), dtype=np.int)
        matching_output[matching.properly] = 1
        matching_output[matching.mistag] = 2
        matching_output[matching.failed] = 3
        output[mask] = matching_output
        grooming_results[
            f"{grooming_method}_{measured_like_jets_label}_{generator_like_jets_label}_matching_{label}"
        ] = output

    return grooming_results


def calculate_embedding_skim(  # noqa: C901
    input_filename: Path,
    iterative_splittings: bool,
    prefixes: Mapping[str, str],
    scale_factors: Sequence[float],
    train_directory: Path,
    jet_R: float,
    output_filename: Path,
    output_tree_name: str = "tree",
    create_friend_tree: bool = False,
    draw_example_splittings: bool = False,
) -> bool:
    """ Determine the response and prong matching for jets substructure techniques.

    Args:
        input_filename: Input file path.
        iterative_splittings: If True, we should only consider iterative splittings.
        create_friend_tree: Create a friend tree instead of the standard tree. It contains
            supplemental information. See the code for precisely what it contains. Default: False.
        draw_example_splittings: If True, draw a few interesting splitting graphs. Default: False.
    """
    # Validation
    # Setup
    # Use the train configuration to extract the train number and pt hard bin, which are used to get the scale factor.
    y = yaml.yaml()
    with open(train_directory / "config.yaml", "r") as f:
       train_config = y.load(f)
    train_number = train_config["number"]
    pt_hard_bin = train_config["pt_hard_bin"]
    logger.debug(f"Extracted train number: {train_number}, pt hard bin: {pt_hard_bin}")
    scale_factor = scale_factors[pt_hard_bin]

    # Actual setup.
    logger.info(f"Skimming tree from file {input_filename}")
    all_jets = new_methods.parquet_to_substructure_analysis(filename = input_filename, prefixes=list(prefixes.keys()))
    true_jets = all_jets["matched"]
    det_level_jets = all_jets["detLevel"]
    hybrid_jets = all_jets["data"]

    # Do the calculations
    # Do not mask on the number of constituents. This would prevent tagged <-> untagged migrations in the response.
    # mask = (
    #    (true_jets.constituents.counts > 1)
    #    & (det_level_jets.constituents.counts > 1)
    #    & (hybrid_jets.constituents.counts > 1)
    # )
    # Require that we have jets that aren't dominated by hybrid jets.
    # It's super important to be ">=". That allows the leading jet in the hybrid to be the same
    # as the leading jet in the true (which would be good - we've probably found the right jet).
    # NOTE: We already apply this cut at the analysis level, so it shouldn't really do anything here.
    #       We're just applying it again to be certain.
    # NOTE: As of 7 May 2020, we skip this cut at the analysis level, so it's super important to
    #       apply it here.
    # NOTE: As of 19 May 2019, we disable this cut event though it's not applied at the analysis level.
    #       This will allow L+L to study this at the analysis level.
    # mask = mask & (det_level_jets.constituents.max_pt >= hybrid_jets.constituents.max_pt)
    mask = hybrid_jets.jet_pt > 0

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
    if create_friend_tree:
        # Extract eta-phi of jets.
        output_filename = Path(str(output_filename.with_suffix("")) + "_friend.root")
        # As the skim is re-run, values are generally transitioned to the standard tree the next time it's generated.
    else:
        grooming_results["scale_factor"] = (true_jets.jet_pt[mask] * 0) + scale_factor
        # Add jet pt for all prefixes.
        grooming_results[f"{prefixes['matched']}_jet_pt"] = masked_true_jets.jet_pt
        grooming_results[f"{prefixes['detLevel']}_jet_pt"] = masked_det_level_jets.jet_pt
        grooming_results[f"{prefixes['data']}_jet_pt"] = masked_hybrid_jets.jet_pt
        # Add general jet properties.
        for prefix, jets in zip([(prefixes["matched"], masked_true_jets), (prefixes["detLevel"], masked_det_level_jets), (prefixes["data"], masked_hybrid_jets)]):
            # Jet eta phi
            # jet_four_vec = jets.jet_constituents.four_vectors().sum()
            # Since vector isn't ready yet, just do this by hand...
            constituents = jets.jet_constituents
            px = ak.sum(constituents.pt * np.cos(constituents.phi), axis=1)
            py = ak.sum(constituents.pt * np.sin(constituents.phi), axis=1)
            pz = ak.sum(constituents.pt * np.sinh(constituents.eta), axis=1)
            # Formulas just from inverting the above.
            grooming_results[f"{prefix}_jet_eta"] = np.arcsinh(pz / np.sqrt(px ** 2 + py ** 2))
            grooming_results[f"{prefix}_jet_phi"] = np.arctan2(py, px)

            # Leading track
            grooming_results[f"{prefix}_leading_track"] = ak.max(jets.jet_constituents.pt, axis=1)

        # Perform our calculations.
        functions = _define_calculation_functions(jet_R=jet_R, iterative_splittings=iterative_splittings)
        for func_name, func in functions.items():
            logger.debug(f"func_name: {func_name}")
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
                (prefixes["matched"], true_jets_calculation),
                (prefixes["detLevel"], det_level_jets_calculation),
                (prefixes["data"], hybrid_jets_calculation),
            ]:
                # Calculate splitting number for the appropriate cases.
                groomed_splittings = calculation.splittings
                # Number of splittings until the selected splitting, irrespective of the grooming conditions.
                n_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need all splitting indices (unrestricted by any possible grooming selections).
                    restricted_splittings_indices=calculation.input_splittings_indices,
                )
                logger.debug("Done with first splitting calculation")
                # Number of splittings which pass the grooming conditions until the selected splitting.
                n_groomed_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need the indices that correspond to the splittings that pass the grooming.
                    restricted_splittings_indices=calculation.possible_indices,
                    debug=False,
                )
                logger.debug("Done with second splitting calculation")

                # We pad with the UNFILLED_VALUE constant to account for any calculations that don't find a splitting.
                grooming_result = GroomingResultForTree(
                    grooming_method=func_name,
                    delta_R=ak.flatten(
                        ak.fill_none(ak.pad_none(groomed_splittings.delta_R, 1), new_methods.UNFILLED_VALUE)
                    ),
                    z=ak.flatten(ak.fill_none(ak.pad_none(groomed_splittings.z, 1), new_methods.UNFILLED_VALUE)),
                    kt=ak.flatten(ak.fill_none(ak.pad_none(groomed_splittings.kt, 1), new_methods.UNFILLED_VALUE)),
                    # All of the numbers are already flattened. 0 means untagged.
                    n_to_split=n_to_split,
                    n_groomed_to_split=n_groomed_to_split,
                    # Number of splittings which pass the grooming condition. For SoftDrop, this is n_sd.
                    n_passed_grooming=ak.num(calculation.possible_indices, axis=1),
                )
                grooming_results.update(grooming_result.asdict(prefix=prefix))

            logger.debug("Before prong matching")
            # IPython.embed()
            # Hybrid-det level matching.
            # We match using distance here because the labels don't align anymore due to the subtraction mixing the labels.
            hybrid_det_level_matching_results = prong_matching_numba_wrapper(
                measured_like_jets_calculation=hybrid_jets_calculation,
                measured_like_jets_label=prefixes["data"],
                generator_like_jets_calculation=det_level_jets_calculation,
                generator_like_jets_label=prefixes["detLevel"],
                grooming_method=func_name,
                match_using_distance=False,
            )
            grooming_results.update(hybrid_det_level_matching_results)
            logger.debug("Done with first prong matching")
            # Det level-true matching
            # We match using labels here because otherwise the reconstruction can cause the particles to move
            # enough that they may not match within a particular distance.
            det_level_true_matching_results = prong_matching_numba_wrapper(
                measured_like_jets_calculation=det_level_jets_calculation,
                measured_like_jets_label=prefixes["detLevel"],
                generator_like_jets_calculation=true_jets_calculation,
                generator_like_jets_label=prefixes["matched"],
                grooming_method=func_name,
                match_using_distance=False,
            )
            grooming_results.update(det_level_true_matching_results)
            logger.debug("Done with second prong matching")

            # Look for leading kt just because it's easier to understand conceptually.
            hybrid_det_level_leading_matching = grooming_results[f"{func_name}_{prefixes['data']}_{prefixes['detLevel']}_matching_leading"]
            hybrid_det_level_subleading_matching = grooming_results[f"{func_name}_{prefixes['data']}_{prefixes['detLevel']}_matching_subleading"]
            if (
                draw_example_splittings
                and func_name == "leading_kt"
                and ak.any((hybrid_det_level_leading_matching == 1) & (hybrid_det_level_subleading_matching == 3))
            ):
                from jet_substructure.analysis import draw_splitting

                # Find a sufficiently interesting jet (ie high enough pt)
                mask_jets_of_interest = (
                    (hybrid_det_level_leading_matching.properly & hybrid_det_level_subleading_matching.failed)
                    & (masked_hybrid_jets.jet_pt > 80)
                    & (det_level_jets_calculation.splittings.kt > 10).flatten()
                )

                # Look at most the first 5 jets.
                for i, hybrid_jet in enumerate(masked_hybrid_jets[mask_jets_of_interest][:5]):
                    # Find the hybrid jet and splitting of interest.
                    # hybrid_jet = masked_hybrid_jets[mask_jets_of_interest][0]
                    # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                    hybrid_jet_selected_splitting_index = hybrid_jets_calculation.indices[mask_jets_of_interest][i][0]
                    # Same for det level.
                    det_level_jet = masked_det_level_jets[mask_jets_of_interest][i]
                    # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                    det_level_jet_selected_splitting_index = det_level_jets_calculation.indices[mask_jets_of_interest][
                        i
                    ][0]

                    # Draw the splittings
                    draw_splitting.splittings_graph(
                        jet=hybrid_jet,
                        path=train_directory / "leading_correct_subleading_failed/",
                        filename=f"{i}_hybrid_splittings_jet_pt_{hybrid_jet.jet_pt:.1f}GeV_selected_splitting_index_{hybrid_jet_selected_splitting_index}",
                        show_subjet_pt=True,
                        selected_splitting_index=hybrid_jet_selected_splitting_index,
                    )
                    draw_splitting.splittings_graph(
                        jet=det_level_jet,
                        path=train_directory / "leading_correct_subleading_failed/",
                        filename=f"{i}_det_level_splittings_jet_pt_{det_level_jet.jet_pt:.1f}GeV_selected_splitting_index_{det_level_jet_selected_splitting_index}",
                        show_subjet_pt=True,
                        selected_splitting_index=det_level_jet_selected_splitting_index,
                    )

            logger.debug(f"Completed {func_name}")

    # Convert to numpy since we want to write to an output tree.
    grooming_results_np = {k: ak.to_numpy(v) for k, v in grooming_results.items()}
    # Write with uproot
    branches = {k: v.dtype for k, v in grooming_results_np.items()}
    logger.info(f"Writing embedding skim to {output_filename}")
    with uproot3.recreate(output_filename) as output_file:
        output_file[output_tree_name] = uproot3.newtree(branches)
        # Write all of the calculations
        output_file[output_tree_name].extend(grooming_results_np)

    logger.info(f"Finished processing tree from file {input_filename}")
    return True


if __name__ == "__main__":
    # import IPython; IPython.start_ipython(user_ns=locals())
    # TODO: Optimize the data sizes...
    #calculate_and_skim_embedding(True)
    ...
