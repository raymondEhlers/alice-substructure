#!/usr/bin/env python3

""" Analyze the dynamical grooming tree.

.. codeauthor:: Raymnod Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import argparse
import functools
import gzip
import logging
import pickle
import zlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

import attr
import awkward as ak
import enlighten
import IPython
import numpy as np
from pachyderm import binned_data, yaml
from pathos.multiprocessing import ProcessingPool as Pool

from jet_substructure.analysis import plot_results
from jet_substructure.base import analysis_objects, data_manager, helpers, substructure_methods
from jet_substructure.base.helpers import UprootArray


logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=analysis_objects.SubstructureHists)


@attr.s
class SubstructureResult:
    name: str = attr.ib()
    title: str = attr.ib()
    values: UprootArray[float] = attr.ib()
    indices: UprootArray[int] = attr.ib()
    subjet: substructure_methods.JetSplittingArray
    # TODO: Need to store iterative splitting information somehow!
    #       Perhaps just below...?

    @property
    def splitting_number(self) -> UprootArray[int]:
        try:
            return self._splitting_number
        except AttributeError:
            # +1 because splittings counts from 1, but indexing starts from 0.
            splitting_number = self.indices + 1
            # If there were no splittings, we want to set that to 0.
            splitting_number = splitting_number.pad(1).fillna(0)
            # Must flatten because the indices are still jagged.
            self._splitting_number: UprootArray[int] = splitting_number.flatten()

        return self._splitting_number

    @splitting_number.setter
    def splitting_number(self, value: UprootArray[int]) -> None:
        self._splitting_number = value


def setup_yaml() -> yaml.ruamel.yaml.YAML:
    return yaml.yaml(modules_to_register=[binned_data, analysis_objects, helpers])


def _convert_and_write_hists(
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]],
    tree_filename: Path,
    yaml_filename: Path,
    y: yaml.ruamel.yaml.YAML,
) -> Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]:
    # Convert to BinnedData and store the hists
    for h in hists.values():
        h.convert_boost_histograms_to_binned_data()
    with open(yaml_filename, "w") as f:
        logger.info(f"Writing hists of the tree {tree_filename} to {yaml_filename}")
        y.dump(hists, f)

    return hists


def _construct_jets_from_tree(prefix: str, tree: data_manager.Tree,) -> substructure_methods.SubstructureJetArray:
    """ Construct the substructure jet objects for data stored under a given prefix in a tree.

    Ideally, the object has already been created and stored. If not, it will be created and then
    stored in the tree for the future (where retrieving the created object from a file is far faster).

    Args:
        prefix: Prefix under which the data of interest is stored.
        tree: Tree where the data is stored.
    Returns:
        Constructed jet object.
    """
    constructed_name = f"{prefix}_constructed"
    if constructed_name in tree:
        logger.debug("Using fully constructed object")
        jets = cast(substructure_methods.SubstructureJetArray, tree[constructed_name])
        # Check whether the JaggedArrays which are stored in the file were constructed properly
        try:
            jets.constituents.max_pt
        except AttributeError:
            jets = substructure_methods.SubstructureJetArray._from_serialization(
                jet_pt=jets.jet_pt,
                jet_constituents=jets.constituents,
                subjets=jets.subjets,
                jet_splittings=jets.splittings,
            )
    else:
        logger.debug("Constructing object")
        jets = substructure_methods.SubstructureJetArray.from_tree(tree, prefix=prefix)

        # Save calculate columns so we don't need to re-calculate them every time.
        # NOTE: We always check if they already exist because HDF5 doesn't like us
        #       overwriting columns.
        # Calculated subjet constituents.
        name = f"{prefix}.fSubjets.constituents"
        if name not in tree:
            tree[name] = jets.subjets.constituents

        # Store the full treee in h5.
        # This provides a huge speed up in terms of processing speed!
        if constructed_name not in tree:
            tree[constructed_name] = jets

    # Flush the hdf5 portion of the tree to ensure that it's been written properly.
    # Otherwise, the file may end up corrupted.
    tree._hdf5_tree.flush()

    return jets


def _define_calculation_funcs(
    dataset: analysis_objects.Dataset,
) -> Tuple[
    functools.partial[Tuple[UprootArray[float], UprootArray[int]]],
    functools.partial[Tuple[UprootArray[float], UprootArray[int]]],
    functools.partial[Tuple[UprootArray[float], UprootArray[int]]],
    functools.partial[Tuple[UprootArray[float], UprootArray[int]]],
    functools.partial[Tuple[UprootArray[float], UprootArray[int]]],
]:
    dynamical_z_func = functools.partial(substructure_methods.JetSplittingArray.dynamical_z, R=dataset.settings.jet_R)
    dynamical_kt_func = functools.partial(substructure_methods.JetSplittingArray.dynamical_kt, R=dataset.settings.jet_R)
    dynamical_time_func = functools.partial(
        substructure_methods.JetSplittingArray.dynamical_time, R=dataset.settings.jet_R
    )
    leading_kt_func = functools.partial(substructure_methods.JetSplittingArray.leading_kt,)
    leading_kt_hard_cutoff_func = functools.partial(
        substructure_methods.JetSplittingArray.leading_kt, z_cutoff=dataset.settings.z_cutoff
    )
    return dynamical_z_func, dynamical_kt_func, dynamical_time_func, leading_kt_func, leading_kt_hard_cutoff_func


def analyze_single_tree(
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    hists_filename_stem: str,
    force_reprocessing: bool = False,
) -> Tuple[
    Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]],
]:
    # Setup
    logger.info(f"Processing tree from file {tree.filename}")
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]] = {}
    # If the output file already exist, skip processing the tree and just return the hists instead (which is way faster!)
    train_number = tree.filename.parent.name
    pkl_filename = dataset.output / f"{train_number}_{tree.filename.stem}_{hists_filename_stem}.pgz"
    if pkl_filename.exists() and not force_reprocessing:
        logger.info(f"Skipping processing of tree {tree.filename} by loading data from stored hists.")
        with gzip.GzipFile(pkl_filename, "r") as pkl_file:
            hists = pickle.load(pkl_file)  # type: ignore
            return (hists,)

    # Since we're actually processing, we setup the output hists
    for iterative_splittings in [False, True]:
        for jet_pt_bin in jet_pt_bins:
            hists[
                analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
            ] = analysis_objects.create_substructure_hists(
                iterative_splittings=iterative_splittings, z_cutoff=dataset.settings.z_cutoff
            )

    # Add a convenient wrapper.
    logger.debug(f"Accessing data from the tree {tree.filename}.")
    successfully_accessed_data = False
    try:
        # If there are 0 entries, then just return - it won't work...
        if len(tree) > 0:
            logger.debug("Constructing data jets")
            jets = _construct_jets_from_tree(prefix="data", tree=tree)
            successfully_accessed_data = True
        else:
            logger.warning(f"No jets are in file {tree.filename}. Skipping")
    except zlib.error as e:
        logger.warning(f"Issue reading the data: {e}. Skipping")

    # Catch all failed cases.
    if not successfully_accessed_data:
        # Return the empty hists. We can't process this data :-(
        return (hists,)

    # Sanity check using iterative splittings information stored with the splittings
    # This is used as a local testing cross check. We don't want to include it with the standard output
    # because it will increase the output size with redundant information
    try:
        iterative_splittings_mask = tree["data.fJetSplittings.fIterativeSplitting"]
        logger.debug("Checking iteartive splittings are calculated correctly.")
        # The jet_pt_mask is just a hack for selecting everything.
        _, temp_iterative_splittings = _select_and_retrieve_splittings(
            jets, jet_pt_mask=np.ones_like(jets) > 0, iterative_splittings=True,
        )
        assert (jets.splittings[iterative_splittings_mask] == temp_iterative_splittings).all().all()
    except KeyError:
        ...

    # Loop over iterations (jet pt ranges, iterative splitting)
    progress_manager = enlighten.get_manager()
    with progress_manager.counter(
        total=len(hists), desc="Analyzing", unit="variation", leave=False
    ) as selections_counter:
        for identifier, h in selections_counter(hists.items()):
            restricted_jets, splittings = _select_and_retrieve_splittings(
                jets,
                jet_pt_mask=identifier.jet_pt_bin.mask_array(jets.jet_pt),
                iterative_splittings=identifier.iterative_splittings,
            )

            # Fill the hists as appropriate
            # Inclusive
            inputs = analysis_objects.FillHistogramInput(
                restricted_jets,
                splittings,
                # Fake the calculations, taking all values, and not masking
                # anything out.
                splittings.kt.ones_like().flatten(),
                splittings.localindex,
            )
            hists[identifier].inclusive.fill(inputs, jet_R=dataset.settings.jet_R)
            # Dynamical z
            inputs = analysis_objects.FillHistogramInput(
                restricted_jets, splittings, *splittings.dynamical_z(R=dataset.settings.jet_R)
            )
            hists[identifier].dynamical_z.fill(inputs, jet_R=dataset.settings.jet_R)
            # Dynamical kt
            inputs = analysis_objects.FillHistogramInput(
                restricted_jets, splittings, *splittings.dynamical_kt(R=dataset.settings.jet_R)
            )
            hists[identifier].dynamical_kt.fill(inputs, jet_R=dataset.settings.jet_R)
            # Dynamical time
            inputs = analysis_objects.FillHistogramInput(
                restricted_jets, splittings, *splittings.dynamical_time(R=dataset.settings.jet_R)
            )
            hists[identifier].dynamical_time.fill(inputs, jet_R=dataset.settings.jet_R)
            # Leading kt
            inputs = analysis_objects.FillHistogramInput(restricted_jets, splittings, *splittings.leading_kt())
            hists[identifier].leading_kt.fill(inputs, jet_R=dataset.settings.jet_R)
            # Leading kt with z cutoff
            inputs = analysis_objects.FillHistogramInput(
                restricted_jets, splittings, *splittings.leading_kt(z_cutoff=dataset.settings.z_cutoff)
            )
            hists[identifier].leading_kt_hard_cutoff.fill(inputs, jet_R=dataset.settings.jet_R)
            # import numpy as np
            # if (np.log(1.0 / splittings[indices].delta_R.flatten()) < 2).any() and (np.log(splittings[indices].kt.flatten()) < -1).any():
            #    logger.warning("Maybe the z cut isn't working??")
            #    import IPython; IPython.embed()

    # Store hists with pickle because it takes too longer otherwise (and for consistency).
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        pickle.dump(hists, pkl_file)  # type: ignore

    return (hists,)


def analyze_single_tree_toy(
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    hists_filename_stem: str,
    force_reprocessing: bool,
    **kwargs: str,
) -> Tuple[
    Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureToyHists]],
]:
    logger.info(f"Processing tree from file {tree.filename}")

    data_prefix: str = kwargs["data_prefix"]
    # Validation
    if data_prefix == "hybrid":
        data_prefix = "data"
    # Setup
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureToyHists]] = {}
    # If the output file already exist, skip processing the tree and just return the hists instead (which is way faster!)
    train_number = tree.filename.parent.name
    pkl_filename = dataset.output / f"{train_number}_{tree.filename.stem}_{hists_filename_stem}.pgz"
    if pkl_filename.exists() and not force_reprocessing:
        logger.info(f"Skipping processing of tree {tree.filename} by loading data from stored hists.")
        with gzip.GzipFile(pkl_filename, "r") as pkl_file:
            hists = pickle.load(pkl_file)  # type: ignore
            return (hists,)

    # Since we're actually processing, we setup the output hists
    for iterative_splittings in [False, True]:
        for jet_pt_bin in jet_pt_bins:
            hists[
                analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
            ] = analysis_objects.create_substructure_toy_hists(
                iterative_splittings=iterative_splittings, z_cutoff=dataset.settings.z_cutoff
            )

    # Add a convenient wrapper.
    logger.debug(f"Accessing data from the tree {tree.filename}.")
    successfully_accessed_data = False
    try:
        # If there are 0 entries, then just return - it won't work...
        if len(tree) > 0:
            logger.debug(f"Constructing {data_prefix} jets")
            data_jets = _construct_jets_from_tree(prefix=data_prefix, tree=tree)
            logger.debug("Constructing true jets")
            true_jets = _construct_jets_from_tree(prefix="true", tree=tree)
            successfully_accessed_data = True
        else:
            logger.warning(f"No jets are in file {tree.filename}. Skipping")
    except zlib.error as e:
        logger.warning(f"Issue reading the data: {e}. Skipping")

    # Catch all failed cases.
    if not successfully_accessed_data:
        # Convert, write, and return the empty hists. We can't process this data :-(
        return (hists,)

    # Loop over iterations (jet pt ranges, iterative splitting)
    progress_manager = enlighten.get_manager()
    with progress_manager.counter(
        total=len(hists), desc="Analyzing", unit="variation", leave=False
    ) as selections_counter:
        for identifier, h in selections_counter(hists.items()):
            # We want to restrict a constant hybrid jet pt range for both true and hybrid.
            # This will allow us to compare to measured jet pt ranges.
            jet_pt_mask = identifier.jet_pt_bin.mask_array(data_jets.jet_pt)
            # Add additional restrictions that we can't handle single constituent jets.
            # TODO: Can we do better???
            jet_pt_mask = jet_pt_mask & (data_jets.subjets.counts > 2)
            restricted_data_jets, restricted_data_jets_splittings = _select_and_retrieve_splittings(
                data_jets, jet_pt_mask, identifier.iterative_splittings
            )
            restricted_true_jets, restricted_true_jets_splittings = _select_and_retrieve_splittings(
                true_jets, jet_pt_mask, identifier.iterative_splittings
            )

            # TODO: What about additional cuts? Pt hard? etc
            weight = 1.0

            # Fill the hists as appropriate
            # TODO: Inclusive
            # Dynamical z
            data_inputs = analysis_objects.FillHistogramInput(
                restricted_data_jets,
                restricted_data_jets_splittings,
                *restricted_data_jets_splittings.dynamical_z(R=dataset.settings.jet_R),
            )
            # TODO: We absolutely shouldn't be calculating the splitting properties here!
            # TODO: If we take the leading, we already know that it was only one splitting, and we already
            # TODO: know the values...
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.dynamical_z(R=dataset.settings.jet_R),
            )
            hists[identifier].dynamical_z.fill(
                data_inputs=data_inputs, true_inputs=true_inputs, jet_R=dataset.settings.jet_R, weight=weight,
            )
            # Dynamical kt
            data_inputs = analysis_objects.FillHistogramInput(
                restricted_data_jets,
                restricted_data_jets_splittings,
                *restricted_data_jets_splittings.dynamical_kt(R=dataset.settings.jet_R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.dynamical_kt(R=dataset.settings.jet_R),
            )
            hists[identifier].dynamical_kt.fill(
                data_inputs=data_inputs, true_inputs=true_inputs, jet_R=dataset.settings.jet_R, weight=weight,
            )
            # Dynamical time
            data_inputs = analysis_objects.FillHistogramInput(
                restricted_data_jets,
                restricted_data_jets_splittings,
                *restricted_data_jets_splittings.dynamical_time(R=dataset.settings.jet_R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.dynamical_time(R=dataset.settings.jet_R),
            )
            hists[identifier].dynamical_time.fill(
                data_inputs=data_inputs, true_inputs=true_inputs, jet_R=dataset.settings.jet_R, weight=weight,
            )
            # Leading kt
            data_inputs = analysis_objects.FillHistogramInput(
                restricted_data_jets, restricted_data_jets_splittings, *restricted_data_jets_splittings.leading_kt(),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets, restricted_true_jets_splittings, *restricted_true_jets_splittings.leading_kt()
            )
            hists[identifier].leading_kt.fill(
                data_inputs=data_inputs, true_inputs=true_inputs, jet_R=dataset.settings.jet_R, weight=weight,
            )
            # Leading kt with z cutoff
            data_inputs = analysis_objects.FillHistogramInput(
                restricted_data_jets,
                restricted_data_jets_splittings,
                *restricted_data_jets_splittings.leading_kt(z_cutoff=dataset.settings.z_cutoff),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.leading_kt(z_cutoff=dataset.settings.z_cutoff),
            )
            # Ensure that there are sufficient values!
            # TODO: This doesn't work because the true frequently fails. They somehow need to be the same length...
            # IPython.embed()
            # hists[identifier].leading_kt.fill(
            #    data_inputs=data_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            # )

    # Store hists with pickle because it takes too longer otherwise.
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        pickle.dump(hists, pkl_file)  # type: ignore

    return (hists,)


def _select_and_retrieve_splittings(
    jets: substructure_methods.SubstructureJetArray, jet_pt_mask: UprootArray[bool], iterative_splittings: bool
) -> Tuple[substructure_methods.SubstructureJetArray, substructure_methods.JetSplittingArray]:
    # Ensure that there are sufficient counts
    restricted_jets = jets[jet_pt_mask]
    if iterative_splittings:
        # Only keep iterative splittings.
        splittings = restricted_jets.splittings.iterative_splittings(restricted_jets.subjets)

        # Enable this test to determine if we've selected different sets of splittings with the
        # recursive vs iterative selections.
        # if (splittings.counts != restricted_jets.splittings.counts).any():
        #    logger.warning("Disagreement between number of inclusive and recursive splittings (as expected!)")
        #    IPython.embed()
    else:
        splittings = restricted_jets.splittings

    return restricted_jets, splittings


def _subjets_contributing_to_splittings(
    inputs: analysis_objects.FillHistogramInput,
) -> substructure_methods.SubjetArray:
    """ Determine which subjets contribute to the selected splitting.

    We do this by looking for subjets with a a parent splitting index that is equal to the selected index.

    Args:
        inputs: Jets and splittings selected by a particular algorithm.
    Returns:
        Subjets which contributed to the selected splittings. There will always be 2 subjets by definition.
    """
    # In order to compare to the subjets directly, we need to expand the indices to the same dimension as the subjets.
    selected_indices_mask = inputs.jets.subjets.parent_splitting_index.ones_like() * inputs.indices.flatten()
    matched_subjets_unsorted = inputs.jets.subjets[selected_indices_mask == inputs.jets.subjets.parent_splitting_index]
    return cast(substructure_methods.SubjetArray, matched_subjets_unsorted)


def _get_leading_and_subleading_subjets(
    subjets_unsorted: substructure_methods.SubjetArray,
) -> Tuple[substructure_methods.SubjetArray, substructure_methods.SubjetArray]:
    """ Determine the leading and subleading subjets based on the sum of subjet constituents pt.

    Args:
        subjets_unsorted: Unsorted subjets of a given splitting. There are two subjets by definition.
    Returns:
        Leading subjets, subleading subjets.
    """
    # Sort the subjets such that 0 is always the leading subjet.
    # Coerces the bool into an integer by taking 1 - array.
    # The leading subjet will be have a 0, while the subleading will have a 1.
    # NOTE: We actually want to add the four vectors rather than just summing the constituent pt. It doesn't
    #       have a huge impact, but it's the right way to do it.
    unsorted_subjet_pt = subjets_unsorted.constituents.four_vectors().sum().pt
    subjets_pt_comparison = 1 - (unsorted_subjet_pt[:, 0] > unsorted_subjet_pt[:, 1])
    # For each subjet_pt_comparison, we want to take the index of the leading subjet and use that to extract the leading subjet.
    leading_indices = ak.JaggedArray.fromoffsets(range(len(subjets_pt_comparison) + 1), subjets_pt_comparison)
    subjets_leading = subjets_unsorted[leading_indices].flatten()
    # Same idea for the subleading subjet (which is necessarily 1 - subjets_pt_comparison because there are only two subjets.
    subleading_indices = ak.JaggedArray.fromoffsets(range(len(subjets_pt_comparison) + 1), 1 - subjets_pt_comparison)
    subjets_subleading = subjets_unsorted[subleading_indices].flatten()

    return subjets_leading, subjets_subleading


def _split_array(
    a: substructure_methods.SubjetArray, n: int
) -> Iterable[Tuple[substructure_methods.SubjetArray, slice]]:
    """ Split an array into n chunks.

    Currently the typing suggests that it will only work for SubjetArray, but it should work for any array.

    From: https://stackoverflow.com/a/2135920/12907985
    """
    k, m = divmod(len(a), n)
    return (
        (
            a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)],  # noqa: E203 . Conflicts with black...
            slice(i * k + min(i, m), (i + 1) * k + min(i + 1, m)),
        )
        for i in range(n)
    )


def _determine_matching_types(
    matched_subjets: substructure_methods.SubjetArray, hybrid_subjets: substructure_methods.SubjetArray,
) -> UprootArray[bool]:
    """ Determine whether the given subjets match.

    Args:
        matched_subjets: Subjets from the matched jets.
        hybrid_subjets: Subjets from the hybrid jets.
    Returns:
        Mask indicating when these subjets matched.
    """
    # We split the array into chunks to keep memory usage to a more reasonable level.
    number_of_chunks = 6

    shared_constituents_pts = np.zeros(len(matched_subjets))
    # Can't use np.array_split (even though it would be nice and probably better tested) because the output ends up as
    # numpy array, and in this case, we don't want such a conversion.
    for (matched_subset, selected_range), (hybrid_subset, _) in zip(
        _split_array(matched_subjets.constituents, number_of_chunks),
        _split_array(hybrid_subjets.constituents, number_of_chunks),
    ):
        constituent_pairs = matched_subset.argcross(hybrid_subset)
        matched_leading_indices, hybrid_leading_indices = constituent_pairs.unzip()

        index_matching = (
            matched_subset[matched_leading_indices].global_index == hybrid_subset[hybrid_leading_indices].global_index
        )

        shared_constituents_pts[selected_range] = matched_subset[matched_leading_indices][index_matching].pt.sum()

    # Sanity check
    if (shared_constituents_pts > matched_subjets.constituents.four_vectors().sum().pt).any():
        logger.warning("Constituent pts are greater than the subjet pts...")
        IPython.embed()
        raise ValueError("Constituent pts are greater than the subjet pts...")

    matched = (shared_constituents_pts / matched_subjets.constituents.four_vectors().sum().pt) > 0.5
    return cast(UprootArray[bool], matched)


def determine_matched_jets(
    hybrid_inputs: analysis_objects.FillHistogramInput, matched_inputs: analysis_objects.FillHistogramInput
) -> Tuple[analysis_objects.MatchingResult, analysis_objects.MatchingResult]:
    """ Determine the matching between subjets.

    The passed jets need to have the selected indices already applied.
    We need to work with the indices applied to these jets.

    Args:
        hybrid_inputs: The selected hybrid jets and splittings.
        matched_inputs: The selected matched jets and splittings.
    Returns:
        Leading subjet matching results, subleading subjet matching results.
    """
    # Setup
    # delta = 0.001
    # Determine which subjets contribute to the selected splitting.
    matched_subjets_unsorted = _subjets_contributing_to_splittings(inputs=matched_inputs)
    hybrid_subjets_unsorted = _subjets_contributing_to_splittings(inputs=hybrid_inputs)

    # Sort the subjets such that 0 is always the leading subjet.
    matched_subjets_leading, matched_subjets_subleading = _get_leading_and_subleading_subjets(matched_subjets_unsorted)
    hybrid_subjets_leading, hybrid_subjets_subleading = _get_leading_and_subleading_subjets(hybrid_subjets_unsorted)

    # Now, determine the matching types based on the possible combinations of leading and subleading subjets.
    matched_leading_properly = _determine_matching_types(matched_subjets_leading, hybrid_subjets_leading)
    matched_leading_mistag = _determine_matching_types(matched_subjets_leading, hybrid_subjets_subleading)
    matched_subleading_properly = _determine_matching_types(matched_subjets_subleading, hybrid_subjets_subleading)
    matched_subleading_mistag = _determine_matching_types(matched_subjets_subleading, hybrid_subjets_leading)
    # Combine those cases to determine when the we failed to find the leading and subleading subjets.
    matched_leading_failed = ~matched_leading_properly & ~matched_leading_mistag
    matched_subleading_failed = ~matched_subleading_properly & ~matched_subleading_mistag

    return (
        analysis_objects.MatchingResult(matched_leading_properly, matched_leading_mistag, matched_leading_failed),
        analysis_objects.MatchingResult(
            matched_subleading_properly, matched_subleading_mistag, matched_subleading_failed
        ),
    )


def _fill_embedded_hists_with_calculation(
    calculation: functools.partial[Tuple[UprootArray[float], UprootArray[int]]],
    fill_attr_name: str,
    restricted_hybrid_jets: substructure_methods.SubstructureJetArray,
    restricted_hybrid_jets_splittings: substructure_methods.JetSplittingArray,
    restricted_true_jets: substructure_methods.SubstructureJetArray,
    restricted_true_jets_splittings: substructure_methods.JetSplittingArray,
    true_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    hybrid_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    response_hists: analysis_objects.Hists[analysis_objects.SubstructureResponseHists],
    jet_R: float,
    weight: float,
) -> None:
    # Calculate the inputs
    hybrid_inputs = analysis_objects.FillHistogramInput(
        restricted_hybrid_jets, restricted_hybrid_jets_splittings, *calculation(restricted_hybrid_jets_splittings),
    )
    true_inputs = analysis_objects.FillHistogramInput(
        restricted_true_jets, restricted_true_jets_splittings, *calculation(restricted_true_jets_splittings),
    )
    # And fill the results.
    getattr(true_hists, fill_attr_name).fill(
        inputs=true_inputs, jet_R=jet_R, weight=weight,
    )
    getattr(hybrid_hists, fill_attr_name).fill(
        inputs=hybrid_inputs, jet_R=jet_R, weight=weight,
    )
    getattr(response_hists, fill_attr_name).fill(
        hybrid_inputs=hybrid_inputs, true_inputs=true_inputs, jet_R=jet_R, weight=weight,
    )


def analyze_single_tree_embedding(  # noqa: C901
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    hists_filename_stem: str,
    force_reprocessing: bool = False,
    scale_n_jets_when_loading_hists: bool = False,
) -> Tuple[
    Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]],
    Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]],
    Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]],
    Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]],
]:
    """ Determine the response and prong matching for jets substructure techniques.

    Why combine them together? Because then we only have to open and process a tree once.
    At a future date (beyond the start of April 2020), it would be better to refactor them more separately,
    such that we can enable or disable the different options and still have appropriate return values.
    But for now, we don't worry about it.
    """
    # Setup
    logger.info(f"Processing tree from file {tree.filename}")
    # Help out mypy...
    assert isinstance(dataset.settings, analysis_objects.PtHardAnalysisSettings)
    true_hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]] = {}
    hybrid_hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]] = {}
    response_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]
    ] = {}
    matching_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
    ] = {}
    # Determine scale factor
    # NOTE: This relies on the train_number being up a directory!
    train_number = tree.filename.parent.name
    pt_hard_bin = dataset.settings.train_number_to_pt_hard_bin[int(train_number)]
    scale_factor = dataset.settings.scale_factors[pt_hard_bin]

    # If the output file already exist, skip processing the tree and just return the hists instead (which is way faster!)
    pkl_filename = dataset.output / f"{train_number}_{tree.filename.stem}_{hists_filename_stem}.pgz"
    if pkl_filename.exists() and not force_reprocessing:
        logger.info(f"Skipping processing of tree {tree.filename} by loading data from stored hists.")
        with gzip.GzipFile(pkl_filename, "r") as pkl_file:
            true_hists, hybrid_hists, response_hists, matching_hists = pickle.load(pkl_file)  # type: ignore
            # NOTE: This is transient for loading files this way. However, it won't be transient if we load
            #       for just plotting (as it will be saved in the merge hists)
            if scale_n_jets_when_loading_hists:
                logger.warning(
                    "Rescaling n_jets by scale factor because it was forgotten during processing. This is transient for the individual files, but not for the merged!"
                )
                for hists in true_hists.values():
                    for technique, technique_hists in hists:
                        technique_hists.n_jets *= scale_factor
                for hists in hybrid_hists.values():
                    for technique, technique_hists in hists:
                        technique_hists.n_jets *= scale_factor
            return true_hists, hybrid_hists, response_hists, matching_hists

    # Since we're actually processing, we setup the output hists
    for iterative_splittings in [False, True]:
        for jet_pt_bin in jet_pt_bins:
            # True hists
            true_hists[
                analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
            ] = analysis_objects.create_substructure_hists(
                iterative_splittings=iterative_splittings, z_cutoff=dataset.settings.z_cutoff
            )
            # Hybrid hists
            hybrid_hists[
                analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
            ] = analysis_objects.create_substructure_hists(
                iterative_splittings=iterative_splittings, z_cutoff=dataset.settings.z_cutoff
            )
            # Responses
            response_hists[
                analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
            ] = analysis_objects.create_substructure_response_hists(
                iterative_splittings=iterative_splittings, z_cutoff=dataset.settings.z_cutoff
            )
        # Matching. We're going to plot again jet pt, so we don't want to select on jet pt bins here.
        matching_hists[
            analysis_objects.Identifier(iterative_splittings, jet_pt_bin=helpers.RangeSelector(0, 150))
        ] = analysis_objects.create_matching_hists(
            iterative_splittings=iterative_splittings, z_cutoff=dataset.settings.z_cutoff
        )

    # Add a convenient wrapper.
    logger.debug(f"Accessing data from the tree {tree.filename}.")
    successfully_accessed_data = False
    try:
        # If there are 0 entries, then just return - it won't work...
        if len(tree) > 0:
            logger.debug("Constructing hybrid jets")
            hybrid_jets = _construct_jets_from_tree(prefix="data", tree=tree)
            logger.debug("Constructing pythia true (matched) jets")
            true_jets = _construct_jets_from_tree(prefix="matched", tree=tree)
            logger.debug("Constructing pythia det level jets")
            det_level_jets = _construct_jets_from_tree(prefix="detLevel", tree=tree)
            successfully_accessed_data = True
        else:
            logger.warning(f"No jets are in file {tree.filename}. Skipping")
    except zlib.error as e:
        logger.warning(f"Issue reading the data: {e}. Skipping")

    # Catch all failed cases.
    if not successfully_accessed_data:
        # Return the empty hists. We can't process this data :-(
        return true_hists, hybrid_hists, response_hists, matching_hists

    # Define calculation functions
    (
        dynamical_z_func,
        dynamical_kt_func,
        dynamical_time_func,
        leading_kt_func,
        leading_kt_hard_cutoff_func,
    ) = _define_calculation_funcs(dataset)

    # Loop over iterations (jet pt ranges, iterative splitting)
    progress_manager = enlighten.get_manager()
    with progress_manager.counter(
        total=len(response_hists), desc="Analyzing", unit="variation", leave=False
    ) as selections_counter:
        for identifier, h in selections_counter(response_hists.items()):
            # We want to restrict a constant hybrid jet pt range for both true and hybrid.
            # This will allow us to compare to measured jet pt ranges.
            jet_pt_mask = identifier.jet_pt_bin.mask_array(hybrid_jets.jet_pt)
            # Ensure that we don't have single track jets because the splitting won't be defined for that case.
            # No actual jet pt range restrictions.
            jet_pt_mask = jet_pt_mask & (hybrid_jets.constituents.counts > 1) & (true_jets.constituents.counts > 1)
            # Require that we have jets that aren't dominated by hybrid jets.
            # It's super important to be ">=". That allows the leading jet in the hybrid to be the same
            # as the leading jet in the true (which would be good - we've probably found the right jet).
            jet_pt_mask = jet_pt_mask & (true_jets.constituents.max_pt >= hybrid_jets.constituents.max_pt)

            # Then restrict our jets.
            restricted_hybrid_jets, restricted_hybrid_jets_splittings = _select_and_retrieve_splittings(
                hybrid_jets, jet_pt_mask, identifier.iterative_splittings
            )
            restricted_true_jets, restricted_true_jets_splittings = _select_and_retrieve_splittings(
                true_jets, jet_pt_mask, identifier.iterative_splittings
            )

            # Scale factor to account for pt hard bin.
            weight = scale_factor

            # Fill the hists as appropriate
            # TODO: Inclusive
            for func, attr_name in [
                (dynamical_z_func, "dynamical_z"),
                (dynamical_kt_func, "dynamical_kt"),
                (dynamical_time_func, "dynamical_time"),
                (leading_kt_func, "leading_kt"),
                (leading_kt_hard_cutoff_func, "leading_kt_hard_cutoff"),
            ]:
                _fill_embedded_hists_with_calculation(
                    calculation=func,
                    fill_attr_name=attr_name,
                    restricted_hybrid_jets=restricted_hybrid_jets,
                    restricted_hybrid_jets_splittings=restricted_hybrid_jets_splittings,
                    restricted_true_jets=restricted_true_jets,
                    restricted_true_jets_splittings=restricted_true_jets_splittings,
                    true_hists=true_hists[identifier],
                    hybrid_hists=hybrid_hists[identifier],
                    response_hists=response_hists[identifier],
                    jet_R=dataset.settings.jet_R,
                    weight=weight,
                )

    # Store the hists
    # Store hists with pickle because it takes too longer otherwise.
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        pickle.dump((true_hists, hybrid_hists, response_hists, matching_hists), pkl_file)  # type: ignore

    # Look at matched jets
    matching_hists = matching(
        matching_hists=matching_hists,
        matched_jets=det_level_jets,
        hybrid_jets=hybrid_jets,
        dataset=dataset,
        scale_factor=scale_factor,
        progress_manager=progress_manager,
    )

    # Store hists with pickle because it takes too longer otherwise.
    # Write again here (despite the waste of writing twice) so we can keep the response hists even
    # if the matching fails
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        pickle.dump((true_hists, hybrid_hists, response_hists, matching_hists), pkl_file)  # type: ignore

    return true_hists, hybrid_hists, response_hists, matching_hists


def _fill_matching_hists_with_calculation(
    calculation: functools.partial[Tuple[UprootArray[float], UprootArray[int]]],
    fill_attr_name: str,
    restricted_hybrid_jets: substructure_methods.SubstructureJetArray,
    restricted_hybrid_jets_splittings: substructure_methods.JetSplittingArray,
    restricted_matched_jets: substructure_methods.SubstructureJetArray,
    restricted_matched_jets_splittings: substructure_methods.JetSplittingArray,
    matching_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
    ],
    identifier: analysis_objects.Identifier,
    weight: float,
) -> None:
    # Calculate the inputs
    hybrid_inputs = analysis_objects.FillHistogramInput(
        restricted_hybrid_jets, restricted_hybrid_jets_splittings, *calculation(restricted_hybrid_jets_splittings),
    )
    matched_inputs = analysis_objects.FillHistogramInput(
        restricted_matched_jets, restricted_matched_jets_splittings, *calculation(restricted_matched_jets_splittings),
    )
    leading_matching, subleading_matching = determine_matched_jets(hybrid_inputs, matched_inputs)
    # And fill the results.
    temp_identifier = analysis_objects.Identifier(
        iterative_splittings=identifier.iterative_splittings, jet_pt_bin=identifier.jet_pt_bin,
    )
    getattr(matching_hists[temp_identifier], fill_attr_name).fill(
        matched_inputs=matched_inputs,
        hybrid_inputs=hybrid_inputs,
        leading=leading_matching,
        subleading=subleading_matching,
        weight=weight,
    )


def matching(
    matching_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
    ],
    matched_jets: substructure_methods.SubstructureJetArray,
    hybrid_jets: substructure_methods.SubstructureJetArray,
    dataset: analysis_objects.Dataset,
    scale_factor: float,
    progress_manager: enlighten.Manager,
) -> Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]]:
    """ Determine the prong matching for jets substructure techniques.

    """
    # Setup
    # Define calculation functions
    (
        dynamical_z_func,
        dynamical_kt_func,
        dynamical_time_func,
        leading_kt_func,
        leading_kt_hard_cutoff_func,
    ) = _define_calculation_funcs(dataset)

    # Actually perform the matching
    logger.info("Starting matching")
    number_of_grooming_methods = 4
    with progress_manager.counter(
        total=len(matching_hists) * number_of_grooming_methods, desc="Analyzing", unit="variation", leave=False
    ) as selections_counter:
        for identifier, h in selections_counter(matching_hists.items()):
            # Ensure that we don't have single track jets because the splitting won't be defined for that case.
            # No actual jet pt range restrictions.
            mask = (hybrid_jets.constituents.counts > 1) & (matched_jets.constituents.counts > 1)
            # Require that we have jets that aren't dominated by hybrid jets.
            # It's super important to be ">=". That allows the leading jet in the hybrid to be the same
            # as the leading jet in the true (which would be good - we've probably found the right jet).
            mask = mask & (matched_jets.constituents.max_pt >= hybrid_jets.constituents.max_pt)

            restricted_hybrid_jets, restricted_hybrid_jets_splittings = _select_and_retrieve_splittings(
                hybrid_jets, mask, identifier.iterative_splittings
            )
            restricted_matched_jets, restricted_matched_jets_splittings = _select_and_retrieve_splittings(
                matched_jets, mask, identifier.iterative_splittings
            )

            # Scale factor to account for pt hard bin.
            weight = scale_factor

            # Fill the hists as appropriate
            # TODO: Inclusive
            # TODO: SD
            for func, attr_name in [
                (dynamical_z_func, "dynamical_z"),
                (dynamical_kt_func, "dynamical_kt"),
                (dynamical_time_func, "dynamical_time"),
                (leading_kt_func, "leading_kt"),
            ]:
                _fill_matching_hists_with_calculation(
                    calculation=func,
                    fill_attr_name=attr_name,
                    restricted_hybrid_jets=restricted_hybrid_jets,
                    restricted_hybrid_jets_splittings=restricted_hybrid_jets_splittings,
                    restricted_matched_jets=restricted_matched_jets,
                    restricted_matched_jets_splittings=restricted_matched_jets_splittings,
                    matching_hists=matching_hists,
                    identifier=identifier,
                    weight=weight,
                )
                selections_counter.update()

    return matching_hists


def _wrap_multiprocessing(
    tree: Callable[[], data_manager.Tree],
    analysis_function: Callable[
        [data_manager.Tree],
        Sequence[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]],
    ],
) -> Sequence[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]]:
    """ Wrap analysis function to instantiate the fully lazy tree.

    To be used in conjunction with multiprocessing (which is why we need to delay instantiating the tree).

    Args:
        tree: Tree to be instantiated.
        analysis_function: Analysis function to be called. All of the other arguments should be bound with partial.
    Returns:
        Executes the analysis function with the instantiated tree.
    """
    return analysis_function(tree())


def run_shared(  # noqa: C901
    collision_system: str,
    analysis_function: Callable[
        [data_manager.Tree, analysis_objects.Dataset, Sequence[helpers.RangeSelector], str, bool],
        Sequence[Mapping[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]],
    ],
    dataset_config_filename: Path,
    hists_filename: str,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    z_cutoff: float = 0.2,
    output: Path = Path("output"),
    plot_only: bool = False,
    force_reprocessing: bool = False,
    number_of_cores: int = 1,
    override_filenames: Optional[Sequence[Union[str, Path]]] = None,
    additional_kwargs_for_analysis: Optional[Dict[str, str]] = None,
) -> Tuple[
    List[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]],
    analysis_objects.Dataset,
]:
    """ Run the given analysis function.

    Args:
        collision_system: Name of the collision system.
        analysis_function: Function to perform the desired analysis on a single tree.
        hists_filename: Filename to be used for the merged hists generated in this analysis.
        jet_pt_bins: Jet pt bins to be selected in the analysis.
        z_cutoff: Z cutoff. Default: 0.2.
        output: Output directory. Default: `Path("output")`.
        plot_only: Only plot using the stored, fully merged hists. Don't event try to access
            the underlying files. Default: False.
        force_reprocessing: Force the trees to be reprocessed regardless of whether they already
            have output histograms.
        number_of_cores: Number of cores to be used for processing. If more than 1, then use multiprocessing.
            Careful of memory usage!! Default: 1.
        override_filenames: Filenames to be used during the analysis, overriding those specified in the
            configuration. Default: None, in which cause the filenames in the configuration are used.
        additional_kwargs_for_analysis: Additional keyword arguments to pass on to the single tree analysis
            function. Default: {}.
    Returns:
        ((hists returned from the analysis, merged over all of the inputs file), dataset configuration)
    """
    # Validation
    if additional_kwargs_for_analysis is None:
        additional_kwargs_for_analysis = {}

    # Configuration
    # Only need to set options which vary from the default.
    settings_class_map: Mapping[str, Type[analysis_objects.AnalysisSettings]] = {
        "embedPythia": analysis_objects.PtHardAnalysisSettings,
    }
    dataset = analysis_objects.Dataset.from_config_file(
        collision_system=collision_system,
        config_filename=dataset_config_filename,
        override_filenames=override_filenames,
        hists_filename_stem=hists_filename,
        output_base=output,
        settings_class=settings_class_map.get(collision_system, analysis_objects.AnalysisSettings),
        z_cutoff=z_cutoff,
    )

    # Output hists
    # The list is because there could be more than one set of hists returned from an analysis function.
    output_hists: List[
        Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]
    ] = []

    # Have a special option if we're plotting only so we can just read the final files.
    # Even though merging isn't very hard, all of that I/O is still slow than reading them once.
    if plot_only:
        logger.info(f"Loading system {collision_system} with dataset {dataset.name} for plotting only")
        if dataset.hists_filename.exists():
            # Read the stored hists.
            # We don't use YAML because it would be super slow!
            with gzip.GzipFile(dataset.hists_filename, "r") as pkl_file:
                output_hists = pickle.load(pkl_file)  # type: ignore

            return output_hists, dataset

        # If the file doesn't exist, we still need to process
        logger.warning("Requested plotting only, but the hists aren't available. Continuing on to processing.")

    # Setup dataset
    dm = data_manager.IterateTrees(
        filenames=dataset.filenames,
        tree_name=dataset.tree_name,
        # Mypy is getting confused by Sequence[str] because str is an iterable, so we ignore the type...
        branches=dataset.branches,  # type: ignore
    )
    logger.info("Setup complete. Beginning processing of trees.")

    # Create the analysis functions
    # We bind them with partial so we can execute them using map (which enables multiprocessing).
    analyze_single_tree_func = functools.partial(
        analysis_function,
        dataset=dataset,
        jet_pt_bins=jet_pt_bins,
        hists_filename_stem=dataset.hists_filename.stem,
        force_reprocessing=force_reprocessing,
        **additional_kwargs_for_analysis,
    )
    analyze_single_tree_func_multiprocessing = functools.partial(
        _wrap_multiprocessing, analysis_function=analyze_single_tree_func,
    )

    # Iterate over trees.
    progress_manager = enlighten.get_manager()
    # We need to use fully lazy iteration if we're using multiprocessing. Otherwise, we run into problems
    # with pickling objects (which is necessary from them to be sent to the other processes).
    dm_iterator = dm.lazy_iteration(fully_lazy=(number_of_cores > 1))
    results: List[
        Sequence[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]]
    ] = []
    with progress_manager.counter(total=len(dm), desc="Analyzing", unit="tree") as tree_counter:
        if number_of_cores > 1:
            # Only use 2 nodes because memory usage may become too large...
            with Pool(nodes=number_of_cores) as pool:
                for r in tree_counter(pool.imap(analyze_single_tree_func_multiprocessing, dm_iterator)):
                    results.append(r)
        else:
            for r in tree_counter(map(analyze_single_tree_func, dm_iterator)):
                results.append(r)

    # Convert and merge all of the result hists.
    for result in results:
        for hist_result in result:
            for h in hist_result.values():
                h.convert_boost_histograms_to_binned_data()

    # For each hist result, we want to merge the output from all of the files.
    for i, hist_result in enumerate(results[0]):
        hist_output = {}
        for k in hist_result.keys():
            hist_output[k] = cast(
                analysis_objects.Hists[analysis_objects.T_SubstructureHists],
                sum([per_file_result[i][k] for per_file_result in results]),
            )
        output_hists.append(hist_output)

    # Write out the merged hists
    # Write with pkl because yaml is super slow for hists that are this large.
    with gzip.GzipFile(dataset.hists_filename, "w") as pkl_file:
        pickle.dump(output_hists, pkl_file)  # type: ignore

    progress_manager.stop()

    return output_hists, dataset


# def compare_PbPb_to_embedded(pbpb_hists_path: Path, embedded_hists_path: Path) -> None:
#    # Setup
#    y = setup_yaml()
#    with open(pbpb_hists_path, "r") as f:
#        pbpb_hists = y.load(f)
#    with open(embedded_hists_path, "r") as f:
#        embedded_hists = y.load(f)
#    ...


def parse_arguments(name: str) -> List[Path]:
    parser = argparse.ArgumentParser(description=f"Run {name}")

    parser.add_argument("-f", "--filenames", nargs="+", default=[])
    args = parser.parse_args()
    # Validation for filenames
    filenames = [Path(f) for f in args.filenames]
    return filenames


def embed_pythia_entry_point() -> None:
    helpers.setup_logging()
    filenames = parse_arguments(name="embed pythia")

    collision_system = "embedPythia"
    jet_pt_bins = [
        helpers.RangeSelector(min=0, max=120),
        helpers.RangeSelector(min=40, max=120),
        # Most likely where we will actually measure.
        helpers.RangeSelector(min=80, max=120),
        helpers.RangeSelector(min=60, max=80),
        helpers.RangeSelector(min=80, max=100),
        helpers.RangeSelector(min=100, max=120),
    ]

    (response_hists, matching_hists), dataset = run_shared(  # type: ignore
        collision_system=collision_system,
        analysis_function=analyze_single_tree_embedding,
        dataset_config_filename=Path("config") / "datasets.yaml",
        hists_filename="embedding_hists",
        jet_pt_bins=jet_pt_bins,
        z_cutoff=0.2,
        override_filenames=filenames,
    )
    logger.info(f"Finished processing embedPythia for: {filenames}")


if __name__ == "__main__":
    helpers.setup_logging()

    # Setup and run
    config_filename = Path("config") / "datasets.yaml"
    plot_only = False
    jet_pt_bins = [
        # Broadest range
        helpers.RangeSelector(min=40, max=120),
        # Most likely where we will actually measure.
        helpers.RangeSelector(min=80, max=120),
        # Individual ranges.
        helpers.RangeSelector(min=40, max=60),
        helpers.RangeSelector(min=60, max=80),
        helpers.RangeSelector(min=80, max=100),
        helpers.RangeSelector(min=100, max=120),
    ]
    z_cutoff = 0.4
    # Standard analysis
    # (data_hists,), data_dataset = run_shared(
    #    collision_system="pp",
    #    analysis_function=analyze_single_tree,
    #    dataset_config_filename=config_filename,
    #    hists_filename="data_hists",
    #    jet_pt_bins=jet_pt_bins,
    #    z_cutoff=z_cutoff,
    #    plot_only=plot_only,
    #    force_reprocessing=True,
    #    number_of_cores=2,
    # )
    # plot_results.lund_plane(all_hists=data_hists, jet_type_label="det", path=data_dataset.output)
    # Toy
    # data_prefix = "hybrid"
    # collision_system = f"toy_true_{data_prefix}_splittings_iterative_allTrueSplittings_delta_R_040"
    # (toy_hists,), dataset = run_shared(
    #    collision_system=collision_system,
    #    analysis_function=analyze_single_tree_toy,
    #    dataset_config_filename=config_filename,
    #    hists_filename="toy_hists",
    #    jet_pt_bins=jet_pt_bins,
    #    z_cutoff=z_cutoff,
    #    plot_only=plot_only,
    #    number_of_cores=2,
    #    additional_kwargs_for_analysis=dict(
    #        data_prefix=data_prefix,
    #    )
    # )
    # plot_results.toy(all_toy_hists=toy_hists, data_prefix=data_prefix, path=dataset.output)
    # Embedding
    (embedded_true_hists, embedded_hybrid_hists, response_hists, matching_hists), embedded_dataset = run_shared(  # type: ignore
        collision_system="embedPythia",
        analysis_function=analyze_single_tree_embedding,
        dataset_config_filename=config_filename,
        hists_filename="embedding_hists",
        jet_pt_bins=jet_pt_bins,
        z_cutoff=z_cutoff,
        plot_only=plot_only,
        force_reprocessing=True,
        number_of_cores=1,
        # additional_kwargs_for_analysis=dict(
        #    scale_n_jets_when_loading_hists=True,
        # )
    )
    ## True, hybrid hists
    # plot_results.lund_plane(all_hists=embedded_true_hists, jet_type_label="true", path=embedded_dataset.output)
    # plot_results.lund_plane(all_hists=embedded_hybrid_hists, jet_type_label="hybrid", path=embedded_dataset.output)
    ## Responses
    # plot_results.responses(all_response_hists=response_hists, path=embedded_dataset.output)
    ## Matching
    # plot_results.matching(all_matching_hists=matching_hists, path=embedded_dataset.output)

    # Comparison
    # plot_results.compare_kt(all_data_hists=data_hists, all_embedded_hists=embedded_hybrid_hists, data_dataset=data_dataset, embedded_dataset=embedded_dataset)

    IPython.start_ipython(user_ns=locals())
