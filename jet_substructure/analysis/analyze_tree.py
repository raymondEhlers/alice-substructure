#!/usr/bin/env python3

""" Analyze the dynamical grooming tree.

.. codeauthor:: Raymnod Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import argparse
import logging
import pickle
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

import attr
import awkward as ak
import coloredlogs
import enlighten
import IPython
import numpy as np
from pachyderm import binned_data, yaml

from jet_substructure.analysis import plot_results
from jet_substructure.base import analysis_objects, data_manager, helpers, substructure_methods
from jet_substructure.base.helpers import UprootArray


logger = logging.getLogger(__name__)


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
    """ Construct the subtructure jet objects for data stored under a given prefix in a tree.

    Ideally, the object has alrady been created and stored. If not, it will be created and then
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
                jet_pt=jets.jet_pt, jet_constituents=jets.constituents,
                subjets=jets.subjets, jet_splittings=jets.splittings
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

    return jets


def analyze_single_tree(
    tree: data_manager.Tree,
    z_cutoff: float,
    R: float,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    progress_manager: enlighten.Manager,
    y: yaml.ruamel.yaml.YAML,
    output: Path,
    force_reprocessing: bool = False,
) -> Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]]:
    # Setup
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]] = {}
    # If the hists already exist, skip processing the tree and just return the hists instead (which is way faster!)
    train_number = tree.filename.parent.name
    yaml_filename = output / f"{train_number}_{tree.filename.with_suffix('.yaml').name}"
    if yaml_filename.exists() and not force_reprocessing:
        logger.info(f"Skipping processing of tree {tree.filename} by loading data from stored hists.")
        with open(yaml_filename, "r") as f:
            hists = y.load(f)
            return hists

    # Since we're actually processing, we setup the output hists
    for iterative_splittings in [False, True]:
        for jet_pt_bin in jet_pt_bins:
            hists[
                analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
            ] = analysis_objects.create_substructure_hists(iterative_splittings=iterative_splittings, z_cutoff=z_cutoff)

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
        # Convert, write, and return the empty hists. We can't process this data :-(
        return _convert_and_write_hists(hists=hists, tree_filename=tree.filename, yaml_filename=yaml_filename, y=y)

    # Loop over iterations (jet pt ranges, iterative splitting)
    with progress_manager.counter(
        total=len(hists), desc="Analyzing", unit="variation", leave=False
    ) as variations_counter:
        for identifier, h in variations_counter(hists.items()):
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
            hists[identifier].inclusive.fill(inputs, jet_R=R)
            # Dynamical z
            inputs = analysis_objects.FillHistogramInput(restricted_jets, splittings, *splittings.dynamical_z(R=R))
            hists[identifier].dynamical_z.fill(inputs, jet_R=R)
            # Dynamical kt
            inputs = analysis_objects.FillHistogramInput(restricted_jets, splittings, *splittings.dynamical_kt(R=R))
            hists[identifier].dynamical_kt.fill(inputs, jet_R=R)
            # Dynamical time
            inputs = analysis_objects.FillHistogramInput(restricted_jets, splittings, *splittings.dynamical_time(R=R))
            hists[identifier].dynamical_time.fill(inputs, jet_R=R)
            # Leading kt
            inputs = analysis_objects.FillHistogramInput(restricted_jets, splittings, *splittings.leading_kt())
            hists[identifier].leading_kt.fill(inputs, jet_R=R)
            # Leading kt with z cutoff
            inputs = analysis_objects.FillHistogramInput(
                restricted_jets, splittings, *splittings.leading_kt(z_cutoff=z_cutoff)
            )
            hists[identifier].leading_kt_hard_cutoff.fill(inputs, jet_R=R)
            # import numpy as np
            # if (np.log(1.0 / splittings[indices].delta_R.flatten()) < 2).any() and (np.log(splittings[indices].kt.flatten()) < -1).any():
            #    logger.warning("Maybe the z cut isn't working??")
            #    import IPython; IPython.embed()

    # IPython.start_ipython(user_ns=locals())

    return _convert_and_write_hists(hists=hists, tree_filename=tree.filename, yaml_filename=yaml_filename, y=y)


def analyze_single_tree_toy(
    tree: data_manager.Tree,
    data_prefix: str,
    z_cutoff: float,
    R: float,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    progress_manager: enlighten.Manager,
    y: yaml.ruamel.yaml.YAML,
    output: Path,
    force_reprocessing: bool = False,
) -> Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureToyHists]]:
    # Validation
    if data_prefix == "hybrid":
        data_prefix = "data"
    # Setup
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureToyHists]] = {}
    # If the hists already exist, skip processing the tree and just return the hists instead (which is way faster!)
    train_number = tree.filename.parent.name
    pkl_filename = output / f"{train_number}_{tree.filename.with_suffix('.pkl').name}"
    if pkl_filename.exists() and not force_reprocessing:
        logger.info(f"Skipping processing of tree {tree.filename} by loading data from stored hists.")
        with open(pkl_filename, "rb") as f:
            hists = pickle.load(f)
            return hists

    # Since we're actually processing, we setup the output hists
    for iterative_splittings in [False, True]:
        for jet_pt_bin in jet_pt_bins:
            hists[
                analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
            ] = analysis_objects.create_substructure_toy_hists(
                iterative_splittings=iterative_splittings, z_cutoff=z_cutoff
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
        return hists

    # Loop over iterations (jet pt ranges, iterative splitting)
    with progress_manager.counter(
        total=len(hists), desc="Analyzing", unit="variation", leave=False
    ) as variations_counter:
        for identifier, h in variations_counter(hists.items()):
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
                *restricted_data_jets_splittings.dynamical_z(R=R),
            )
            # TODO: We absolutely shouldn't be calcuating the splitting properties here!
            # TODO: If we take the leading, we already know that it was only one splitting, and we already
            # TODO: know the values...
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets, restricted_true_jets_splittings, *restricted_true_jets_splittings.dynamical_z(R=R)
            )
            hists[identifier].dynamical_z.fill(
                data_inputs=data_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            )
            # Dynamical kt
            data_inputs = analysis_objects.FillHistogramInput(
                restricted_data_jets,
                restricted_data_jets_splittings,
                *restricted_data_jets_splittings.dynamical_kt(R=R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.dynamical_kt(R=R),
            )
            hists[identifier].dynamical_kt.fill(
                data_inputs=data_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            )
            # Dynamical time
            data_inputs = analysis_objects.FillHistogramInput(
                restricted_data_jets,
                restricted_data_jets_splittings,
                *restricted_data_jets_splittings.dynamical_time(R=R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.dynamical_time(R=R),
            )
            hists[identifier].dynamical_time.fill(
                data_inputs=data_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            )
            # Leading kt
            data_inputs = analysis_objects.FillHistogramInput(
                restricted_data_jets, restricted_data_jets_splittings, *restricted_data_jets_splittings.leading_kt(),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets, restricted_true_jets_splittings, *restricted_true_jets_splittings.leading_kt()
            )
            hists[identifier].leading_kt.fill(
                data_inputs=data_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            )
            # Leading kt with z cutoff
            data_inputs = analysis_objects.FillHistogramInput(
                restricted_data_jets,
                restricted_data_jets_splittings,
                *restricted_data_jets_splittings.leading_kt(z_cutoff=z_cutoff),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.leading_kt(z_cutoff=z_cutoff),
            )
            # Ensure that there are sufficient values!
            # TODO: This doesn't work because the true frequently fails. They somehow need to be the same length...
            # IPython.embed()
            # hists[identifier].leading_kt.fill(
            #    data_inputs=data_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            # )

    IPython.start_ipython(user_ns=locals())

    # Store hists with pickle because it takes too longer otherwise.
    with open(pkl_filename, "wb") as pkl_file:
        pickle.dump(hists, pkl_file)

    return hists


def _select_and_retrieve_splittings(
    jets: substructure_methods.SubstructureJetArray, jet_pt_mask: UprootArray[bool], iterative_splittings: bool
) -> Tuple[substructure_methods.SubstructureJetArray, substructure_methods.JetSplittingArray]:
    # Ensure that there are sufficient counts
    restricted_jets = jets[jet_pt_mask]
    if iterative_splittings:
        # Only keep iterative splittings.
        splittings = restricted_jets.splittings.iterative_splittings(restricted_jets.subjets)
    else:
        splittings = restricted_jets.splittings

        # TODO: Test this more extensively.
        # comparison = restricted_jets.splittings.iterative_splittings(restricted_jets.subjets)
        # if (comparison != splittings).any().any():
        #    logger.warning("An actual disagreement in pythia!!")
        #    IPython.embed()

    return restricted_jets, splittings


def _subjets_contributing_to_splittings(inputs: analysis_objects.FillHistogramInput) -> substructure_methods.SubjetArray:
    """ Determine which subjets contribute to the selected splitting.

    We do this by looking for subjets with a a parent splitting index that is equal to the selected index.

    Args:
        inputs: Jets and splittings selected by a particular algorithm.
    Returns:
        Subjets which contributed to the selected splittings. There will always be 2 subjets by definition.
    """
    # In order to compare to the subjets directly, we need to expand the indices to the same dimension as the subjets.
    selected_indices_mask = (
        inputs.jets.subjets.parent_splitting_index.ones_like() * inputs.indices.flatten()
    )
    matched_subjets_unsorted = inputs.jets.subjets[
        selected_indices_mask == inputs.jets.subjets.parent_splitting_index
    ]
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
    # Coerces the bool into an integer by taking 1 - array.
    # The leading subjet will be have a 0, while the subleading will have a 1.
    subjets_pt_comparison = 1 - (
        subjets_unsorted[:, 0].constituents.pt.sum() > subjets_unsorted[:, 1].constituents.pt.sum()
    )
    # For each subjet_pt_comparison, we want to take the index of the leading subjet and use that to extract the leading subjet.
    leading_indices = ak.JaggedArray.fromoffsets(range(len(subjets_pt_comparison) + 1), subjets_pt_comparison)
    subjets_leading = subjets_unsorted[leading_indices].flatten()
    # Same idea for the subleading subjet (which is necessarily 1 - subjets_pt_comparison because there are only two subjets.
    subleading_indices = ak.JaggedArray.fromoffsets(range(len(subjets_pt_comparison) + 1), 1 - subjets_pt_comparison)
    subjets_subleading = subjets_unsorted[subleading_indices].flatten()

    return subjets_leading, subjets_subleading


def determine_matching_types(
    matched_subjets: substructure_methods.SubjetArray, hybrid_subjets: substructure_methods.SubjetArray,
) -> UprootArray[bool]:
    # We split the array into chunks to keep memory usage to a more reasonable level.
    number_of_chunks = 5

    def split(a: substructure_methods.SubjetArray, n: int) -> Iterable[Tuple[substructure_methods.SubjetArray, slice]]:
        """ Split an array into n chunks.

        From: https://stackoverflow.com/a/2135920/12907985
        """
        k, m = divmod(len(a), n)
        return ((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)], slice(i * k + min(i, m), (i + 1) * k + min(i + 1, m))) for i in range(n))

    shared_constituents_pts = np.zeros(len(matched_subjets))
    #for matched_subset, hybrid_subset, shared_constituents_pts_subset in zip(np.array_split(matched_subjets.constituents, number_of_chunks), np.array_split(hybrid_subjets.constituents, number_of_chunks), np.array_split(shared_constituents_pts, number_of_chunks)):
    # Can't use np.array_split (even though it would be nice and probably better tested) because the output ends up as numpy array.
    # In this case, we don't want such a conversion.
    for (matched_subset, selected_range), (hybrid_subset, _) in zip(split(matched_subjets.constituents, number_of_chunks), split(hybrid_subjets.constituents, number_of_chunks)):
        constituent_pairs = matched_subset.argcross(hybrid_subset)
        matched_leading_indices, hybrid_leading_indices = constituent_pairs.unzip()

        index_matching = (
            matched_subset[matched_leading_indices].global_index
            == hybrid_subset[hybrid_leading_indices].global_index
        )

        shared_constituents_pts[selected_range] = matched_subset[matched_leading_indices][index_matching].pt.sum()

    # Sanity check
    if (shared_constituents_pts > matched_subjets.constituents.pt.sum()).any():
        logger.warning("Constituent pts are greater than the subjet pts...")
        IPython.embed()
        raise ValueError("Constituent pts are greater than the subjet pts...")

    matched = (shared_constituents_pts / matched_subjets.constituents.pt.sum()) > 0.5
    return cast(UprootArray[bool], matched)


def determine_matched_jets(
    hybrid_inputs: analysis_objects.FillHistogramInput, matched_inputs: analysis_objects.FillHistogramInput
) -> Tuple[analysis_objects.MatchingResult, analysis_objects.MatchingResult]:
    """

    The passed jets need to have the selected indices already applied.
    We need to work with the indices applied to these jets.

    """
    # Setup
    # delta = 0.001
    # Determine which subjets contribute to the selected splitting.
    matched_subjets_unsorted = _subjets_contributing_to_splittings(inputs=matched_inputs)
    hybrid_subjets_unsorted = _subjets_contributing_to_splittings(inputs=hybrid_inputs)

    # hybrid_inputs.subjets.parent_splitting_index.ones_like() * hybrid_inputs.indices.flatten()
    # Sort the subjets such that 0 is always the leading subjet.
    # matched_subjets_pt_comparison = 1 - (matched_subjets_unsorted[:, 0].constituents.pt.sum() > matched_subjets_unsorted[:, 1].constituents.pt.sum() * 1)
    # matched_subjets_leading = matched_subjets_unsorted[matched_subjets_pt_comparison]
    # matched_subjets_subleading = matched_subjets_unsorted[1 - matched_subjets_pt_comparison]
    matched_subjets_leading, matched_subjets_subleading = _get_leading_and_subleading_subjets(matched_subjets_unsorted)
    hybrid_subjets_leading, hybrid_subjets_subleading = _get_leading_and_subleading_subjets(hybrid_subjets_unsorted)

    # This works...
    # In [119]: matched_subjets[:, 0].constituents.argcross(hybrid_subjets[:, 0].constituents)
    # Out[119]: <JaggedArray [[(0, 0) (0, 1) (0, 2) ... (10, 9) (10, 10) (10, 11)] [(0, 0) (0, 1) (0, 2) ... (16, 14) (16, 15) (16, 1
    # 6)] [(0, 0) (0, 1) (0, 2) ... (16, 9) (16, 10) (16, 11)] ... [(0, 0) (0, 1) (0, 2) ... (8, 13) (8, 14) (8, 15)] [(0, 0) (0, 1)
    # (0, 2) ... (8, 13) (8, 14) (8, 15)] [(0, 0) (0, 1) (0, 2) ... (8, 13) (8, 14) (8, 15)]] at 0x7f035ec61790>

    # shared_constituent_pts_matched_leading_hybrid_leading = determine_matching_types(matched_subjets_leading, hybrid_subjets_leading)
    # shared_constituent_pts_matched_leading_hybrid_subleading = determine_matching_types(matched_subjets_leading, hybrid_subjets_subleading)
    # shared_constituent_pts_matched_subleading_hybrid_leading = determine_matching_types(matched_subjets_subleading, hybrid_subjets_leading)
    # shared_constituent_pts_matched_subleading_hybrid_subleading = determine_matching_types(matched_subjets_subleading, hybrid_subjets_subleading)

    # matched_leading_properly = (shared_constituent_pts_matched_leading_hybrid_leading / matched_inputs.jets.jet_pt) > 0.5
    # matched_leading_mistag = (shared_constituent_pts_matched_leading_hybrid_subleading / matched_inputs.jets.jet_pt) > 0.5
    # matched_leading_failed = ~matched_leading_properly & ~matched_leading_mistag
    # matched_subleading_properly = (shared_constituent_pts_matched_subleading_hybrid_leading / matched_inputs.jets.jet_pt) > 0.5
    # matched_subleading_mistag = (shared_constituent_pts_matched_subleading_hybrid_subleading / matched_inputs.jets.jet_pt) > 0.5
    # matched_subleading_failed = ~matched_subleading_properly & ~matched_subleading_mistag

    matched_leading_properly = determine_matching_types(matched_subjets_leading, hybrid_subjets_leading)
    matched_leading_mistag = determine_matching_types(matched_subjets_leading, hybrid_subjets_subleading)
    matched_subleading_properly = determine_matching_types(matched_subjets_subleading, hybrid_subjets_leading)
    matched_subleading_mistag = determine_matching_types(matched_subjets_subleading, hybrid_subjets_subleading)
    matched_leading_failed = ~matched_leading_properly & ~matched_leading_mistag
    matched_subleading_failed = ~matched_subleading_properly & ~matched_subleading_mistag

    # IPython.embed()

    return (
        analysis_objects.MatchingResult(matched_leading_properly, matched_leading_mistag, matched_leading_failed),
        analysis_objects.MatchingResult(
            matched_subleading_properly, matched_subleading_mistag, matched_subleading_failed
        ),
    )

    # Moved to function
    # constituent_pairs = matched_subjets_leading.constituents.argcross(hybrid_subjets_leading.constituents)
    # matched_leading_indices, hybrid_leading_indices = constituent_pairs.unzip()

    # index_matching = (
    #    matched_subjets_leading.constituents[matched_leading_indices].global_index
    #    == hybrid_subjets_leading.constituents[hybrid_leading_indices].global_index
    # )

    # constituent_pts = matched_subjets_leading.constituents[matched_leading_indices][index_matching].pt.sum()
    # END moved to function

    # IPython.embed()

    # TODO: Use distance at some point. Can use delta_R that I impelemented.
    # delta_eta = matched_subjets_leading.constituents[matched_leading_indices].eta - \
    #    hybrid_subjets_leading.constituents[hybrid_leading_indices].eta
    # delta_phi = matched_subjets_leading.constituents[matched_leading_indices].phi - \
    #    hybrid_subjets_leading.constituents[hybrid_leading_indices].phi

    # delta_eta_mask = np.abs(delta_eta) < delta
    # delta_phi_mask = np.abs(delta_phi) < delta

    # subjet_pairs = matched_subjets.argcross(hybrid_subjets)
    # matched_subjets_indices, hybrid_subjets_indices = subjet_pairs.unzip()

    # matched_subjets_leading = matched_subjets[subjet_pairs][:, 0]
    # matched_subjets_subleading = matched_subjets[subjet_pairs][:, 1]

    # Work with matched jets.
    # Match constituents.
    # Take the two hybrid subjets, and the two detlevel subjets. Compare them.
    # track_pairs = matched_subjets.constituents.argcross(hybrid_subjets.constituents, nested=True)
    ##delta_phi = matched_subjets.constituents[track_pairs[0, :]].phi() - hybrid_subjets.constituents[track_pairs[1, :]].phi()
    ##delta_eta = matched_subjets.constituents[track_pairs[0, :]].eta() - hybrid_subjets.constituents[track_pairs[1, :]].eta()
    ## use unzip here instead...
    # matched_subjets_indices, hybrid_subjets_indices = track_pairs.unzip()
    # delta_phi = matched_subjets.constituents[matched_subjets_indices].phi - hybrid_subjets.constituents[hybrid_subjets_indices].phi
    # delta_eta = matched_subjets.constituents[matched_subjets_indices].eta - hybrid_subjets.constituents[hybrid_subjets_indices].eta

    # constituent_pts = matched_subjets.constituents[(delta_phi < delta) & (delta_eta < delta)]

    # return (constituent_pts / matched_inputs.jets.jet_pt) > 0.5

    # for (int i = 0; i < constDet->size(); i++)
    # {
    #    float eta_det = constDet->at(i).eta();
    #    float phi_det = constDet->at(i).phi();
    #    for (int j  = 0; j < constHyb->size(); j++)
    #    {
    #        float eta_hyb = constHyb->at(j).eta();
    #        float phi_hyb = constHyb->at(j).phi();
    #        float deta = eta_hyb - eta_det;
    #        deta = std::sqrt(deta*deta);
    #        if (deta > delta) continue;
    #        float dphi = phi_hyb - phi_det;
    #        dphi = std::sqrt(dphi*dphi);
    #        if (dphi > delta) continue;
    #        sumpT+=constDet->at(i).pt();
    #    }
    # }

    # if sumpT / matched_jets.jet_pt() > 0.5:
    #    return True
    # return False


# def process_matching_results(matched_inputs: analysis_objects.FillHistogramInput, leading_matching: analysis_objects.MatchingResult,
#                             subleading_matching: analysis_objects.MatchingResult) -> None:
#    h_leading_matched_all.fill(matched_inputs.jets.jet_pt)
#    h_leading_matched_properly.fill(matched_inputs.jets.jet_pt[leading_matching.properly])
#    h_leading_matched_missed.fill(matched_inputs.jets.jet_pt[leading_matching.mistag])
#    h_leading_matched_failed.fill(matched_inputs.jets.jet_pt[leading_matching.failed])


def analyze_single_tree_embedding(
    tree: data_manager.Tree,
    z_cutoff: float,
    R: float,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    progress_manager: enlighten.Manager,
    y: yaml.ruamel.yaml.YAML,
    scale_factors: Mapping[int, float],
    train_number_to_pt_hard_bin: Mapping[int, int],
    output: Path,
    force_reprocessing: bool = False,
) -> Tuple[
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
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]] = {}
    matching_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
    ] = {}
    # If the output file already exist, skip processing the tree and just return the hists instead (which is way faster!)
    train_number = tree.filename.parent.name
    pkl_filename = output / f"{train_number}_{tree.filename.with_suffix('.pkl').name}"
    if pkl_filename.exists() and not force_reprocessing:
        logger.info(f"Skipping processing of tree {tree.filename} by loading data from stored hists.")
        with open(pkl_filename, "rb") as f:
            hists, matching_hists = pickle.load(f)
            return hists, matching_hists
    # Determine scale factor
    # NOTE: This relies on the train_number being up a directory!
    pt_hard_bin = train_number_to_pt_hard_bin[int(train_number)]
    scale_factor = scale_factors[pt_hard_bin]

    # Since we're actually processing, we setup the output hists
    # Responses
    for iterative_splittings in [False, True]:
        for jet_pt_bin in jet_pt_bins:
            hists[
                analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
            ] = analysis_objects.create_substructure_response_hists(
                iterative_splittings=iterative_splittings, z_cutoff=z_cutoff
            )
    # Matching
    for iterative_splittings in [False, True]:
        matching_hists[
            analysis_objects.Identifier(iterative_splittings, jet_pt_bin=helpers.RangeSelector(0, 200))
        ] = analysis_objects.create_matching_hists(iterative_splittings=iterative_splittings, z_cutoff=z_cutoff)

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
        # Convert, write, and return the empty hists. We can't process this data :-(
        return hists, matching_hists

    # Loop over iterations (jet pt ranges, iterative splitting)
    with progress_manager.counter(
        total=len(hists), desc="Analyzing", unit="variation", leave=False
    ) as variations_counter:
        for identifier, h in variations_counter(hists.items()):
            # We want to restrict a constant hybrid jet pt range for both true and hybrid.
            # This will allow us to compare to measured jet pt ranges.
            jet_pt_mask = identifier.jet_pt_bin.mask_array(hybrid_jets.jet_pt)
            # Add additional restrictions that we can't handle single constituent jets.
            # TODO: Can we do better???
            jet_pt_mask = jet_pt_mask & (hybrid_jets.constituents.counts > 1) & (true_jets.constituents.counts > 1)
            # Require that we have jets that aren't dominated by hybrid jets.
            jet_pt_mask = jet_pt_mask & (true_jets.constituents.max_pt > hybrid_jets.constituents.max_pt)
            # TODO: Do we need any additional cuts??

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
            # Dynamical z
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.dynamical_z(R=R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets, restricted_true_jets_splittings, *restricted_true_jets_splittings.dynamical_z(R=R)
            )
            hists[identifier].dynamical_z.fill(
                hybrid_inputs=hybrid_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            )
            # Dynamical kt
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.dynamical_kt(R=R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.dynamical_kt(R=R),
            )
            hists[identifier].dynamical_kt.fill(
                hybrid_inputs=hybrid_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            )
            # Dynamical time
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.dynamical_time(R=R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.dynamical_time(R=R),
            )
            hists[identifier].dynamical_time.fill(
                hybrid_inputs=hybrid_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            )
            # Leading kt
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.leading_kt(),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets, restricted_true_jets_splittings, *restricted_true_jets_splittings.leading_kt()
            )
            hists[identifier].leading_kt.fill(
                hybrid_inputs=hybrid_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            )
            # Leading kt with z cutoff
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.leading_kt(z_cutoff=z_cutoff),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_true_jets,
                restricted_true_jets_splittings,
                *restricted_true_jets_splittings.leading_kt(z_cutoff=z_cutoff),
            )
            # Ensure that there are sufficient values!
            # IPython.embed()
            hists[identifier].leading_kt.fill(
                hybrid_inputs=hybrid_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            )

            # TODO: USE THESE HISTS!!!

    # Convert to BinnedData and store the hists
    # for h in hists.values():
    #    h.convert_boost_histograms_to_binned_data()
    # Store hists with pickle because it takes too longer otherwise.
    with open(pkl_filename, "wb") as pkl_file:
        pickle.dump((hists, matching_hists), pkl_file)

    # Look at matched jets
    matching_hists = matching(
        matching_hists=matching_hists,
        matched_jets=det_level_jets,
        hybrid_jets=hybrid_jets,
        z_cutoff=z_cutoff,
        R=R,
        progress_manager=progress_manager,
    )

    # Store hists with pickle because it takes too longer otherwise.
    # Write again here (despite the waste of writing twice) so we can keep the response hists even
    # if the matching fails
    with open(pkl_filename, "wb") as pkl_file:
        pickle.dump((hists, matching_hists), pkl_file)

    return hists, matching_hists


def matching(
    matching_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
    ],
    matched_jets: substructure_methods.SubstructureJetArray,
    hybrid_jets: substructure_methods.SubstructureJetArray,
    z_cutoff: float,
    R: float,
    progress_manager: enlighten.Manager,
) -> Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]]:
    """ Determine the prong matching for jets substructure techniques.

    """
    with progress_manager.counter(
        total=len(matching_hists), desc="Analyzing", unit="variation", leave=False
    ) as variations_counter:
        for identifier, h in variations_counter(matching_hists.items()):
            # Ensure that we don't have single track jets because the splitting won't be defined for that case.
            # No actual jet pt range restrictions.
            mask = (hybrid_jets.constituents.counts > 1) & (matched_jets.constituents.counts > 1)
            # Require that we have jets that aren't dominated by hybrid jets.
            mask = mask & (matched_jets.constituents.max_pt > hybrid_jets.constituents.max_pt)

            restricted_hybrid_jets, restricted_hybrid_jets_splittings = _select_and_retrieve_splittings(
                hybrid_jets, mask, identifier.iterative_splittings
            )
            restricted_matched_jets, restricted_matched_jets_splittings = _select_and_retrieve_splittings(
                matched_jets, mask, identifier.iterative_splittings
            )

            # Dynamical z
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.dynamical_z(R=R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_matched_jets,
                restricted_matched_jets_splittings,
                *restricted_matched_jets_splittings.dynamical_z(R=R),
            )
            leading_matching, subleading_matching = determine_matched_jets(hybrid_inputs, true_inputs)
            matching_hists[identifier].dynamical_z.fill(
                matched_inputs=true_inputs, leading=leading_matching, subleading=subleading_matching,
            )
            # Dynamical kt
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.dynamical_kt(R=R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_matched_jets,
                restricted_matched_jets_splittings,
                *restricted_matched_jets_splittings.dynamical_kt(R=R),
            )
            leading_matching, subleading_matching = determine_matched_jets(hybrid_inputs, true_inputs)
            matching_hists[identifier].dynamical_kt.fill(
                matched_inputs=true_inputs, leading=leading_matching, subleading=subleading_matching,
            )
            # Dynamical time
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.dynamical_time(R=R),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_matched_jets,
                restricted_matched_jets_splittings,
                *restricted_matched_jets_splittings.dynamical_time(R=R),
            )
            leading_matching, subleading_matching = determine_matched_jets(hybrid_inputs, true_inputs)
            matching_hists[identifier].dynamical_time.fill(
                matched_inputs=true_inputs, leading=leading_matching, subleading=subleading_matching,
            )
            # Leading kt
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.leading_kt(),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_matched_jets,
                restricted_matched_jets_splittings,
                *restricted_matched_jets_splittings.leading_kt(),
            )
            leading_matching, subleading_matching = determine_matched_jets(hybrid_inputs, true_inputs)
            matching_hists[identifier].leading_kt.fill(
                matched_inputs=true_inputs, leading=leading_matching, subleading=subleading_matching,
            )
            # Leading kt with z cutoff
            hybrid_inputs = analysis_objects.FillHistogramInput(
                restricted_hybrid_jets,
                restricted_hybrid_jets_splittings,
                *restricted_hybrid_jets_splittings.leading_kt(z_cutoff=z_cutoff),
            )
            true_inputs = analysis_objects.FillHistogramInput(
                restricted_matched_jets,
                restricted_matched_jets_splittings,
                *restricted_matched_jets_splittings.leading_kt(z_cutoff=z_cutoff),
            )
            # mask = (hybrid_inputs.indices.counts != 0) & (true_inputs.indices.counts != 0)
            # hybrid_inputs = analysis_objects.FillHistogramInput(
            #    hybrid_inputs.jets[mask],
            #    hybrid_inputs.splittings[mask],
            #    hybrid_inputs.values,
            #    hybrid_inputs.indices[mask],
            # )
            # true_inputs = analysis_objects.FillHistogramInput(
            #    true_inputs.jets[mask],
            #    true_inputs.splittings[mask],
            #    true_inputs.values,
            #    true_inputs.indices.pad(1)[mask],
            # )
            # IPython.embed()
            # determine_matched_jets(hybrid_inputs, true_inputs)
            # Ensure that there are sufficient values!
            # IPython.embed()
            # leading_matching, subleading_matching = determine_matched_jets(hybrid_inputs, true_inputs)
            # matching_hists[identifier].leading_kt_hard_cutoff.fill(
            #    matched_inputs=true_inputs, leading=leading_matching, subleading=subleading_matching,
            # )

    ...
    return matching_hists


def run(
    collision_system: str, jet_pt_bins: Sequence[helpers.RangeSelector], dataset_config_filename: Path, output: Path
) -> Tuple[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]], Path]:
    # Setup
    z_cutoff = 0.2
    # Configuration
    y = setup_yaml()
    with open(dataset_config_filename, "r") as f:
        config = y.load(f)
    dataset_config = config["datasets"][collision_system]["dataset"]
    dataset_name = dataset_config["name"]
    # finalize setup
    output = output / collision_system / dataset_name
    output.mkdir(parents=True, exist_ok=True)

    # Retrieve and setup data
    selected_dataset_config = config["available_datasets"][dataset_name]
    R = selected_dataset_config["jet_R"]
    dm = data_manager.IterateTrees(
        filenames=selected_dataset_config["files"],
        tree_name=selected_dataset_config["tree_name"],
        branches=dataset_config["branches"],
    )
    logger.info("Setup complete. Beginning processing of trees.")

    # Iterate over trees.
    progress_manager = enlighten.get_manager()
    results: List[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]]] = []
    with progress_manager.counter(total=len(dm), desc="Analyzing", unit="tree") as tree_counter:
        for tree in tree_counter(dm):
            logger.info(f"Processing tree from file {tree.filename}")
            tree_hists = analyze_single_tree(
                tree,
                z_cutoff=z_cutoff,
                R=R,
                jet_pt_bins=jet_pt_bins,
                progress_manager=progress_manager,
                y=y,
                output=output,
                force_reprocessing=True,
            )
            # hists[tree.filename] = tree_hists
            results.append(tree_hists)

    # Merge the hists
    full_hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]] = {}
    for k in results[0].keys():
        full_hists[k] = cast(
            analysis_objects.Hists[analysis_objects.SubstructureHists], sum([hists[k] for hists in results])
        )

    # Write out the merged hists
    with open(output / "hists.yaml", "w") as f:
        y.dump(full_hists, f)

    progress_manager.stop()

    return full_hists, output


def run_toy(
    collision_system: str,
    data_prefix: str,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    dataset_config_filename: Path,
    output: Path,
) -> Tuple[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureToyHists]], Path]:
    # Setup
    z_cutoff = 0.2
    # Configuration
    y = setup_yaml()
    with open(dataset_config_filename, "r") as f:
        config = y.load(f)
    dataset_config = config["datasets"][collision_system]["dataset"]
    dataset_name = dataset_config["name"]
    # finalize setup
    output = output / collision_system / dataset_name
    output.mkdir(parents=True, exist_ok=True)

    # Retrieve and setup data
    selected_dataset_config = config["available_datasets"][dataset_name]
    R = selected_dataset_config["jet_R"]
    dm = data_manager.IterateTrees(
        filenames=selected_dataset_config["files"],
        tree_name=selected_dataset_config["tree_name"],
        branches=dataset_config["branches"],
    )
    logger.info("Setup complete. Beginning processing of trees.")

    # Iterate over trees.
    progress_manager = enlighten.get_manager()
    results: List[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureToyHists]]] = []
    with progress_manager.counter(total=len(dm), desc="Analyzing", unit="tree") as tree_counter:
        for tree in tree_counter(dm):
            logger.info(f"Processing toy tree from file {tree.filename}")
            tree_hists = analyze_single_tree_toy(
                tree,
                data_prefix=data_prefix,
                z_cutoff=z_cutoff,
                R=R,
                jet_pt_bins=jet_pt_bins,
                progress_manager=progress_manager,
                y=y,
                output=output,
                force_reprocessing=False,
            )
            # hists[tree.filename] = tree_hists
            results.append(tree_hists)

    # Convert hists from boost hist to pachyderm
    # Still need to convert to BinnedData (because we stored data with pickle instead to speed up writing).
    for hists in results:
        for h in hists.values():
            h.convert_boost_histograms_to_binned_data()

    # Merge the hists
    full_hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureToyHists]] = {}
    for k in results[0].keys():
        full_hists[k] = cast(
            analysis_objects.Hists[analysis_objects.SubstructureToyHists], sum([hists[k] for hists in results])
        )

    # Write out the merged hists
    with open(output / "toy_hists.yaml", "w") as f:
        y.dump(full_hists, f)

    progress_manager.stop()

    return full_hists, output


def run_embedding(
    collision_system: str, jet_pt_bins: Sequence[helpers.RangeSelector], dataset_config_filename: Path, output: Path, filenames: Optional[Sequence[Union[str, Path]]] = None,
) -> Tuple[
    Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]],
    Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]],
    Path,
]:
    # Validation
    if filenames is None:
        filenames = []
    # Setup
    z_cutoff = 0.2
    # Configuration
    y = setup_yaml()
    with open(dataset_config_filename, "r") as f:
        config = y.load(f)
    dataset_config = config["datasets"][collision_system]["dataset"]
    dataset_name = dataset_config["name"]
    # finalize setup
    output = output / collision_system / dataset_name
    output.mkdir(parents=True, exist_ok=True)

    # Retrieve and setup data
    selected_dataset_config = config["available_datasets"][dataset_name]
    R = selected_dataset_config["jet_R"]
    scale_factors = selected_dataset_config["scale_factors"]
    train_number_to_pt_hard_bin = selected_dataset_config["train_number_to_pt_hard_bin"]
    # Take the passed filenames if provided. Otherwise, use the files in the configuration file.
    if not filenames:
        filenames = selected_dataset_config["files"]
    # And then setup the data manager.
    dm = data_manager.IterateTrees(
        filenames=filenames,
        tree_name=selected_dataset_config["tree_name"],
        branches=dataset_config["branches"],
    )
    logger.info("Setup complete. Beginning processing of trees.")

    # Iterate over trees.
    progress_manager = enlighten.get_manager()
    results: List[
        Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]]
    ] = []
    matching_results: List[
        Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]]
    ] = []
    with progress_manager.counter(total=len(dm), desc="Analyzing", unit="tree") as tree_counter:
        for tree in tree_counter(dm):
            logger.info(f"Processing tree from file {tree.filename}")
            tree_hists, matching_hists = analyze_single_tree_embedding(
                tree,
                z_cutoff=z_cutoff,
                R=R,
                jet_pt_bins=jet_pt_bins,
                progress_manager=progress_manager,
                y=y,
                scale_factors=scale_factors,
                train_number_to_pt_hard_bin=train_number_to_pt_hard_bin,
                output=output,
                force_reprocessing=False,
            )
            # hists[tree.filename] = tree_hists
            results.append(tree_hists)
            matching_results.append(matching_hists)

    # Convert hists from boost hist to pachyderm
    # Still need to convert to BinnedData (because we stored data with pickle instead to speed up writing).
    for hists in results:
        for h in hists.values():
            h.convert_boost_histograms_to_binned_data()
    for matching_hists in matching_results:
        for match_hist in matching_hists.values():
            match_hist.convert_boost_histograms_to_binned_data()

    # Merge the hists
    full_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]
    ] = {}
    for k in results[0].keys():
        full_hists[k] = cast(
            analysis_objects.Hists[analysis_objects.SubstructureResponseHists], sum([hists[k] for hists in results])
        )

    full_matching_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
    ] = {}
    for k in matching_results[0].keys():
        full_matching_hists[k] = cast(
            analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists],
            sum([hists[k] for hists in matching_results]),
        )

    # Write out the merged hists
    # Disable for now because it's super slow!
    # with open(output / "response_hists.yaml", "w") as f:
    #    y.dump(full_hists, f)

    progress_manager.stop()

    return full_hists, full_matching_hists, output


# def compare_PbPb_to_embedded(pbpb_hists_path: Path, embedded_hists_path: Path) -> None:
#    # Setup
#    y = setup_yaml()
#    with open(pbpb_hists_path, "r") as f:
#        pbpb_hists = y.load(f)
#    with open(embedded_hists_path, "r") as f:
#        embedded_hists = y.load(f)
#    ...

def setup_entry_point() -> None:
    # Basic setup
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s")
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)
    # Quiet down BinndData copy warnings
    logging.getLogger("pachyderm.binned_data").setLevel(logging.INFO)

def parse_arguments(name: str) -> List[Path]:
    parser = argparse.ArgumentParser(description=f"Run {name}")

    parser.add_argument("-f", "--filenames", nargs="+", default=[])
    args = parser.parse_args()
    # Validation for filenames
    filenames = [Path(f) for f in args.filenames]
    return filenames

def embed_pythia_entry_point() -> None:
    setup_entry_point()
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

    response_hists, matching_hists, output = run_embedding(
        collision_system=collision_system,
        jet_pt_bins=jet_pt_bins,
        dataset_config_filename=Path("config") / "datasets.yaml",
        output=Path("output"),
        filenames=filenames,
    )
    #return response_hists, matching_hists
    #plot_results.responses(all_response_hists=response_hists, path=output)
    #plot_results.matching(all_matching_hists=matching_hists, path=output)

if __name__ == "__main__":
    setup_entry_point()

    # Setup and run
    collision_system = "embedPythia"
    # data_prefix = "hybrid"
    # collision_system = f"toy_true_{data_prefix}_splittings_iterative_allTrueSplittings_delta_R_040"
    jet_pt_bins = [
        helpers.RangeSelector(min=0, max=120),
        helpers.RangeSelector(min=40, max=120),
        # Most likely where we will actually measure.
        helpers.RangeSelector(min=80, max=120),
        helpers.RangeSelector(min=60, max=80),
        helpers.RangeSelector(min=80, max=100),
        helpers.RangeSelector(min=100, max=120),
    ]
    # hists, output = run(
    #   collision_system=collision_system,
    #   jet_pt_bins=jet_pt_bins,
    #   dataset_config_filename=Path("config") / "datasets.yaml",
    #   output=Path("output"),
    # )
    # plot_results.lund_plane(all_hists=hists, path=output)
    # hists, output = run_toy(
    #    collision_system=collision_system,
    #    data_prefix=data_prefix,
    #    jet_pt_bins=jet_pt_bins,
    #    dataset_config_filename=Path("config") / "datasets.yaml",
    #    output=Path("output"),
    # )
    # plot_results.toy(all_toy_hists=hists, data_prefix=data_prefix, path=output)
    response_hists, matching_hists, output = run_embedding(
        collision_system=collision_system,
        jet_pt_bins=jet_pt_bins,
        dataset_config_filename=Path("config") / "datasets.yaml",
        output=Path("output"),
    )
    plot_results.responses(all_response_hists=response_hists, path=output)
    plot_results.matching(all_matching_hists=matching_hists, path=output)

    IPython.start_ipython(user_ns=locals())
