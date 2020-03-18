#!/usr/bin/env python3

""" Analyze the dynamical grooming tree.

.. codeauthor:: Raymnod Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, cast

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
    values: UprootArray = attr.ib()
    indices: UprootArray = attr.ib()
    subjet: substructure_methods.JetSplittingArray
    # TODO: Need to store iterative splitting information somehow!
    #       Perhaps just below...?

    @property
    def splitting_number(self) -> UprootArray:
        try:
            return self._splitting_number
        except AttributeError:
            # +1 because splittings counts from 1, but indexing starts from 0.
            splitting_number = self.indices + 1
            # If there were no splittings, we want to set that to 0.
            splitting_number = splitting_number.pad(1).fillna(0)
            # Must flatten because the indices are still jagged.
            self._splitting_number: UprootArray = splitting_number.flatten()

        return self._splitting_number

    @splitting_number.setter
    def splitting_number(self, value: UprootArray) -> None:
        self._splitting_number = value


def setup_yaml() -> yaml.ruamel.yaml.Yaml:
    return yaml.yaml(modules_to_register=[binned_data, analysis_objects, helpers])


def _convert_and_write_hists(
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]],
    tree_filename: Path,
    yaml_filename: Path,
    y: yaml.ruamel.yaml.Yaml,
) -> Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]:
    # Convert to BinnedData and store the hists
    for h in hists.values():
        h.convert_boost_histograms_to_binned_data()
    with open(yaml_filename, "w") as f:
        logger.info(f"Writing hists of the tree {tree_filename} to {yaml_filename}")
        y.dump(hists, f)

    return hists


def analyze_single_tree(
    tree: data_manager.Tree,
    z_cutoff: float,
    R: float,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    progress_manager: enlighten.Manager,
    y: yaml.ruamel.yaml.Yaml,
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
            prefix = "data"
            jets = substructure_methods.SubstructureJetArray.from_tree(tree, prefix=prefix)
            # Save calculate columns so we don't need to re-calculate them every time.
            # NOTE: We always check if they already exist because HDF5 doesn't like us
            #       overwriting columns.
            # Calculated subjet constituents.
            name = f"{prefix}.fSubjets.constituents"
            if name not in tree:
                tree[name] = jets.subjets.constituents

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
                jets, jet_pt_mask=identifier.jet_pt_bin.mask_array(jets.jet_pt), identifier=identifier
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

    return _convert_and_write_hists(hists=hists, tree_filename=tree.filename, yaml_filename=yaml_filename, y=y)


def _select_and_retrieve_splittings(
    jets: substructure_methods.SubstructureJetArray, jet_pt_mask: UprootArray, identifier: analysis_objects.Identifier
) -> Tuple[substructure_methods.SubstructureJetArray, substructure_methods.JetSplittingArray]:
    # Ensure that there are sufficient counts
    restricted_jets = jets[jet_pt_mask]
    if identifier.iterative_splittings:
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


def _get_leading_and_subleading_subjets(
    subjets_unsorted: substructure_methods.SubjetArray,
) -> Tuple[substructure_methods.SubjetArray, substructure_methods.SubjetArray]:
    subjets_pt_comparison = 1 - (
        subjets_unsorted[:, 0].constituents.pt.sum() > subjets_unsorted[:, 1].constituents.pt.sum() * 1
    )
    leading_indices = ak.JaggedArray.fromoffsets(range(len(subjets_pt_comparison) + 1), subjets_pt_comparison)
    subjets_leading = subjets_unsorted[leading_indices].flatten()
    subleading_indices = ak.JaggedArray.fromoffsets(range(len(subjets_pt_comparison) + 1), 1 - subjets_pt_comparison)
    subjets_subleading = subjets_unsorted[subleading_indices].flatten()

    return subjets_leading, subjets_subleading


def determine_matching_types():
    ...


def determine_matched_jets(
    hybrid_inputs: analysis_objects.FillHistogramInput, matched_inputs: analysis_objects.FillHistogramInput
) -> UprootArray:
    """

    The passed jets need to have the selected indices already applied.
    We need to work with the indices applied to these jets.

    """
    # Setup
    delta = 0.001
    try:
        selected_indices_mask = (
            matched_inputs.jets.subjets.parent_splitting_index.ones_like() * matched_inputs.indices.flatten()
        )
        matched_subjets_unsorted = matched_inputs.jets.subjets[
            selected_indices_mask == matched_inputs.jets.subjets.parent_splitting_index
        ]
        selected_indices_mask = (
            hybrid_inputs.jets.subjets.parent_splitting_index.ones_like() * hybrid_inputs.indices.flatten()
        )
        hybrid_subjets_unsorted = hybrid_inputs.jets.subjets[
            selected_indices_mask == hybrid_inputs.jets.subjets.parent_splitting_index
        ]
    except Exception as e:
        print(e)
        IPython.embed()
        exit(0)
    # hybrid_inputs.subjets.parent_splitting_index.ones_like() * hybrid_inputs.indices.flatten()
    # Sort the subjets such that 0 is always the leading subjet.
    # matched_subjets_pt_comparison = 1 - (matched_subjets_unsorted[:, 0].constituents.pt.sum() > matched_subjets_unsorted[:, 1].constituents.pt.sum() * 1)
    # matched_subjets_leading = matched_subjets_unsorted[matched_subjets_pt_comparison]
    # matched_subjets_subleading = matched_subjets_unsorted[1 - matched_subjets_pt_comparison]
    matched_subjets_leading, matched_subjets_subleading = _get_leading_and_subleading_subjets(matched_subjets_unsorted)
    hybrid_subjets_leading, hybrid_subjets_subleading = _get_leading_and_subleading_subjets(matched_subjets_unsorted)

    # This works...
    # In [119]: matched_subjets[:, 0].constituents.argcross(hybrid_subjets[:, 0].constituents)
    # Out[119]: <JaggedArray [[(0, 0) (0, 1) (0, 2) ... (10, 9) (10, 10) (10, 11)] [(0, 0) (0, 1) (0, 2) ... (16, 14) (16, 15) (16, 1
    # 6)] [(0, 0) (0, 1) (0, 2) ... (16, 9) (16, 10) (16, 11)] ... [(0, 0) (0, 1) (0, 2) ... (8, 13) (8, 14) (8, 15)] [(0, 0) (0, 1)
    # (0, 2) ... (8, 13) (8, 14) (8, 15)] [(0, 0) (0, 1) (0, 2) ... (8, 13) (8, 14) (8, 15)]] at 0x7f035ec61790>

    constituent_pairs = matched_subjets_leading.constituents.argcross(hybrid_subjets_leading.constituents)
    matched_leading_indices, hybrid_leading_indices = constituent_pairs.unzip()

    index_matching = (
        matched_subjets_leading.constituents[matched_leading_indices].global_index
        == hybrid_subjets_leading.constituents[hybrid_leading_indices].global_index
    )

    constituent_pts = matched_subjets_leading.constituents[matched_leading_indices][index_matching].pt.sum()

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

    if (constituent_pts > matched_inputs.jets.jet_pt).any():
        logger.warning("Constituent pts are greater than the jet pts...")
        IPython.embed()
        raise ValueError("Constituent pts are greater than the jet pts...")

    return (constituent_pts / matched_inputs.jets.jet_pt) > 0.5

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


def analyze_single_tree_embedding(
    tree: data_manager.Tree,
    z_cutoff: float,
    R: float,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    progress_manager: enlighten.Manager,
    y: yaml.ruamel.yaml.Yaml,
    output: Path,
    force_reprocessing: bool = False,
) -> Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]]:
    # Setup
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]] = {}
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
            ] = analysis_objects.create_substructure_response_hists(
                iterative_splittings=iterative_splittings, z_cutoff=z_cutoff
            )

    # Add a convenient wrapper.
    logger.debug(f"Accessing data from the tree {tree.filename}.")
    successfully_accessed_data = False
    try:
        # If there are 0 entries, then just return - it won't work...
        if len(tree) > 0:
            prefix = "data"
            hybrid_jets = substructure_methods.SubstructureJetArray.from_tree(tree, prefix=prefix)
            prefix = "matched"
            true_jets = substructure_methods.SubstructureJetArray.from_tree(tree, prefix=prefix)
            # prefix = "detLevel"
            # det_level_jets = substructure_methods.SubstructureJetArray.from_tree(tree, prefix=prefix)
            # for prefix, jets in [("data", hybrid_jets), ("matched", true_jets), ("detLevel", det_level_jets)]:
            for prefix, jets in [("data", hybrid_jets), ("matched", true_jets)]:
                # Save calculate columns so we don't need to re-calculate them every time.
                # NOTE: We always check if they already exist because HDF5 doesn't like us
                #       overwriting columns.
                # Calculated subjet constituents.
                name = f"{prefix}.fSubjets.constituents"
                if name not in tree:
                    tree[name] = jets.subjets.constituents

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
            # We want to restrict a constant hybrid jet pt range for both true and hybrid.
            # This will allow us to compare to measured jet pt ranges.
            jet_pt_mask = identifier.jet_pt_bin.mask_array(hybrid_jets.jet_pt)
            # Add additional restrictions that we can't handle single constituent jets.
            # TODO: Can we do better???
            jet_pt_mask = jet_pt_mask & (hybrid_jets.constituents.counts > 1) & (true_jets.constituents.counts > 1)
            restricted_hybrid_jets, restricted_hybrid_jets_splittings = _select_and_retrieve_splittings(
                hybrid_jets, jet_pt_mask, identifier
            )
            restricted_true_jets, restricted_true_jets_splittings = _select_and_retrieve_splittings(
                true_jets, jet_pt_mask, identifier
            )

            # TODO: What about additional cuts? Pt hard? etc
            weight = 1.0

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
            # hists[identifier].dynamical_z.fill(
            #    hybrid_inputs=hybrid_inputs, true_inputs=true_inputs, jet_R=R, weight=weight,
            # )
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
            determine_matched_jets(hybrid_inputs, true_inputs)
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

    # Convert to BinnedData and store the hists
    # for h in hists.values():
    #    h.convert_boost_histograms_to_binned_data()
    with open(yaml_filename.with_suffix(".pkl"), "wb") as f:
        import pickle

        pickle.dump(hists, f)
    # with open(yaml_filename, "w") as f:
    #    logger.info(f"Writing hists of the tree {tree.filename} to {yaml_filename}")
    #    IPython.embed()
    #    y.dump(hists, f)

    return hists


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
            tree_hists = analyze_single_tree_embedding(
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


def run_embedding(
    collision_system: str, jet_pt_bins: Sequence[helpers.RangeSelector], dataset_config_filename: Path, output: Path
) -> Tuple[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]], Path]:
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
    results: List[
        Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]]
    ] = []
    with progress_manager.counter(total=len(dm), desc="Analyzing", unit="tree") as tree_counter:
        for tree in tree_counter(dm):
            logger.info(f"Processing tree from file {tree.filename}")
            tree_hists = analyze_single_tree_embedding(
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
    full_hists: Dict[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]
    ] = {}
    for k in results[0].keys():
        full_hists[k] = cast(
            analysis_objects.Hists[analysis_objects.SubstructureResponseHists], sum([hists[k] for hists in results])
        )

    # Write out the merged hists
    with open(output / "response_hists.yaml", "w") as f:
        y.dump(full_hists, f)

    progress_manager.stop()

    return full_hists, output


# def compare_PbPb_to_embedded(pbpb_hists_path: Path, embedded_hists_path: Path) -> None:
#    # Setup
#    y = setup_yaml()
#    with open(pbpb_hists_path, "r") as f:
#        pbpb_hists = y.load(f)
#    with open(embedded_hists_path, "r") as f:
#        embedded_hists = y.load(f)
#    ...


if __name__ == "__main__":
    # Basic setup
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s")
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)
    # Quiet down BinndData copy warnings
    logging.getLogger("pachyderm.binned_data").setLevel(logging.INFO)

    # Setup and run
    collision_system = "toy"
    jet_pt_bins = [
        helpers.RangeSelector(min=60, max=80),
        helpers.RangeSelector(min=80, max=100),
        helpers.RangeSelector(min=100, max=120),
        # Most likely where we will actually measure.
        helpers.RangeSelector(min=80, max=120),
    ]
    hists, output = run(
        collision_system=collision_system,
        jet_pt_bins=jet_pt_bins,
        dataset_config_filename=Path("config") / "datasets.yaml",
        output=Path("output"),
    )

    plot_results.lund_plane(all_hists=hists, path=output)

    IPython.start_ipython(user_ns=locals())
