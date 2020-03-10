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
import coloredlogs
import enlighten
import IPython
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


def analyze_single_tree(
    tree: data_manager.Tree,
    z_cutoff: float,
    R: float,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    progress_manager: enlighten.Manager,
    y: yaml.ruamel.yaml.Yaml,
    output: Path,
    force_reprocessing: bool = False,
) -> Dict[analysis_objects.Identifier, analysis_objects.Hists]:
    # Setup
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists] = {}
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
            ] = analysis_objects.Hists.create_boost_histograms(
                iterative_splittings=iterative_splittings, z_cutoff=z_cutoff
            )

    # Add a convenient wrapper.
    logger.debug(f"Accessing data from the tree {tree.filename}.")
    try:
        jets = substructure_methods.SubstructureJetArray.from_tree(tree, prefix="data")
    except zlib.error as e:
        logger.warning(f"Issue reading the data: {e}. Skipping")
        # Return the empty hists
        return hists

    # Loop over iterations (jet pt ranges, iterative splitting)
    with progress_manager.counter(total=len(jet_pt_bins) * 2, desc="Analyzing", unit="variation") as variations:
        for iterative_splittings in [False, True]:
            for jet_pt_bin in jet_pt_bins:
                jet_pt_mask = jet_pt_bin.mask_array(jets.jet_pt)
                restricted_jets = jets[jet_pt_mask]
                if iterative_splittings:
                    # Only keep iterative splittings.
                    splittings = restricted_jets.splittings.iterative_splittings(restricted_jets.subjets)
                else:
                    splittings = restricted_jets.splittings

                # Fill the hists as appropriate
                values, indices = splittings.dynamical_z(R=R)
                hists[analysis_objects.Identifier(iterative_splittings, jet_pt_bin)].dynamical_z.fill(
                    values=values, indices=indices, splittings=splittings, jet_R=R
                )
                values, indices = splittings.dynamical_kt(R=R)
                hists[analysis_objects.Identifier(iterative_splittings, jet_pt_bin)].dynamical_kt.fill(
                    values=values, indices=indices, splittings=splittings, jet_R=R
                )
                values, indices = splittings.dynamical_time(R=R)
                hists[analysis_objects.Identifier(iterative_splittings, jet_pt_bin)].dynamical_time.fill(
                    values=values, indices=indices, splittings=splittings, jet_R=R
                )
                values, indices = splittings.leading_kt()
                hists[analysis_objects.Identifier(iterative_splittings, jet_pt_bin)].leading_kt.fill(
                    values=values, indices=indices, splittings=splittings, jet_R=R
                )
                values, indices = splittings.leading_kt(z_cutoff=z_cutoff)
                hists[analysis_objects.Identifier(iterative_splittings, jet_pt_bin)].leading_kt_hard_cutoff.fill(
                    values=values, indices=indices, splittings=splittings, jet_R=R
                )
                # Update progress
                variations.update()

    # Convert to BinnedData and store the hists
    for h in hists.values():
        h.convert_boost_histograms_to_binned_data()
    with open(yaml_filename, "w") as f:
        logger.info("Writing hists of the tree {tree.filename} to {yaml_filename}")
        y.dump(hists, f)

    return hists


def run(
    collision_system: str, jet_pt_bins: Sequence[helpers.RangeSelector], dataset_config_filename: Path, output: Path
) -> Tuple[Dict[analysis_objects.Identifier, analysis_objects.Hists], Path]:
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
    results: List[Dict[analysis_objects.Identifier, analysis_objects.Hists]] = []
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
            )
            # hists[tree.filename] = tree_hists
            results.append(tree_hists)

    # Merge the hists
    full_hists: Dict[analysis_objects.Identifier, analysis_objects.Hists] = {}
    for k in results[0].keys():
        full_hists[k] = cast(analysis_objects.Hists, sum([hists[k] for hists in results]))

    # Write out the merged hists
    with open(output / "hists.yaml", "w") as f:
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

    # Setup and run
    collision_system = "pythia"
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
