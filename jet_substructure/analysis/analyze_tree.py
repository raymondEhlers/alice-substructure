#!/usr/bin/env python3

""" Analyze the dynamical grooming tree.

.. codeauthor:: Raymnod Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple

import attr
import coloredlogs
import enlighten
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

    # Define hists
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists] = {}
    for iterative_splittings in [False, True]:
        for jet_pt_bin in jet_pt_bins:
            hists[
                analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
            ] = analysis_objects.Hists.create_boost_histograms(
                iterative_splittings=iterative_splittings, z_cutoff=z_cutoff
            )

    logger.info("Setup complete. Beginning processing of trees.")

    # Iterate over trees.
    progress_manager = enlighten.get_manager()
    with progress_manager.counter(total=len(dm), desc="Analyzing", unit="tree") as tree_counter:
        for tree in tree_counter(dm):
            logger.info(f"Processing tree from file {tree.filename}")
            # Add a convenient wrapper.
            jets = substructure_methods.SubstructureJetArray.from_tree(tree, prefix="data")

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
                        hists[
                            analysis_objects.Identifier(iterative_splittings, jet_pt_bin)
                        ].leading_kt_hard_cutoff.fill(values=values, indices=indices, splittings=splittings, jet_R=R)
                        # Update progress
                        variations.update()

            # Convert to BinnedData and store the hists
            for h in hists.values():
                for _, technique_hists in h:
                    technique_hists.convert_boost_histograms_to_binned_data()
            with open(output / tree.filename.with_suffix(".yaml").name, "w") as f:
                y.dump(hists, f)

    progress_manager.stop()

    return hists, output

    # Calculate the substructure variables
    # res = substructure.calculate_substructure_variables(arrays, R = 0.4, prefix = "data")
    # dynamical_z, dynamical_kt, dynamical_time, soft_drop, leading_kt, leading_kt_hard_cutoff = res

    # import IPython; IPython.embed()
    # TODO: Store the hists!

    # plot_results.kt(results = res, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)
    # plot_results.z(results = res, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)
    # plot_results.delta_R(results = res, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)
    # plot_results.theta(results = res, jet_R=0.4, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)
    # plot_results.splitting_number(results = res, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)


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
