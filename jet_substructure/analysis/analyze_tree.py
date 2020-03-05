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
from pachyderm import yaml

from jet_substructure.base import data_manager, helpers, substructure_methods
from jet_substructure.base.analysis_objects import Hists, Identifier
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


def run(
    collision_system: str, jet_pt_bins: Sequence[helpers.RangeSelector], dataset_config_filename: Path, output: Path
) -> Tuple[Dict[Identifier, Hists], Path]:
    # Setup
    z_cutoff = 0.2
    # Configuration
    y = yaml.yaml()
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
    hists: Dict[Identifier, Hists] = {}
    for iterative_splittings in [False, True]:
        for jet_pt_bin in jet_pt_bins:
            hists[Identifier(iterative_splittings, jet_pt_bin)] = Hists.create_boost_histograms(
                iterative_splittings=iterative_splittings, z_cutoff=z_cutoff
            )

    # Iterate over trees.
    for i, tree in enumerate(dm.data_for_analysis(), start=1):
        logger.info(f"Processing tree {i}")
        # Add a convenient wrapper.
        jets = substructure_methods.SubstructureJetArray.from_tree(tree, prefix="data")

        # Loop over jet pt ranges
        for iterative_splittings in [False, True]:
            for jet_pt_bin in jet_pt_bins:
                jet_pt_mask = jet_pt_bin.mask_array(jets.jet_pt)
                restricted_jets = jets[jet_pt_mask]

                # TODO: Need deltaR, etc for _each_ selection!!
                # Fill the hists as appropriate
                values, indices = restricted_jets.dynamical_z(R=R)
                hists[Identifier(iterative_splittings, jet_pt_bin)].dynamical_z.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )
                values, indices = restricted_jets.dynamical_kt(R=R)
                hists[Identifier(iterative_splittings, jet_pt_bin)].dynamical_kt.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )
                values, indices = restricted_jets.dynamical_time(R=R)
                hists[Identifier(iterative_splittings, jet_pt_bin)].dynamical_time.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )
                values, indices = restricted_jets.leading_kt()
                hists[Identifier(iterative_splittings, jet_pt_bin)].leading_kt.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )
                values, indices = restricted_jets.leading_kt(z_cutoff=z_cutoff)
                hists[Identifier(iterative_splittings, jet_pt_bin)].leading_kt_hard_cutoff.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )

    return hists, output

    # Calculate the substructure variables
    # res = substructure.calculate_substructure_variables(arrays, R = 0.4, prefix = "data")
    # dynamical_z, dynamical_kt, dynamical_time, soft_drop, leading_kt, leading_kt_hard_cutoff = res

    # import IPython; IPython.embed()
    # TODO: Now: Plot the histograms (in a new function...)

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
    run(
        collision_system=collision_system,
        jet_pt_bins=jet_pt_bins,
        dataset_config_filename=Path("config") / "datasets.yaml",
        output=Path("output"),
    )
