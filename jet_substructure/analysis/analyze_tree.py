#!/usr/bin/env python3

""" Analyze the dynamical grooming tree.

.. codeauthor:: Raymnod Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Type, Union

import attr
import boost_histogram as bh
import coloredlogs
import numpy as np
from pachyderm import binned_data, yaml

from jet_substructure.base import data_manager, helpers, substructure_methods
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


@attr.s
class SubstructureHists:
    name: str = attr.ib()
    title: str = attr.ib()
    iterative_splittings: bool = attr.ib()
    values: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    kt: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    z: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    delta_R: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    theta: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    splitting_number: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    lund_plane: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()

    @classmethod
    def create_boost_histograms(
        cls: Type["SubstructureHists"], name: str, title: str, iterative_splittings: bool, values_axis: bh.Histogram
    ) -> "SubstructureHists":
        kt_axis = bh.axis.Regular(50, 0, 25)
        z_axis = bh.axis.Regular(20, 0, 0.5)
        delta_R_axis = bh.axis.Regular(20, 0, 0.4)
        theta_axis = bh.axis.Regular(50, 0, 1)
        splitting_number_axis = bh.axis.Regular(10, 0, 10)
        lund_plane_axes = [bh.axis.Regular(25, 0, 5), bh.axis.Regular(25, -5.0, 5.0)]
        return cls(
            name=name,
            title=title,
            iterative_splittings=iterative_splittings,
            values=bh.Histogram(values_axis, storage=bh.storage.Weight()),
            kt=bh.Histogram(kt_axis, storage=bh.storage.Weight()),
            z=bh.Histogram(z_axis, storage=bh.storage.Weight()),
            delta_R=bh.Histogram(delta_R_axis, storage=bh.storage.Weight()),
            theta=bh.Histogram(theta_axis, storage=bh.storage.Weight()),
            splitting_number=bh.Histogram(splitting_number_axis, storage=bh.storage.Weight()),
            lund_plane=bh.Histogram(*lund_plane_axes, storage=bh.storage.Weight()),
        )

    def fill(
        self,
        values: UprootArray,
        indices: UprootArray,
        jets: substructure_methods.SubstructureJetArray,
        jet_R: float,
        splitting_number: Optional[UprootArray] = None,
    ) -> None:
        # For convenience
        if self.iterative_splittings:
            splittings = jets.splittings.iterative_splittings(jets.subjets)[indices]
        else:
            splittings = jets.splittings[indices]

        # Help out mypy...
        assert (
            isinstance(self.values, bh.Histogram)
            and isinstance(self.kt, bh.Histogram)
            and isinstance(self.z, bh.Histogram)
            and isinstance(self.delta_R, bh.Histogram)
            and isinstance(self.theta, bh.Histogram)
            and isinstance(self.splitting_number, bh.Histogram)
            and isinstance(self.lund_plane, bh.Histogram)
        )
        self.values.fill(values)
        self.kt.fill(splittings.kt.flatten())
        self.z.fill(splittings.z.flatten())
        self.delta_R.fill(splittings.delta_R.flatten())
        self.theta.fill(splittings.delta_R.flatten() / jet_R)
        if splitting_number is None:
            # +1 because splittings counts from 1, but indexing starts from 0.
            splitting_number = indices + 1
            # If there were no splittings, we want to set that to 0.
            splitting_number = splitting_number.pad(1).fillna(0).flatten()
            # Must flatten because the indices are still jagged.
            splitting_number = splitting_number.flatten()
        self.splitting_number.fill(splitting_number)
        self.lund_plane.fill(np.log(1.0 / splittings.delta_R.flatten()), np.log(splittings.kt.flatten()))


@attr.s
class Hists:
    dynamical_z: SubstructureHists = attr.ib()
    dynamical_kt: SubstructureHists = attr.ib()
    dynamical_time: SubstructureHists = attr.ib()
    leading_kt: SubstructureHists = attr.ib()
    leading_kt_hard_cutoff: SubstructureHists = attr.ib()

    def __iter__(self) -> Iterator[Tuple[str, Union[bh.Histogram, binned_data.BinnedData]]]:
        return iter(attr.asdict(self).items())

    def __iadd__(self, other: "Hists") -> "Hists":
        for (k, v), (k_other, v_other) in zip(self, other):
            # Sanity check.
            assert k == k_other
            setattr(self, k, v + v_other)
        return self

    @classmethod
    def create_boost_histograms(cls: Type["Hists"], iterative_splittings: bool, z_cutoff: float) -> "Hists":
        kt_axis = bh.axis.Regular(50, 0, 25)
        dynamical_z = SubstructureHists.create_boost_histograms(
            name="dynamical_z",
            title="zDrop",
            iterative_splittings=iterative_splittings,
            values_axis=bh.axis.Regular(50, 0, 50),
        )
        dynamical_kt = SubstructureHists.create_boost_histograms(
            name="dynamical_kt", title="ktDrop", iterative_splittings=iterative_splittings, values_axis=kt_axis,
        )
        dynamical_time = SubstructureHists.create_boost_histograms(
            name="dynamical_time",
            title="timeDrop",
            iterative_splittings=iterative_splittings,
            values_axis=bh.axis.Regular(50, 0, 50),
        )
        leading_kt = SubstructureHists.create_boost_histograms(
            name="leading_kt",
            title=r"Leading $k_{\text{T}}$",
            iterative_splittings=iterative_splittings,
            values_axis=kt_axis,
        )
        leading_kt_hard_cutoff = SubstructureHists.create_boost_histograms(
            name="leading_kt_hard_cutoff",
            title=fr"SD $z > {z_cutoff}$ Leading $k_{{\text{{T}}}}$",
            iterative_splittings=iterative_splittings,
            values_axis=kt_axis,
        )

        # TODO: SD
        return cls(
            dynamical_z=dynamical_z,
            dynamical_kt=dynamical_kt,
            dynamical_time=dynamical_time,
            leading_kt=leading_kt,
            leading_kt_hard_cutoff=leading_kt_hard_cutoff,
        )


@attr.s(frozen=True)
class Index:
    iterative_splittings: bool = attr.ib()
    jet_pt_bin: helpers.RangeSelector = attr.ib()


def run(collision_system: str, dataset_config_filename: Path, output: Path) -> None:
    # Setup
    jet_pt_bins = [
        helpers.RangeSelector(min=60, max=80),
        helpers.RangeSelector(min=80, max=100),
        helpers.RangeSelector(min=100, max=120),
        # Most likely where we will actually measure.
        helpers.RangeSelector(min=80, max=120),
    ]
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
    hists: Dict[Index, Hists] = {}
    for iterative_splittings in [False, True]:
        for jet_pt_bin in jet_pt_bins:
            hists[Index(iterative_splittings, jet_pt_bin)] = Hists.create_boost_histograms(
                iterative_splittings=iterative_splittings, z_cutoff=z_cutoff
            )

    # Iterate over trees.
    for tree in dm.data_for_analysis():
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
                hists[Index(iterative_splittings, jet_pt_bin)].dynamical_z.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )
                values, indices = restricted_jets.dynamical_kt(R=R)
                hists[Index(iterative_splittings, jet_pt_bin)].dynamical_kt.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )
                values, indices = restricted_jets.dynamical_time(R=R)
                hists[Index(iterative_splittings, jet_pt_bin)].dynamical_time.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )
                values, indices = restricted_jets.leading_kt()
                hists[Index(iterative_splittings, jet_pt_bin)].leading_kt.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )
                values, indices = restricted_jets.leading_kt(z_cutoff=z_cutoff)
                hists[Index(iterative_splittings, jet_pt_bin)].leading_kt_hard_cutoff.fill(
                    values=values, indices=indices, jets=restricted_jets, jet_R=R
                )

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
    run(
        collision_system=collision_system,
        dataset_config_filename=Path("config") / "datasets.yaml",
        output=Path("output"),
    )
