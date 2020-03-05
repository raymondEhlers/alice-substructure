""" Main analysis objects.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import TYPE_CHECKING, Iterator, Optional, Tuple, Type, Union

import attr
import boost_histogram as bh
import numpy as np
from pachyderm import binned_data

from jet_substructure.base import helpers
from jet_substructure.base.helpers import UprootArray


if TYPE_CHECKING:
    from jet_substructure.base import substructure_methods


@attr.s(frozen=True)
class Identifier:
    iterative_splittings: bool = attr.ib()
    jet_pt_bin: helpers.RangeSelector = attr.ib()

    @property
    def iterative_splittings_label(self) -> str:
        return "iterative" if self.iterative_splittings else "recursive"

    def __str__(self) -> str:
        return f"jetPt_{self.jet_pt_bin.min}_{self.jet_pt_bin.max}_{self.iterative_splittings_label}_splittings"

    def display_str(self) -> str:
        return f"{self.iterative_splittings_label.capitalize()} splittings\n${self.jet_pt_bin.display_str()}$"


@attr.s
class SubstructureHists:
    name: str = attr.ib()
    title: str = attr.ib()
    iterative_splittings: bool = attr.ib()
    n_jets: int = attr.ib()
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
            n_jets=0,
            values=bh.Histogram(values_axis, storage=bh.storage.Weight()),
            kt=bh.Histogram(kt_axis, storage=bh.storage.Weight()),
            z=bh.Histogram(z_axis, storage=bh.storage.Weight()),
            delta_R=bh.Histogram(delta_R_axis, storage=bh.storage.Weight()),
            theta=bh.Histogram(theta_axis, storage=bh.storage.Weight()),
            splitting_number=bh.Histogram(splitting_number_axis, storage=bh.storage.Weight()),
            lund_plane=bh.Histogram(*lund_plane_axes, storage=bh.storage.Weight()),
        )

    def __iter__(self) -> Iterator[Tuple[str, Union[bh.Histogram, binned_data.BinnedData]]]:
        return iter(
            {
                k: v
                for k, v in attr.asdict(self).items()
                if k not in ["name", "title", "iterative_splittings", "n_jets"]
            }.items()
        )

    def convert_boost_histograms_to_binned_data(self) -> None:
        # Sanity check
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            raise ValueError("Not all hists are boost histograms! Cannot convert to binned data!")

        for k, v in self:
            setattr(self, k, binned_data.BinnedData.from_boost_histogram(v))

    def fill(
        self,
        values: UprootArray,
        indices: UprootArray,
        splittings: "substructure_methods.JetSplittingArray",
        jet_R: float,
        splitting_number: Optional[UprootArray] = None,
    ) -> None:
        # Give a useful error message
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            raise ValueError("Not all hists are boost histograms! Cannot fill!")
        # And then help out mypy...
        assert (
            isinstance(self.values, bh.Histogram)
            and isinstance(self.kt, bh.Histogram)
            and isinstance(self.z, bh.Histogram)
            and isinstance(self.delta_R, bh.Histogram)
            and isinstance(self.theta, bh.Histogram)
            and isinstance(self.splitting_number, bh.Histogram)
            and isinstance(self.lund_plane, bh.Histogram)
        )
        # Need to store the number of jets along the histograms.
        self.n_jets += len(values)
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

    def __iter__(self) -> Iterator[Tuple[str, SubstructureHists]]:
        # We don't want to recurse because we have to handle the dict conversion more careful
        # for the SubstructureHists
        return iter(attr.asdict(self, recurse=False).items())

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
