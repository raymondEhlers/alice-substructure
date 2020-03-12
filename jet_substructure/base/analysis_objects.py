""" Main analysis objects.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import copy
import logging
from typing import TYPE_CHECKING, Generic, Iterator, List, Optional, Tuple, Type, TypeVar, Union

import attr
import boost_histogram as bh
import numpy as np
from pachyderm import binned_data

from jet_substructure.base import helpers
from jet_substructure.base.helpers import UprootArray


if TYPE_CHECKING:
    from jet_substructure.base import substructure_methods

logger = logging.getLogger(__name__)


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
class FillHistogramInput:
    jets: substructure_methods.SubstructureJetArray = attr.ib()
    _splittings: substructure_methods.JetSplittingArray = attr.ib()
    values: UprootArray = attr.ib()
    indices: UprootArray = attr.ib()

    @property
    def splittings(self) -> UprootArray:
        try:
            return self._restricted_splittings
        except AttributeError:
            self._restricted_splittings: substructure_methods.JetSplittingArray = self._splittings[self.indices]
        return self._restricted_splittings

    @property
    def n_jets(self) -> int:
        """ Number of jets.

        Need to determine all jets which are accepted in the jet pt range.
        Otherwise, those which may fail (such as with a z_cutoff) may not get
        the proper normalization.
        """
        return len(self.jets)


def _calculate_splitting_number(indices: UprootArray) -> UprootArray:
    # +1 because splittings counts from 1, but indexing starts from 0.
    splitting_number = indices + 1
    # If there were no splittings, we want to set that to 0.
    splitting_number = splitting_number.pad(1).fillna(0)
    # Must flatten because the indices are still jagged.
    splitting_number = splitting_number.flatten()
    return splitting_number


@attr.s
class SubstructureHistsBase:
    name: str = attr.ib()
    title: str = attr.ib()
    iterative_splittings: bool = attr.ib()

    @property
    def attributes_to_skip(self) -> List[str]:
        return ["name", "title", "iterative_splittings"]

    def __iter__(self) -> Iterator[Tuple[str, Union[bh.Histogram, binned_data.BinnedData]]]:
        return iter(
            {k: v for k, v in attr.asdict(self, recurse=False).items() if k not in self.attributes_to_skip}.items()
        )

    def convert_boost_histograms_to_binned_data(self) -> None:
        # Sanity check
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            raise ValueError("Not all hists are boost histograms! Cannot convert to binned data!")

        for k, v in self:
            setattr(self, k, binned_data.BinnedData.from_existing_data(v))


@attr.s
class SubstructureHists(SubstructureHistsBase):
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
    splitting_number_perturbative: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    lund_plane: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()

    @property
    def attributes_to_skip(self) -> List[str]:
        attrs = super().attributes_to_skip
        attrs.extend(["n_jets"])
        return attrs

    def __add__(self, other: "SubstructureHists") -> "SubstructureHists":
        """ Handles a = b + c """
        new = copy.deepcopy(self)
        new += other
        return new

    def __iadd__(self, other: "SubstructureHists") -> "SubstructureHists":
        """ Handles a += b """
        # Validation
        if self.iterative_splittings != other.iterative_splittings:
            raise TypeError(
                f"The types of splittings are different! self: {self.iterative_splittings}, other: {other.iterative_splittings}"
            )

        self.name = f"{self.name}_{other.name}" if self.name != other.name else self.name
        self.title = f"{self.title}_{other.title}" if self.title != other.title else self.title
        # Don't need to update iterative_splittings since they must be the same!
        self.n_jets += other.n_jets
        for (k, v), (k_other, v_other) in zip(self, other):
            v += v_other

        return self

    def __radd__(self, other: "SubstructureHists") -> "SubstructureHists":
        """ For use with sum(...). """
        if other == 0:
            return self
        else:
            return self + other

    def __truediv__(self, other: "SubstructureHists") -> "SubstructureHists":
        data = []
        for (k, v), (k_other, v_other) in zip(self, other):
            # Sanity check
            if k != k_other:
                raise ValueError(f"Somehow keys mismatch. self key: {k}, other key: {k_other}")
            # First, normalize the hists by the number of jets.
            temp_v = v / self.n_jets
            temp_v_other = v_other / other.n_jets
            data.append(temp_v / temp_v_other)

        return type(self)(
            f"{self.name}_{other.name}",
            f"{self.title}_{other.title}",
            self.iterative_splittings and other.iterative_splittings,
            1,
            *data,
        )

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
            splitting_number_perturbative=bh.Histogram(splitting_number_axis, storage=bh.storage.Weight()),
            lund_plane=bh.Histogram(*lund_plane_axes, storage=bh.storage.Weight()),
        )

    def fill(self, inputs: FillHistogramInput, jet_R: float, splitting_number: Optional[UprootArray] = None,) -> None:
        # Validation
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
            and isinstance(self.splitting_number_perturbative, bh.Histogram)
            and isinstance(self.lund_plane, bh.Histogram)
        )
        # Need to store the number of jets along the histograms.
        self.n_jets += inputs.n_jets
        self.values.fill(inputs.values)
        self.kt.fill(inputs.splittings.kt.flatten())
        self.z.fill(inputs.splittings.z.flatten())
        self.delta_R.fill(inputs.splittings.delta_R.flatten())
        self.theta.fill(inputs.splittings.delta_R.flatten() / jet_R)
        if splitting_number is None:
            splitting_number = _calculate_splitting_number(inputs.indices)
        self.splitting_number.fill(splitting_number)
        # Select only splittings with kt > 5.
        # +1 because splittings counts from 1, but indexing starts from 0.
        # NOTE: We aren't counting 0 here if it fails, so we aren't preserving counts!
        #       In this simpler case, we can just select directly on the indices.
        splitting_number_perturbative = (inputs.indices + 1)[inputs.splittings.kt > 5].flatten()
        self.splitting_number_perturbative.fill(splitting_number_perturbative)
        self.lund_plane.fill(np.log(1.0 / inputs.splittings.delta_R.flatten()), np.log(inputs.splittings.kt.flatten()))

        # Check the second peak in the z_cutoff recursive Lund Plane.
        if (np.log(1.0 / inputs.splittings.delta_R.flatten()) < 2).any() and (
            np.log(inputs.splittings.kt.flatten()) < -1.5
        ).any():
            # import IPython; IPython.embed()
            pass


@attr.s
class SubstructureResponseHists(SubstructureHistsBase):
    name: str = attr.ib()
    title: str = attr.ib()
    iterative_splittings: bool = attr.ib()
    n_hybrid_jets: int = attr.ib()
    n_true_jets: int = attr.ib()
    response_kt: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    response_z: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    response_delta_R: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    response_theta: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    response_splitting_number: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()

    @property
    def attributes_to_skip(self) -> List[str]:
        attrs = super().attributes_to_skip
        attrs.extend(["n_hybrid_jets", "n_true_jets"])
        return attrs

    def __add__(self, other: "SubstructureResponseHists") -> "SubstructureResponseHists":
        """ Handles a = b + c """
        new = copy.deepcopy(self)
        new += other
        return new

    def __iadd__(self, other: "SubstructureResponseHists") -> "SubstructureResponseHists":
        """ Handles a += b """
        # Validation
        if self.iterative_splittings != other.iterative_splittings:
            raise TypeError(
                f"The types of splittings are different! self: {self.iterative_splittings}, other: {other.iterative_splittings}"
            )

        self.name = f"{self.name}_{other.name}" if self.name != other.name else self.name
        self.title = f"{self.title}_{other.title}" if self.title != other.title else self.title
        # Don't need to update iterative_splittings since they must be the same!
        self.n_hybrid_jets += other.n_hybrid_jets
        self.n_true_jets += other.n_true_jets
        for (k, v), (k_other, v_other) in zip(self, other):
            v += v_other

        return self

    def __radd__(self, other: "SubstructureResponseHists") -> "SubstructureResponseHists":
        """ For use with sum(...). """
        if other == 0:
            return self
        else:
            return self + other

    @classmethod
    def create_boost_histograms(
        cls: Type["SubstructureResponseHists"], name: str, title: str, iterative_splittings: bool
    ) -> "SubstructureResponseHists":
        # TODO: Probably need different binning at true and particle level for variables.
        hybrid_jet_pt_axis = bh.axis.Regular(4, 40, 120)
        true_jet_pt_axis = bh.axis.Regular(8, 0, 160)
        kt_axis = bh.axis.Regular(50, 0, 25)
        z_axis = bh.axis.Regular(20, 0, 0.5)
        delta_R_axis = bh.axis.Regular(20, 0, 0.4)
        theta_axis = bh.axis.Regular(50, 0, 1)
        splitting_number_axis = bh.axis.Regular(10, 0, 10)
        return cls(
            name=name,
            title=title,
            iterative_splittings=iterative_splittings,
            n_hybrid_jets=0,
            n_true_jets=0,
            response_kt=bh.Histogram(
                hybrid_jet_pt_axis, kt_axis, true_jet_pt_axis, kt_axis, storage=bh.storage.Weight()
            ),
            response_z=bh.Histogram(hybrid_jet_pt_axis, z_axis, true_jet_pt_axis, z_axis, storage=bh.storage.Weight()),
            response_delta_R=bh.Histogram(
                hybrid_jet_pt_axis, delta_R_axis, true_jet_pt_axis, delta_R_axis, storage=bh.storage.Weight()
            ),
            response_theta=bh.Histogram(
                hybrid_jet_pt_axis, theta_axis, true_jet_pt_axis, theta_axis, storage=bh.storage.Weight()
            ),
            response_splitting_number=bh.Histogram(
                hybrid_jet_pt_axis,
                splitting_number_axis,
                true_jet_pt_axis,
                splitting_number_axis,
                storage=bh.storage.Weight(),
            ),
        )

    def fill(
        self, hybrid_inputs: FillHistogramInput, true_inputs: FillHistogramInput, weight: float, jet_R: float,
    ) -> None:
        # Validation
        # Give a useful error message
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            raise ValueError("Not all hists are boost histograms! Cannot fill!")

        # And then help out mypy...
        assert (
            isinstance(self.response_kt, bh.Histogram)
            and isinstance(self.response_z, bh.Histogram)
            and isinstance(self.response_delta_R, bh.Histogram)
            and isinstance(self.response_theta, bh.Histogram)
            and isinstance(self.response_splitting_number, bh.Histogram)
        )

        # Need to store the number of jets along the histograms.
        self.n_hybrid_jets += hybrid_inputs.n_jets
        self.n_true_jets += true_inputs.n_jets
        # Store the responses
        self.response_kt.fill(
            hybrid_inputs.jets.jet_pt,
            hybrid_inputs.splittings.kt.flatten(),
            true_inputs.jets.jet_pt,
            true_inputs.splittings.kt.flatten(),
            weight=weight,
        )
        self.response_z.fill(
            hybrid_inputs.jets.jet_pt,
            hybrid_inputs.splittings.kt.flatten(),
            true_inputs.jets.jet_pt,
            true_inputs.splittings.kt.flatten(),
            weight=weight,
        )
        self.response_delta_R.fill(
            hybrid_inputs.jets.jet_pt,
            hybrid_inputs.splittings.delta_R.flatten(),
            true_inputs.jets.jet_pt,
            true_inputs.splittings.delta_R.flatten(),
            weight=weight,
        )
        self.response_theta.fill(
            hybrid_inputs.jets.jet_pt,
            hybrid_inputs.splittings.theta.flatten() / jet_R,
            true_inputs.jets.jet_pt,
            true_inputs.splittings.theta.flatten() / jet_R,
            weight=weight,
        )
        self.response_splitting_number.fill(
            hybrid_inputs.jets.jet_pt,
            _calculate_splitting_number(hybrid_inputs.indices),
            true_inputs.jets.jet_pt,
            _calculate_splitting_number(true_inputs.indices),
            weight=weight,
        )


T_SubstructureHists = TypeVar("T_SubstructureHists", SubstructureHists, SubstructureResponseHists)


@attr.s
class Hists(Generic[T_SubstructureHists]):
    dynamical_z: T_SubstructureHists = attr.ib()
    dynamical_kt: T_SubstructureHists = attr.ib()
    dynamical_time: T_SubstructureHists = attr.ib()
    leading_kt: T_SubstructureHists = attr.ib()
    leading_kt_hard_cutoff: T_SubstructureHists = attr.ib()

    def __iter__(self) -> Iterator[Tuple[str, T_SubstructureHists]]:
        # We don't want to recurse because we have to handle the dict conversion more careful
        # for the SubstructureHists
        return iter(attr.asdict(self, recurse=False).items())

    def __add__(self, other: "Hists[T_SubstructureHists]") -> "Hists[T_SubstructureHists]":
        """ Handles a = b + c. """
        new = copy.deepcopy(self)
        new += other
        return new

    def __iadd__(self, other: "Hists[T_SubstructureHists]") -> "Hists[T_SubstructureHists]":
        """ Handles a += b. """
        # Add the stored values together.
        for (k, v), (k_other, v_other) in zip(self, other):
            # Validation
            if k != k_other:
                raise ValueError(f"Somehow keys mismatch. self key: {k}, other key: {k_other}")

            # Assumes that they are passed by reference.
            v += v_other

        return self

    def __radd__(self, other: "Hists[T_SubstructureHists]") -> "Hists[T_SubstructureHists]":
        """ For use with sum(...). """
        if other == 0:
            return self
        else:
            # Help out mypy
            assert not isinstance(other, int)
            return self + other

    def convert_boost_histograms_to_binned_data(self) -> None:
        for _, v in self:
            v.convert_boost_histograms_to_binned_data()


def create_substructure_hists(iterative_splittings: bool, z_cutoff: float) -> Hists[SubstructureHists]:
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
    return Hists(
        dynamical_z=dynamical_z,
        dynamical_kt=dynamical_kt,
        dynamical_time=dynamical_time,
        leading_kt=leading_kt,
        leading_kt_hard_cutoff=leading_kt_hard_cutoff,
    )


def create_substructure_response_hists(iterative_splittings: bool, z_cutoff: float) -> Hists[SubstructureResponseHists]:
    dynamical_z = SubstructureResponseHists.create_boost_histograms(
        name="dynamical_z_response", title="zDrop", iterative_splittings=iterative_splittings,
    )
    dynamical_kt = SubstructureResponseHists.create_boost_histograms(
        name="dynamical_kt_response", title="ktDrop", iterative_splittings=iterative_splittings
    )
    dynamical_time = SubstructureResponseHists.create_boost_histograms(
        name="dynamical_time_response", title="timeDrop", iterative_splittings=iterative_splittings,
    )
    leading_kt = SubstructureResponseHists.create_boost_histograms(
        name="leading_kt_response", title=r"Leading $k_{\text{T}}$", iterative_splittings=iterative_splittings,
    )
    leading_kt_hard_cutoff = SubstructureResponseHists.create_boost_histograms(
        name="leading_kt_hard_cutoff_response",
        title=fr"SD $z > {z_cutoff}$ Leading $k_{{\text{{T}}}}$",
        iterative_splittings=iterative_splittings,
    )

    # TODO: SD
    return Hists(
        dynamical_z=dynamical_z,
        dynamical_kt=dynamical_kt,
        dynamical_time=dynamical_time,
        leading_kt=leading_kt,
        leading_kt_hard_cutoff=leading_kt_hard_cutoff,
    )
