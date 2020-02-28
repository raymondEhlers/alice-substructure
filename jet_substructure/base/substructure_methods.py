# 1/usr/bin/env python3

""" Tests for Jet Substructure interpretation for uproot.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

import attr
import awkward as ak
import numpy as np


logger = logging.getLogger(__name__)

# Typing helpers
T = TypeVar("T")
UprootArray = Union[np.ndarray, ak.JaggedArray]
Result = Union[UprootArray, T]
# More ideally, I would like:
# It's supposed to carry the semantics of Union[np.ndarray, ak.JaggedArray]
# class UprootArray(Generic[T]): ...
# Result = Union[UprootArray[T], T]


def _dynamical_hardness_measure(
    delta_R: Result[float], z: Result[float], parent_pt: Result[float], R: float, a: float
) -> Result[float]:
    return z * (1 - z) * parent_pt * (delta_R / R) ** a


dynamical_z = functools.partial(_dynamical_hardness_measure, a=0.1)
dynamical_kt = functools.partial(_dynamical_hardness_measure, a=1.0)
dynamical_time = functools.partial(_dynamical_hardness_measure, a=2.0)


def find_leading(values: Result[float]) -> Tuple[Result[float], Result[float]]:
    """ Calculate hardest value.

    Used for dynamical grooming, hardest kt, etc.

    Returns:
        Leading value, index of value.
    """
    arg_max = values.argmax()
    return values[arg_max], arg_max


class ArrayMethods(ak.Methods):  # type: ignore
    # Seems to be required for creating JaggedArray elements within an Array.
    # Otherwise, it will create one object per event, with that object storing arrays of members.
    awkward = ak

    def _trymemo(self, name: str, function: Callable[..., T]) -> Callable[..., T]:
        memoname = "_memo_" + name
        wrap, (array,) = ak.util.unwrap_jagged(type(self), self.JaggedArray, (self,))
        if not hasattr(array, memoname):
            setattr(array, memoname, function(array))
        return cast(Callable[..., T], wrap(getattr(array, memoname)))


@attr.s
class JetConstituent:
    """ Jet constituent.

    Args:
        pt: Jet constituent pt.
        eta: Jet constituent eta.
        phi: Jet constituent phi.
        global_index: Global index assigned to the track during analysis. The index is unique
            for each event.
    """

    pt: float = attr.ib()
    eta: float = attr.ib()
    phi: float = attr.ib()
    _global_index: int = attr.ib()

    @property
    def index(self) -> int:
        return self._global_index


class JetConstituentArrayMethods(ArrayMethods):
    """ Methods for operating on jet constituents arrays.

    These methods operate on externally stored arrays.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """ Create jet constituent views in a table.

        Args:
            table: Table where the constituents will be created.
        """
        self.awkward.ObjectArray.__init__(
            self, table, lambda row: JetConstituent(row["pt"], row["eta"], row["phi"], row["global_index"])
        )

    @property
    def pt(self) -> Result[float]:
        """ Constituents pt. """
        return self["pt"]

    @property
    def eta(self) -> Result[float]:
        """ Constituents eta. """
        return self["eta"]

    @property
    def phi(self) -> Result[float]:
        """ Constituents phi. """
        return self["phi"]

    @property
    def index(self) -> Result[int]:
        """ Constituents global index. """
        return self["global_index"]


# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedJetConstituentArrayMethods = JetConstituentArrayMethods.mixin(JetConstituentArrayMethods, ak.JaggedArray)


class JetConstituentArray(JetConstituentArrayMethods, ak.ObjectArray):  # type:ignore
    """ Array of jet constituents.

    This effectively constructs a virtual object that can operate transparently on arrays.

    Args:
        pt: Array of constituent pt.
        eta: Array of constituent eta.
        phi: Array of constituent phi.
        global_index: Array of constituent global indices.
    """

    def __init__(self, pt: Result[float], eta: Result[float], phi: Result[float], global_index: Result[int]) -> None:
        self._init_object_array(ak.Table())
        self["pt"] = pt
        self["eta"] = eta
        self["phi"] = phi
        self["global_index"] = global_index

    @property
    def max_pt(self) -> Result[float]:
        return self["pt"].max()

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedJetConstituentArrayMethods)  # type: ignore
    def from_jagged(
        cls: Type[T], pt: Result[Result[float]], eta: Result[float], phi: Result[float], global_index: Result[int]
    ) -> T:
        """ Creates a view of constituents with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        Args:
            pt: Jagged constituent pt.
            eta: Jagged constituent eta.
            phi: Jagged constituent phi.
            global_index: Jagged constituent global index.
        """
        return cls(pt, eta, phi, global_index)  # type: ignore


@attr.s
class Subjet:
    """ Subjet found within a jet.

    Args:
        part_of_iterative_splitting: True if the subjet is part of the iterative splitting.
        parent_splitting_index: Index of the splitting which lead to this subjet.
        constituents_indices: Indices of the constituents that are contained within this subjet.
            This is indexed by the number of constituents in a jet (i.e. it is not the global index!).
    """

    part_of_iterative_splitting: bool = attr.ib()
    _parent_splitting_index: int = attr.ib()
    _constituents_indices: Result[int] = attr.ib()

    @property
    def parent_splitting_index(self) -> int:
        return self._parent_splitting_index

    def parent_splitting(self, splittings: Result[Sequence[JetSplitting]]) -> Result[JetSplitting]:
        """ Retrieve the parent splitting of this subjet.

        Args:
            splittings: All of the splittings from the overall jet.
        Returns:
            Splitting which led to this subjet.
        """
        return splittings[self._parent_splitting_index]

    def constituents(self, jet_constituents: Result[Sequence[JetConstituent]]) -> Result[Sequence[JetConstituent]]:
        """ Retrieve the constituents of this subjet.

        Args:
            jet_constituents: Constituents of the overall jet.
        Returns:
            Constituents of this subjet.
        """
        return jet_constituents[self._constituents_indices]
        # return ak.JaggedArray.fromoffsets(self._constituents_indices.offsets, jet_constituents[s.flatten()])
        # return ak.JaggedArray.fromoffsets(
        #    self._constituents_indices.offsets, ak.JaggedArray.fromoffsets(
        #        self._constituents_indices.offsets, jet_constituents.flatten()[self._constituents_indices.flatten(axis=1)]
        #    )
        # )

    # constituents[subjets._constituents_indices.flatten(axis=1)]

    # def test():
    #    ak.JaggedArray.fromoffsets(
    #        subjets._constituents_indices.offsets, ak.JaggedArray.fromoffsets(
    #            subjets._constituents_indices.flatten().offsets,
    #            constituents[subjets._constituents_indices.flatten(axis=1)].flatten()
    #        )
    #    )


def _convert_jagged_constituents_indicies(
    constituents_indices: ak.JaggedArray, jagged_indices: ak.JaggedArray
) -> ak.JaggedArray:
    return ak.fromiter(
        (ak.JaggedArray.fromoffsets(jagged, indices) for jagged, indices in zip(jagged_indices, constituents_indices))
    )


class SubjetArrayMethods(ArrayMethods):
    """ Methods for operating on subjet arrays.

    These methods operate on externally stored arrays.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """ Create jet constituent views in a table.

        Args:
            table: Table where the constituents will be created.
        """
        self.awkward.ObjectArray.__init__(
            self,
            table,
            lambda row: Subjet(
                row["part_of_iterative_splitting"], row["parent_splitting_index"], row["constituents_indices"]
            ),
        )

    @property
    def _constituents_indices(self) -> Result[int]:
        """ Construct constituent indices from stored JaggedArrays.

        We create this property just to make it slightly easier to access. Normally,
        one would just want the constituents directly, so we make it private.

        Note:
            It's currently not possible to directly create doubly Jagged arrays. We unfortunately have to accept
            a loop in python to create it.
        """
        return self["constituents_indices"]

    @property
    def part_of_iterative_splitting(self) -> Result[bool]:
        """ Whether subjets are part of the iterative splitting. """
        return self["part_of_iterative_splitting"]

    @property
    def parent_splitting_index(self) -> Result[int]:
        """ Index of the parent splittings. """
        return self["parent_splitting_index"]

    @property
    def iterative_splitting_index(self) -> Result[int]:
        """ Indices of splittings which were part of the iterative splitting chain. """
        return self.parent_splitting_index[self.part_of_iterative_splitting]

    def parent_splitting(self, splittings: Result[Sequence[JetSplitting]]) -> Result[JetSplitting]:
        """ Retrieve the parent splittings for the subjets.

        Args:
            splittings: Splittings corresponding to the subjets.
        Returns:
            Parent splittings for the subjets.
        """
        return splittings[self["parent_splitting_index"]]

    def constituents(self, jet_constituents: Result[Sequence[JetConstituent]]) -> Result[Sequence[JetConstituent]]:
        """ Constituents of the subjets.

        Args:
            jet_constituents: Constituents of the overall jets which contain the subjets.
        Returns:
            Jet constituents of the subjets.
        """
        return self._try_memo(
            "constituents",
            lambda self: ak.JaggedArray.fromoffsets(
                self._constituents_indices.offsets,
                ak.JaggedArray.fromoffsets(
                    self._constituents_indices.flatten().offsets,
                    jet_constituents[self._constituents_indices.flatten(axis=1)].flatten(),
                ),
            ),
        )
        # return jet_constituents[self._constituents_indices]
        # This doesn't seem super efficient, but I can't seem to broadcast it directly.
        # return ak.JaggedArray.fromoffsets(
        #    self._constituents_indices.offsets, ak.JaggedArray.fromoffsets(
        #        self._constituents_indices.flatten().offsets,
        #        jet_constituents[self._constituents_indices.flatten(axis=1)].flatten()
        #    )
        # )


# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedSubjetArrayMethods = SubjetArrayMethods.mixin(SubjetArrayMethods, ak.JaggedArray)

_T_SubjetArray = TypeVar("_T_SubjetArray", bound="SubjetArray")


class SubjetArray(SubjetArrayMethods, ak.ObjectArray):  # type: ignore
    """ Array of subjets.

    This effectively constructs a virtual object that can operate transparently on arrays.

    Args:
        part_of_iterative_splitting: True if the given subjet is part of the iterative splitting.
        parent_splitting_index: Index of the parent splitting of the subjets.
        constituents_indices: Indices of the constituents of the subjets.
    """

    def __init__(
        self,
        part_of_iterative_splitting: Result[bool],
        parent_splitting_index: Result[int],
        constituents_indices: Result[Result[int]],
    ) -> None:
        self._init_object_array(ak.Table())
        self["part_of_iterative_splitting"] = part_of_iterative_splitting
        self["parent_splitting_index"] = parent_splitting_index
        self["constituents_indices"] = constituents_indices

    @classmethod
    def from_jagged(
        cls: Type[_T_SubjetArray],
        part_of_iterative_splitting: Result[bool],
        parent_splitting_index: Result[int],
        constituents_indices: Result[int],
        constituents_jagged_indices: Optional[Result[int]] = None,
    ) -> _T_SubjetArray:
        """ Creates a view of subjets with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        Note:
            We need this additional wrapper because we can't construct doubly jagged arrays directly, and the jaggedness
            needs to be the same for all parts for of the array. We work around this by constructing the doubly jagged
            array before constructing the `ObjectArray`. See: https://github.com/scikit-hep/uproot/issues/452.

        Args:
            part_of_iterative_splitting: Jagged iterative splitting label.
            parent_splitting_index: Jagged parent splitting index.
            constituents_indices: Jagged constituents indices of the subjet.
        """
        if constituents_jagged_indices:
            constituents_indices = _convert_jagged_constituents_indicies(
                constituents_indices, constituents_jagged_indices
            )
        else:
            constituents_indices = ak.fromiter(constituents_indices)

        return cast(
            _T_SubjetArray,
            cls._from_jagged_impl(part_of_iterative_splitting, parent_splitting_index, constituents_indices),
        )

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedSubjetArrayMethods)  # type: ignore
    def _from_jagged_impl(
        cls: Type[_T_SubjetArray],
        part_of_iterative_splitting: Result[bool],
        parent_splitting_index: Result[int],
        constituents_indices: Result[int],
    ) -> _T_SubjetArray:
        """ Creates a view of subjets with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        This is a bit of a pain to call directly, so it's hidden behind the wrapper defined above.

        Args:
            part_of_iterative_splitting: Jagged iterative splitting label.
            parent_splitting_index: Jagged parent splitting index.
            constituents_indices: Jagged constituents indices of the subjet.
        """
        return cls(part_of_iterative_splitting, parent_splitting_index, constituents_indices)


@attr.s
class JetSplitting:
    """ Properties of a jet splitting.

    Args:
        kt: Kt of the subjets.
        delta_R: Delta R between the subjets.
        z: Z of the softer subjet.
        parent_index: Index of the parent subjet.
    """

    kt: float = attr.ib()
    delta_R: float = attr.ib()
    z: float = attr.ib()
    _parent_index: int = attr.ib()

    def part_of_iterative_splitting(self, subjets: SubjetArray) -> bool:
        """ Determine whether the splitting is iterative.

        Args:
            subjets: Subjets of the overall jet which containing the iterative splitting information.
        Returns:
            True if the splitting is part of the iterative splitting chain.
        """
        # Determine the parent index of the splittings which are iterative.
        # This indexes the splittings, so we then apply it to the object.
        iterative_splittings = subjets.parent_splitting_index[subjets.part_of_iterative_splitting]
        return self._parent_index in iterative_splittings

    @property
    def parent_pt(self) -> Result[float]:
        """ pt of the parent subjet. """
        # parent_pt = subleading / z = kt / sin(delta_R) / z
        return self.kt / np.sin(self.delta_R) / self.z

    def dynamical_z(self, R: float) -> float:
        return dynamical_z(self.delta_R, self.z, self.parent_pt, R)

    def dynamical_kt(self, R: float) -> float:
        return dynamical_kt(self.delta_R, self.z, self.parent_pt, R)

    def dynamical_time(self, R: float) -> float:
        return dynamical_time(self.delta_R, self.z, self.parent_pt, R)


class JetSplittingArrayMethods(ArrayMethods):
    """ Methods for operating on jet splittings arrays.

    These methods operate on externally stored arrays.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """ Create jet splitting views in a table.

        Args:
            table: Table where the constituents will be created.
        """
        self.awkward.ObjectArray.__init__(
            self, table, lambda row: JetConstituent(row["kt"], row["delta_R"], row["z"], row["parent_index"])
        )

    @property
    def kt(self) -> Result[float]:
        return self["kt"]

    @property
    def delta_R(self) -> Result[float]:
        return self["delta_R"]

    @property
    def z(self) -> Result[float]:
        return self["z"]

    def part_of_iterative_splitting(self, subjets: Result[SubjetArray]) -> Result[bool]:
        # iterative_splittings = subjets.parent_splitting_index[subjets.part_of_iterative_splitting]
        iterative_splittings = subjets.iterative_splitting_index
        return self["parent_index"] in iterative_splittings

    def iterative_splittings(self, subjets: Result[SubjetArray]) -> Result[SubjetArray]:
        """ Retrieve iterative splittings. """
        return self[subjets.iterative_splitting_index]

    def dynamical_z(self, R: float) -> Tuple[Result[float], Result[float]]:
        """ Dynamical z of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical z values, leading dynamical z indices.
        """
        return find_leading(dynamical_z(self.delta_R, self.z, self.parent_pt, R))

    def dynamical_kt(self, R: float) -> Tuple[Result[float], Result[float]]:
        """ Dynamical kt of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical kt values, leading dynamical kt indices.
        """
        return find_leading(dynamical_kt(self.delta_R, self.z, self.parent_pt, R))

    def dynamical_time(self, R: float) -> Tuple[Result[float], Result[float]]:
        """ Dynamical time of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical time values, leading dynamical time indices.
        """
        return find_leading(dynamical_time(self.delta_R, self.z, self.parent_pt, R))

    def leading_kt(self, z_cutoff: Optional[float] = None) -> Tuple[Result[float], Result[float]]:
        """ Leading kt of the jet splittings.

        Args:
            z_cutoff: Z cutoff to be applied before calculating the leading kt.
        Returns:
            Leading kt values, leading kt indices.
        """
        mask = slice(None)
        if z_cutoff is not None:
            mask = self.z > z_cutoff
        return find_leading(self.kt[mask])

    def soft_drop(self, z_cutoff: float) -> Tuple[Result[float], Result[int], Result[float]]:
        """ Calculate soft drop of the splittings.

        Args:
            z_cutoff: Minimum z for Soft Drop.
        Returns:
            First z passing cutoff (z_g), number of splittings passing SD (n_sd), index of z passing cutoff.
        """
        z_cutoff_mask = self.z > z_cutoff
        # We use :1 because this maintains the jagged structure. That way, we can apply it to initial arrays.
        z_index = self.z.localindex[z_cutoff_mask][:, :1]
        z_g = self.z[z_index].flatten()
        n_sd = self.z[z_cutoff_mask].count_nonzero()

        return z_g, n_sd, z_index


# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedJetSplittingArrayMethods = JetSplittingArrayMethods.mixin(JetSplittingArrayMethods, ak.JaggedArray)


class JetSplittingArray(JetSplittingArrayMethods, ak.ObjectArray):  # type: ignore
    """ Array of jet splittings.

    This effectively constructs a virtual object that can operate transparently on arrays.

    Args:
        kt: Kt of the jet splittings.
        delta_R: Delta R of the subjets.
        z: Momentum fraction of the softer subjet.
        parent_index: Index of the parent splitting.
    """

    def __init__(self, kt: Result[float], delta_R: Result[float], z: Result[float], parent_index: Result[int]) -> None:
        self._init_object_array(ak.Table())
        self["kt"] = kt
        self["delta_R"] = delta_R
        self["z"] = z
        self["parent_index"] = parent_index

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedJetSplittingArrayMethods)  # type: ignore
    def from_jagged(
        cls: Type[T], kt: Result[float], delta_R: Result[float], z: Result[float], parent_index: Result[int]
    ) -> T:
        """ Creates a view of constituents with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        Args:
            kt: Jagged kt.
            delta_R: Jagged delta R.
            z: Jagged z.
            parent_index: Jagged parent splitting index.
        """
        return cls(kt, delta_R, z, parent_index)  # type: ignore


class SubstructureJetCommonMethods:
    """ Common methods for jet substructure methods.

    Note:
        These only work if properties have the same names in both the single and array classes.
    """

    if TYPE_CHECKING:
        constituents: Union[Result[JetConstituentArray], JetConstituentArray]
        splittings: Union[Result[JetSplittingArray], JetSplittingArray]

    @property
    def leading_track_pt(self) -> float:
        """ Leading track pt. """
        return self.constituents.max_pt

    def dynamical_z(self, R: float) -> Tuple[float, float]:
        return self.splittings.dynamical_z(R=R)

    def dynamical_kt(self, R: float) -> Tuple[float, float]:
        return self.splittings.dynamical_kt(R=R)

    def dynamical_time(self, R: float) -> Tuple[float, float]:
        return self.splittings.dynamical_time(R=R)

    def leading_kt(self, z_cutoff: Optional[float] = None) -> Tuple[float, float]:
        """ Leading kt. """
        return self.splittings.leading_kt(z_cutoff=z_cutoff)

    def soft_drop_kt(self, z_cutoff: float) -> Tuple[float, int, float]:
        """ Calculate soft drop of the splittings.

        Args:
            z_cutoff: Minimum z for Soft Drop.
        Returns:
            First z passing cutoff (z_g), number of splittings passing SD (n_sd), index of z passing cutoff.
        """
        return self.splittings.soft_drop(z_cutoff=z_cutoff)


@attr.s
class SubstructureJet(SubstructureJetCommonMethods):
    """ Substructure of a jet.

    Args:
        jet_pt: Jet pt.
        constituents: Jet constituents.
        subjets: Subjets.
        splittings: Jet splittings.
    """

    jet_pt: float = attr.ib()
    constituents: JetConstituentArray = attr.ib()
    subjets: SubjetArray = attr.ib()
    splittings: JetSplittingArray = attr.ib()


class SubstructureJetArrayMethods(SubstructureJetCommonMethods, ArrayMethods):
    def _init_object_array(self, table: ak.Table) -> None:
        self.awkward.ObjectArray.__init__(
            self,
            table,
            lambda row: SubstructureJet(row["jet_pt"], row["constituents"], row["subjets"], row["splittings"]),
        )

    @property
    def jet_pt(self) -> Result[float]:
        return self["jet_pt"]

    @property
    def constituents(self) -> Result[JetConstituentArray]:
        return self["constituents"]

    @property
    def subjets(self) -> Result[SubjetArray]:
        return self["subjets"]

    @property
    def splittings(self) -> Result[JetSplittingArray]:
        return self["splittings"]


# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedSubstructureJetArrayMethods = SubstructureJetArrayMethods.mixin(SubstructureJetArrayMethods, ak.JaggedArray)


class SubstructureJetArray(SubstructureJetArrayMethods, ak.ObjectArray):  # type: ignore
    """ Array of substructure jets.

    Note:
        This can't support a `from_jagged(...)` method because the contained arrays have different
        jaggedness. The overlay expanding into the jagged dimension only works if they have the same
        jaggedness. However, we can still create one object per element in the array (ie. for each jet).
    """

    def __init__(
        self,
        jet_pt: Result[float],
        jet_constituents: Result[JetConstituentArray],
        subjets: Result[SubjetArray],
        jet_splittings: Result[JetSplittingArray],
    ) -> None:
        self._init_object_array(ak.Table())
        self["jet_pt"] = jet_pt
        self["constituents"] = jet_constituents
        self["subjets"] = subjets
        self["splittings"] = jet_splittings

    @classmethod
    def from_tree(cls: Type[T], tree: ak.JaggedArray, prefix: str) -> T:
        """ Construct from a tree.

        Args:
            tree: Tree containing the splittings.
            prefix: Prefix under which the branches are stored.
        Returns:
            Substructure jet array wrapping all of the arrays.
        """
        logger.debug("Creating substructure jet arrays.")
        constituents = JetConstituentArray.from_jagged(
            tree[f"{prefix}.fJetConstituents.fPt"],
            tree[f"{prefix}.fJetConstituents.fEta"],
            tree[f"{prefix}.fJetConstituents.fPhi"],
            tree[f"{prefix}.fJetConstituents.fGlobalIndex"],
        )
        logger.debug("Done with constructing constituents")
        splittings = JetSplittingArray.from_jagged(
            tree[f"{prefix}.fJetSplittings.fKt"],
            tree[f"{prefix}.fJetSplittings.fDeltaR"],
            tree[f"{prefix}.fJetSplittings.fZ"],
            tree[f"{prefix}.fJetSplittings.fParentIndex"],
        )
        logger.debug("Done with constructing splittings")
        subjets = SubjetArray.from_jagged(
            tree[f"{prefix}.fSubjets.fPartOfIterativeSplitting"],
            tree[f"{prefix}.fSubjets.fSplittingNodeIndex"],
            tree[f"{prefix}.fSubjets.fConstituentIndices"],
            tree.get(f"{prefix}.fSubjets.fConstituentJaggedIndices", None),
        )
        logger.debug("Done with constructing subjets.")

        # Construct substructure jets using the above
        return cls(  # type: ignore
            tree[f"{prefix}.fJetPt"], constituents, subjets, splittings,
        )
