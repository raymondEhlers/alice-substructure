#!/usr/bin/env python3

""" Tests for Jet Substructure interpretation for uproot.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import functools
import logging
import typing
from typing import TYPE_CHECKING, Any, Callable, Collection, Optional, Tuple, Type, TypeVar, Union, cast

import attr
import awkward as ak
import numpy as np

#from jet_substructure.base.helpers import UprootArray


logger = logging.getLogger(__name__)

# Typing helpers
T = TypeVar("T")
class UprootArrayTyped(Collection[T], Any):  # type: ignore

    @typing.overload
    def __getitem__(self, key: UprootArrayTyped[bool]) -> UprootArrayTyped[T]: ...

    @typing.overload
    def __getitem__(self, key: UprootArrayTyped[int]) -> UprootArrayTyped[T]: ...

    @typing.overload
    def __getitem__(self, key: bool) -> T: ...

    @typing.overload
    def __getitem__(self, key: int) -> T: ...

    def __getitem__(self, key):  # type: ignore
        raise NotImplementedError("Just typing information.")

    @typing.overload
    def __truediv__(self, other: float) -> UprootArrayTyped[T]: ...

    @typing.overload
    def __truediv__(self, other: UprootArrayTyped[T]) -> UprootArrayTyped[T]: ...

    def __truediv__(self, other):  # type: ignore
        raise NotImplementedError("Just typing information.")


ArrayOrScalar = Union[UprootArrayTyped[T], T]
#Result = Union[UprootArray, T]
# More ideally, I would like:
# It's supposed to carry the semantics of Union[np.ndarray, ak.JaggedArray]
# class NDArray(Generic[T]):
#    def argmax(self) -> T: ...
# class UprootArrayTyped(Generic[T]):
#    def argmax(self) -> NDArray[int]: ...
#    @overload
#    def __getitem__(self, key: NDArray[bool]) -> Union[UprootArrayTyped[T], NDArray[T]]: ...
#    @overload
#    def __getitem__(self, key: NDArray[int]) -> NDArray[T]: ...

# UprootArrayTyped = Union[UprootArray[T], T]

@typing.overload
def _dynamical_hardness_measure(
    delta_R: UprootArrayTyped[float], z: UprootArrayTyped[float], parent_pt: UprootArrayTyped[float], R: float, a: float
) -> UprootArrayTyped[float]: ...

@typing.overload
def _dynamical_hardness_measure(
    delta_R: float, z: float, parent_pt: float, R: float, a: float
) -> float: ...

def _dynamical_hardness_measure(delta_R, z, parent_pt, R, a):  # type: ignore
    return z * (1 - z) * parent_pt * (delta_R / R) ** a


dynamical_z = functools.partial(_dynamical_hardness_measure, a=0.1)
dynamical_kt = functools.partial(_dynamical_hardness_measure, a=1.0)
dynamical_time = functools.partial(_dynamical_hardness_measure, a=2.0)


def find_leading(values: UprootArrayTyped[T]) -> Tuple[np.ndarray, UprootArrayTyped[T]]:
    """ Calculate hardest value given a set of values.

    Used for dynamical grooming, hardest kt, etc.

    Returns:
        Leading value, index of value.
    """
    arg_max = values.argmax()
    return values[arg_max].flatten(), arg_max


class ArrayMethods(ak.Methods):  # type: ignore
    """ Base class containing methods for use in awkward `ObjectArray`s. """
    # Seems to be required for creating JaggedArray elements within an Array.
    # Otherwise, it will create one object per event, with that object storing arrays of members.
    awkward = ak

    def _try_memo(self, name: str, function: Callable[..., T]) -> Callable[..., T]:
        """ Try to memorize the result of a function so it doesn't need to be recalculated.

        It unwraps a layer of jaggedness, so some care is required for using it.

        Note:
            This is taken from `uproot-methods`.
        """
        memoname = "_memo_" + name
        wrap, (array,) = ak.util.unwrap_jagged(type(self), self.JaggedArray, (self,))
        if not hasattr(array, memoname):
            setattr(array, memoname, function(array))
        return cast(Callable[..., T], wrap(getattr(array, memoname)))


@attr.s
class JetConstituent:
    """ A single jet constituent.

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

    def delta_R(self, other: "JetConstituent") -> float:
        return cast(float, np.sqrt((self.phi - other.phi) ** 2 + (self.eta - other.eta) ** 2))


class JetConstituentArrayMethods(ArrayMethods):
    """ Methods for operating on jet constituents arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
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
    def pt(self) -> UprootArrayTyped[float]:
        """ Constituent pts. """
        return cast(UprootArrayTyped[float], self["pt"])

    @property
    def eta(self) -> UprootArrayTyped[float]:
        """ Constituent etas. """
        return cast(UprootArrayTyped[float], self["eta"])

    @property
    def phi(self) -> UprootArrayTyped[float]:
        """ Constituent phis. """
        return cast(UprootArrayTyped[float], self["phi"])

    @property
    def index(self) -> UprootArrayTyped[int]:
        """ Constituent global indices. """
        return cast(UprootArrayTyped[int], self["global_index"])

    @property
    def max_pt(self) -> ArrayOrScalar[float]:
        """ Maximum pt of the stored constituent. """
        return cast(ArrayOrScalar[float], self["pt"].max())

    def delta_R(self, other: "JetConstituentArray") -> UprootArrayTyped[float]:
        """ Delta R between one set of constituents and the others. """
        return cast(UprootArrayTyped[float], np.sqrt((self["phi"] - other["phi"]) ** 2 + (self["eta"] - other["eta"]) ** 2))


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

    def __init__(self, pt: UprootArrayTyped[float], eta: UprootArrayTyped[float], phi: UprootArrayTyped[float], global_index: UprootArrayTyped[int]) -> None:
        self._init_object_array(ak.Table())
        self["pt"] = pt
        self["eta"] = eta
        self["phi"] = phi
        self["global_index"] = global_index

    def __awkward_serialize__(self, serializer: ak.persist.Serializer) -> ak.persist.Serializer:
        """ Serialize to storage. """
        self._valid()
        pt, eta, phi, global_index = self.pt, self.eta, self.phi, self.global_index
        return serializer.encode_call(
            ["jet_substructure.base.substructure_methods", "JetConstituentArray", "from_jagged"],
            serializer(pt, "JetConstituentArray.pt"),
            serializer(eta, "JetConstituentArray.eta"),
            serializer(phi, "JetConstituentArray.phi"),
            serializer(global_index, "JetConstituentArray.global_index"),
        )

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedJetConstituentArrayMethods)  # type: ignore
    def from_jagged(
        cls: Type[T], pt: UprootArrayTyped[UprootArrayTyped[float]], eta: UprootArrayTyped[UprootArrayTyped[float]],
        phi: UprootArrayTyped[UprootArrayTyped[float]], global_index: UprootArrayTyped[UprootArrayTyped[int]]
    ) -> T:
        """ Creates a view of constituents with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it goes a step
        further in, constructing virtual objects in the jagged structure. Basically, it unwarps the first layer of
        jaggedness using a wrapper.

        Args:
            pt: Jagged constituent pt.
            eta: Jagged constituent eta.
            phi: Jagged constituent phi.
            global_index: Jagged constituent global index.

        Returns:
            Jet constituents array acting within the jagged array contents.
        """
        return cls(pt, eta, phi, global_index)  # type: ignore


@attr.s
class Subjet:
    """ Subjet found within a jet.

    Args:
        part_of_iterative_splitting: True if the subjet is part of the iterative splitting.
        parent_splitting_index: Index of the splitting which lead to this subjet.
        constituents: Constituents which are contained within this subjet.
    """

    part_of_iterative_splitting: bool = attr.ib()
    _parent_splitting_index: int = attr.ib()
    _constituents: JetConstituentArray = attr.ib()

    @classmethod
    def from_constituents_indices(
        cls: Type["Subjet"],
        part_of_iterative_splitting: bool,
        parent_splitting_index: int,
        constituent_indices: UprootArrayTyped[int],
        jet_constituents: JetConstituentArray,
    ) -> "Subjet":
        """ Construct the subjet from the constituent indices and jet constituents.

        Note:
            This is helpful for the case where the subjet constituents aren't directly available. This can be rather
            common because we don't save the entire constituents for each subjet, as this would be incredibly wasteful
            in terms of storage and retrieving from the grid. However, we can generate and store them later in analysis
            to save processing time during analysis when storage space is less critical.

        Args:
            part_of_iterative_splitting: True if the subjet is part of the iterative splitting.
            parent_splitting_index: Index of the splitting which lead to this subjet.
            constituents_indices: Indices of the constituents that are contained within this subjet.
                This is indexed by the number of constituents in a jet (i.e. it is not the global index!).
            jet_constituents: Jet constituents which are indexed by their number in the jet.
        """
        return cls(part_of_iterative_splitting, parent_splitting_index, jet_constituents[constituent_indices],)

    @property
    def parent_splitting_index(self) -> int:
        """ Index of the parent splitting which produced the subjet. """
        return self._parent_splitting_index

    @typing.overload
    def parent_splitting(self, splittings: UprootArrayTyped[JetSplittingArray]) -> JetSplittingArray: ...

    @typing.overload
    def parent_splitting(self, splittings: JetSplittingArray) -> JetSplitting: ...

    def parent_splitting(self, splittings):  # type: ignore
        """ Retrieve the parent splitting of this subjet.

        Args:
            splittings: All of the splittings from the overall jet.
        Returns:
            Splitting which led to this subjet.
        """
        return splittings[self._parent_splitting_index]

    @property
    def constituents(self) -> JetConstituentArray:
        """ Constituents of this subjet. """
        return self._constituents


def _convert_jagged_constituents_indices(
    constituents_indices: ak.JaggedArray, jagged_indices: ak.JaggedArray
) -> ak.JaggedArray:
    """ Convert constituents indices and jagged indices into a doubly jagged constituents array.

    That array can then be used for determining which constituents belong to which subjets.

    Args:
        constituents_indices: Jagged array containing all of the constituents indices of the subjets.
        jagged_indices: Jagged array containing the offsets in the constituents_indices for each individual subjet.

    Returns:
        Doubly jagged indices specifying the constituents in each subjet.
    """
    return ak.fromiter(
        (ak.JaggedArray.fromoffsets(jagged, indices) for jagged, indices in zip(jagged_indices, constituents_indices))
    )


class SubjetArrayMethods(ArrayMethods):
    """ Methods for operating on subjet arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """ Create jet constituent views in a table.

        Args:
            table: Table where the subjets will be created.
        Returns:
            None.
        """
        self.awkward.ObjectArray.__init__(
            self,
            table,
            lambda row: Subjet(row["part_of_iterative_splitting"], row["parent_splitting_index"], row["constituents"]),
        )

    def __awkward_serialize__(self, serializer: ak.persist.Serializer) -> ak.persist.Serializer:
        """ Serialize to storage. """
        self._valid()
        part_of_iterative_splitting, parent_splitting_index, constituents = (
            self.part_of_iterative_splitting,
            self.parent_splitting_index,
            self.constituents,
        )
        return serializer.encode_call(
            ["jet_substructure.base.substructure_methods", "SubjetArrayMethods", "from_jagged"],
            serializer(part_of_iterative_splitting, "SubjetArrayMethods.part_of_iterative_splitting"),
            serializer(parent_splitting_index, "SubjetArrayMethods.parent_splitting_index"),
            serializer(constituents, "SubjetArrayMethods.constituents"),
        )

    @property
    def part_of_iterative_splitting(self) -> UprootArrayTyped[bool]:
        """ Whether subjets are part of the iterative splitting.

        Args:
            None.
        Returns:
            True if the subjets are part of the iterative splitting.
        """
        return cast(UprootArrayTyped[bool], self["part_of_iterative_splitting"])

    @property
    def parent_splitting_index(self) -> UprootArrayTyped[int]:
        """ Indices of the parent splittings. """
        return cast(UprootArrayTyped[int], self["parent_splitting_index"])

    @property
    def iterative_splitting_index(self) -> UprootArrayTyped[int]:
        """ Indices of splittings which were part of the iterative splitting chain. """
        return self.parent_splitting_index[self.part_of_iterative_splitting]

    def parent_splitting(self, splittings: UprootArrayTyped[JetSplittingArray]) -> UprootArrayTyped[JetSplittingArray]:
        """ Retrieve the parent splittings of the subjets.

        Args:
            splittings: Splittings which may have produced the subjets.
        Returns:
            Parent splittings for the subjets.
        """
        return cast(UprootArrayTyped[JetSplittingArray], splittings[self["parent_splitting_index"]])


# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedSubjetArrayMethods = SubjetArrayMethods.mixin(SubjetArrayMethods, ak.JaggedArray)

_T_SubjetArray = TypeVar("_T_SubjetArray", bound="SubjetArray")


class SubjetArray(SubjetArrayMethods, ak.ObjectArray):  # type: ignore
    """ Array of subjets.

    This effectively constructs a virtual object that can operate transparently on arrays.

    Args:
        part_of_iterative_splitting: True if the given subjet is part of the iterative splitting.
        parent_splitting_index: Indices of the parent splitting of the subjets.
        constituents: Constituents which are contained within the subjets.
    """

    def __init__(
        self,
        part_of_iterative_splitting: UprootArrayTyped[bool],
        parent_splitting_index: UprootArrayTyped[int],
        constituents: UprootArrayTyped[JetConstituentArray],
    ) -> None:
        self._init_object_array(ak.Table())
        self["part_of_iterative_splitting"] = part_of_iterative_splitting
        self["parent_splitting_index"] = parent_splitting_index
        self["constituents"] = constituents

    @classmethod
    def from_jagged(
        cls: Type[_T_SubjetArray],
        part_of_iterative_splitting: UprootArrayTyped[bool],
        parent_splitting_index: UprootArrayTyped[int],
        constituents_indices: UprootArrayTyped[int],
        subjet_constituents: Optional[UprootArrayTyped[JetConstituentArray]] = None,
        jet_constituents: Optional[UprootArrayTyped[JetConstituentArray]] = None,
        constituents_jagged_indices: Optional[UprootArrayTyped[int]] = None,
    ) -> _T_SubjetArray:
        """ Creates a view of subjets with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        Note:
            We need this additional wrapper because we can't construct doubly jagged arrays directly, and the jaggedness
            needs to be the same for all parts for of the array. We work around this by constructing the doubly jagged
            array before constructing the `ObjectArray`. See: https://github.com/scikit-hep/uproot/issues/452.

        Note:
            The caller may pass either the `subjet_constituents` or the `jet_constituents`. If the `subjet_constituents`
            are passed, they take precedence over everything else.

        Args:
            part_of_iterative_splitting: Jagged iterative splitting label.
            parent_splitting_index: Jagged parent splitting index.
            constituents_indices: Jagged constituents indices of the subjet.
            subjet_constituents: Subjet constituents. These are eventually constructed by this object, but
                can be saved afterwards to avoid having to reconstruct them again. If they're passed in,
                the rest of the constituents arguments are ignored and these constituents are used. Default: None.
            jet_constituents: Jet constituents. Used in conjunction withe the constituents_indices to determine
                which constituents belong in which subjets. Default: None.
            constituents_jagged_indices: Constituents jagged indices used to convert the constituents_indices into
                doubly jagged indices if they are not already. Default: None.
        Returns:
            Subjet array acting within the jagged array contents.
        """
        # Validation
        if subjet_constituents is None and constituents_indices is None:
            raise ValueError("Must pass subjet constituents or constituents indices.")
        logger.debug("Determining subjet constituents.")

        # We have three modes for creating the indices:
        # 1) The subjet constituents have already been determined in the past. Just use them.
        # 2) Construct the constituent indices using separately stored jagged indices.
        # 3) Construct the doubly jagged indices stored in the tree via fromiter(...)
        # In the case of 2 or 3, the subjets constituents are determine from the jet constituents.
        if subjet_constituents is not None:
            logger.debug("Using pre-calculated constituents.")
            pass
        else:
            if constituents_jagged_indices is not None:
                # Calculate the indices.
                logger.debug("Constructing constituents indices from manually stored jagged indices.")
                constituents_indices = _convert_jagged_constituents_indices(
                    constituents_indices, constituents_jagged_indices
                )
            else:
                # Construct the indices.
                logger.debug(f"Constructing constituents indices from doubly jagged indices.")
                constituents_indices = ak.fromiter(constituents_indices)

            # Help out mypy
            assert jet_constituents is not None

            # This doesn't seem super efficient, but I can't seem to broadcast it directly.
            logger.debug("Calculating subjets constituents from constituents indices.")
            subjet_constituents = ak.JaggedArray.fromoffsets(
                constituents_indices.offsets,
                ak.JaggedArray.fromoffsets(
                    constituents_indices.flatten().offsets,
                    jet_constituents[constituents_indices.flatten(axis=1)].flatten(),
                ),
            )

        return cast(
            _T_SubjetArray,
            cls._from_jagged_impl(part_of_iterative_splitting, parent_splitting_index, subjet_constituents),
        )

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedSubjetArrayMethods)  # type: ignore
    def _from_jagged_impl(
        cls: Type[_T_SubjetArray],
        part_of_iterative_splitting: UprootArrayTyped[bool],
        parent_splitting_index: UprootArrayTyped[int],
        constituents: UprootArrayTyped[JetConstituentArray],
    ) -> _T_SubjetArray:
        """ Creates a view of subjets with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        This is a bit of a pain to call directly, so it's hidden behind the wrapper defined above.

        Args:
            part_of_iterative_splitting: Jagged iterative splitting label.
            parent_splitting_index: Jagged parent splitting index.
            constituents: Constituents of the subjet.
        Returns:
            Subjet array acting within the jagged array contents.
        """
        return cls(part_of_iterative_splitting, parent_splitting_index, constituents)


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
    def parent_pt(self) -> float:
        """ Pt of the (parent) subjet which lead to the splitting.

        The pt can be calculated from the splitting properties via:

        parent_pt = subleading / z = kt / sin(delta_R) / z

        Args:
            None.
        Returns:
            None.
        """
        # parent_pt = subleading / z = kt / sin(delta_R) / z
        return cast(float, self.kt / np.sin(self.delta_R) / self.z)

    def theta(self, jet_R: float) -> float:
        """ Theta of the splitting.

        This is defined as delta_R normalized by the jet resolution parameter.

        Args:
            jet_R: Jet resolution parameter.
        Returns:
            Theta of the splitting.
        """
        return self.delta_R / jet_R

    def dynamical_z(self, R: float) -> float:
        """ Dynamical z of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical z of the splitting.
        """
        return dynamical_z(self.delta_R, self.z, self.parent_pt, R)

    def dynamical_kt(self, R: float) -> float:
        """ Dynamical kt of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical kt of the splitting.
        """
        return dynamical_kt(self.delta_R, self.z, self.parent_pt, R)

    def dynamical_time(self, R: float) -> float:
        """ Dynamical time of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical time of the splitting.
        """
        return dynamical_time(self.delta_R, self.z, self.parent_pt, R)


class JetSplittingArrayMethods(ArrayMethods):
    """ Methods for operating on jet splittings arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """ Create jet splitting views in a table.

        Args:
            table: Table where the jet splittings will be created.
        Returns:
            None.
        """
        self.awkward.ObjectArray.__init__(
            self, table, lambda row: JetSplitting(row["kt"], row["delta_R"], row["z"], row["parent_index"])
        )

    def __awkward_serialize__(self, serializer: ak.persist.Serializer) -> ak.persist.Serializer:
        """ Serialize to storage. """
        self._valid()
        kt, delta_R, z, parent_index = self.kt, self.delta_R, self.z, self.parent_index
        return serializer.encode_call(
            ["jet_substructure.base.substructure_methods", "JetSplittingArrayMethods", "from_jagged"],
            serializer(kt, "JetSplittingArrayMethods.kt"),
            serializer(delta_R, "JetSplittingArrayMethods.delta_R"),
            serializer(z, "JetSplittingArrayMethods.z"),
            serializer(parent_index, "JetSplittingArrayMethods.parent_index"),
        )

    @property
    def kt(self) -> UprootArrayTyped[float]:
        """ Kt of the splittings. """
        return cast(UprootArrayTyped[float], self["kt"])

    @property
    def delta_R(self) -> UprootArrayTyped[float]:
        """ Delta R of the splittings. """
        return cast(UprootArrayTyped[float], self["delta_R"])

    @property
    def z(self) -> UprootArrayTyped[float]:
        """ z of the splitting. """
        return cast(UprootArrayTyped[float], self["z"])

    def part_of_iterative_splitting(self, subjets: UprootArrayTyped[SubjetArray]) -> UprootArrayTyped[bool]:
        """ Determine whether the splitting is iterative.

        Args:
            subjets: Subjets of the jets which containing the iterative splitting information.
        Returns:
            True if the splittings are part of the iterative splitting chain.
        """
        # TODO: I don't think this works!!
        # iterative_splittings = subjets.parent_splitting_index[subjets.part_of_iterative_splitting]
        iterative_splittings = subjets.iterative_splitting_index
        return cast(UprootArrayTyped[bool], self["parent_index"] in iterative_splittings)

    def iterative_splittings(self, subjets: UprootArrayTyped[SubjetArray]) -> UprootArrayTyped[SubjetArray]:
        """ Retriieve the iterative splittings.

        Args:
            subjets: Subjets of the jets which containing the iterative splitting information.
        Returns:
            The splittings which are part of the iterative splitting chain.
        """
        return cast(UprootArrayTyped[SubjetArray], self[subjets.iterative_splitting_index])

    @property
    def parent_pt(self) -> UprootArrayTyped[float]:
        """ Pt of the (parent) subjets which lead to the splittings.

        The pt can be calculated from the splitting properties via:

        parent_pt = subleading / z = kt / sin(delta_R) / z

        Args:
            None.
        Returns:
            None.
        """
        # parent_pt = subleading / z = kt / sin(delta_R) / z
        return cast(UprootArrayTyped[float], self.kt / np.sin(self.delta_R) / self.z)

    def theta(self, jet_R: float) -> UprootArrayTyped[float]:
        """ Theta of the splittings.

        This is defined as delta_R normalized by the jet resolution parameter.

        Args:
            jet_R: Jet resolution parameter.
        Returns:
            Theta of the splittings.
        """
        return self.delta_R / jet_R

    def dynamical_z(self, R: float) -> Tuple[UprootArrayTyped[float], UprootArrayTyped[float]]:
        """ Dynamical z of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical z values, leading dynamical z indices.
        """
        return find_leading(dynamical_z(self.delta_R, self.z, self.parent_pt, R))

    def dynamical_kt(self, R: float) -> Tuple[UprootArrayTyped[float], UprootArrayTyped[float]]:
        """ Dynamical kt of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical kt values, leading dynamical kt indices.
        """
        return find_leading(dynamical_kt(self.delta_R, self.z, self.parent_pt, R))

    def dynamical_time(self, R: float) -> Tuple[UprootArrayTyped[float], UprootArrayTyped[float]]:
        """ Dynamical time of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical time values, leading dynamical time indices.
        """
        return find_leading(dynamical_time(self.delta_R, self.z, self.parent_pt, R))

    def leading_kt(self, z_cutoff: Optional[float] = None) -> Tuple[UprootArrayTyped[float], UprootArrayTyped[float]]:
        """ Leading kt of the jet splittings.

        Args:
            z_cutoff: Z cutoff to be applied before calculating the leading kt.
        Returns:
            Leading kt values, leading kt indices.
        """
        # Need to use the local index because we are going to mask z values. If we index from the masked
        # z values, it it is applied to the unmasked array later, it will give nonsense. So we mask the local index,
        # find the leading, and then apply that index back to the local index, which then gives us the leading index
        # in the unmasked array.
        local_index_mask = self.z.localindex
        if z_cutoff is not None:
            local_index_mask = self.z.localindex[self.z > z_cutoff]
        values, indices = find_leading(self.kt[local_index_mask])
        return values, local_index_mask[indices]

    def soft_drop(self, z_cutoff: float) -> Tuple[UprootArrayTyped[float], UprootArrayTyped[int], UprootArrayTyped[float]]:
        """ Calculate soft drop of the splittings.

        Args:
            z_cutoff: Minimum z for Soft Drop.
        Returns:
            First z passing cutoff (z_g), number of splittings passing SD (n_sd), index of z passing cutoff.
        """
        z_cutoff_mask = cast(UprootArrayTyped[bool], self.z > z_cutoff)
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

    def __init__(self, kt: UprootArrayTyped[float], delta_R: UprootArrayTyped[float], z: UprootArrayTyped[float], parent_index: UprootArrayTyped[int]) -> None:
        self._init_object_array(ak.Table())
        self["kt"] = kt
        self["delta_R"] = delta_R
        self["z"] = z
        self["parent_index"] = parent_index

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedJetSplittingArrayMethods)  # type: ignore
    def from_jagged(
        cls: Type[T], kt: UprootArrayTyped[float], delta_R: UprootArrayTyped[float], z: UprootArrayTyped[float], parent_index: UprootArrayTyped[int]
    ) -> T:
        """ Creates a view of splittings with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        Args:
            kt: Jagged kt.
            delta_R: Jagged delta R.
            z: Jagged z.
            parent_index: Jagged parent splitting index.
        Returns:
            Splittings array acting on the jagged array contents.
        """
        return cls(kt, delta_R, z, parent_index)  # type: ignore


class SubstructureJetCommonMethods:
    """ Common methods for jet substructure methods.

    Note:
        These only work if properties have the same names in both the single and array classes.
    """

    if TYPE_CHECKING:
        _constituents: ArrayOrScalar[JetConstituentArray]
        _subjets: ArrayOrScalar[SubjetArray]
        _splittings: ArrayOrScalar[JetSplittingArray]

    @property
    def constituents(self) -> ArrayOrScalar[JetConstituentArray]:
        return self._constituents

    @property
    def subjets(self) -> ArrayOrScalar[SubjetArray]:
        return self._subjets

    @property
    def splittings(self) -> ArrayOrScalar[JetSplittingArray]:
        return self._splittings

    @property
    def leading_track_pt(self) -> ArrayOrScalar[float]:
        """ Leading track pt. """
        return self.constituents.max_pt

    def dynamical_z(self, R: float) -> Tuple[ArrayOrScalar[float], ArrayOrScalar[int]]:
        """ Dynamical z of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical z value, leading dynamical z index.
        """
        return self.splittings.dynamical_z(R=R)

    def dynamical_kt(self, R: float) -> Tuple[ArrayOrScalar[float], ArrayOrScalar[int]]:
        """ Dynamical kt of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical kt value, leading dynamical kt index.
        """
        return self.splittings.dynamical_kt(R=R)

    def dynamical_time(self, R: float) -> Tuple[ArrayOrScalar[float], ArrayOrScalar[int]]:
        """ Dynamical time of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical time value, leading dynamical time index.
        """
        return self.splittings.dynamical_time(R=R)

    def leading_kt(self, z_cutoff: Optional[float] = None) -> Tuple[float, float]:
        """ Leading kt of the jet splittings.

        Args:
            z_cutoff: Z cutoff to be applied before calculating the leading kt.
        Returns:
            Leading kt values, leading kt indices.
        """
        return self.splittings.leading_kt(z_cutoff=z_cutoff)

    def soft_drop(self, z_cutoff: float) -> Tuple[float, int, float]:
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
    _constituents: JetConstituentArray = attr.ib()
    _subjets: SubjetArray = attr.ib()
    _splittings: JetSplittingArray = attr.ib()


class SubstructureJetArrayMethods(SubstructureJetCommonMethods, ArrayMethods):
    """ Methods for operating on substructure jet arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """ Create substructure jet views in a table.

        Args:
            table: Table where the substructure jets will be created.
        Returns:
            None.
        """
        self.awkward.ObjectArray.__init__(
            self,
            table,
            lambda row: SubstructureJet(row["jet_pt"], row["constituents"], row["subjets"], row["splittings"]),
        )

    def __awkward_serialize__(self, serializer: ak.persist.Serializer) -> ak.persist.Serializer:
        """ Serialize to storage. """
        self._valid()
        jet_pt, constituents, subjets, splittings = self.jet_pt, self.constituents, self.subjets, self.splittings
        return serializer.encode_call(
            ["jet_substructure.base.substructure_methods", "SubstructureJetArrayMethods", "from_jagged"],
            serializer(jet_pt, "SubstructureJetArrayMethods.jet_pt"),
            serializer(constituents, "SubstructureJetArrayMethods.constituents"),
            serializer(subjets, "SubstructureJetArrayMethods.subjets"),
            serializer(splittings, "SubstructureJetArrayMethods.splittings"),
        )

    @property
    def jet_pt(self) -> UprootArrayTyped[float]:
        """ Jet pt. """
        return cast(UprootArrayTyped[float], self["jet_pt"])

    @property
    def constituents(self) -> UprootArrayTyped[JetConstituentArray]:
        """ Jet constituents. """
        return cast(UprootArrayTyped[JetConstituentArray], self["constituents"])

    @property
    def subjets(self) -> UprootArrayTyped[SubjetArray]:
        """ Subjets. """
        return cast(UprootArrayTyped[SubjetArray], self["subjets"])

    @property
    def splittings(self) -> UprootArrayTyped[JetSplittingArray]:
        """ Jet splittings. """
        return cast(UprootArrayTyped[JetSplittingArray], self["splittings"])


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
        jet_pt: UprootArrayTyped[float],
        jet_constituents: UprootArrayTyped[JetConstituentArray],
        subjets: UprootArrayTyped[SubjetArray],
        jet_splittings: UprootArrayTyped[JetSplittingArray],
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
            tree.get(f"{prefix}.fSubjets.constituents", None),
            constituents,
            tree.get(f"{prefix}.fSubjets.fConstituentJaggedIndices", None),
        )

        logger.debug("Done with constructing subjets.")

        # Construct substructure jets using the above
        return cls(  # type: ignore
            tree[f"{prefix}.fJetPt"], constituents, subjets, splittings,
        )
