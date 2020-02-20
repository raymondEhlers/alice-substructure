#1/usr/bin/env python3

""" Tests for Jet Substructure interpretation for uproot.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Dict, Generic, Sequence, TypeVar, Union

import awkward as ak
import numpy as np

# Typing helpers
T = TypeVar("T")
UprootArray = Union[np.ndarray, ak.JaggedArray]
UprootArrays = Dict[str, UprootArray]
Result = Union[UprootArray, T]
# More ideally, I would like:
# It's supposed to carry the semantics of Union[np.ndarray, ak.JaggedArray]
#class UprootArray(Generic[T]): ...
#Result = Union[UprootArray[T], T]

class ArrayMethods(ak.Methods):
    # Seems to be required for creating JaggedArray elements within an Array.
    # Otherwise, it will create one object per event, with that object storing arrays of members.
    awkward = ak

class JetConstituent:
    def __init__(self, pt: Result[float], eta: Result[float], phi: Result[float], global_index: Result[int]) -> None:
        self._pt = pt
        self._eta = eta
        self._phi = phi
        self._global_index = global_index

    @property
    def pt(self) -> Result[float]:
        return self._pt

    @property
    def eta(self) -> Result[float]:
        return self._eta

    @property
    def phi(self) -> Result[float]:
        return self._phi

    @property
    def index(self) -> Result[int]:
        return self._global_index

class JetConstituentArrayMethods(ArrayMethods):
    def _init_object_array(self, table: ak.Table) -> None:
        self.awkward.ObjectArray.__init__(
            self, table, lambda row: JetConstituent(row["pt"], row["eta"], row["phi"], row["global_index"])
        )

    @property
    def pt(self) -> Result[float]:
        return self["pt"]

    @property
    def eta(self) -> Result[float]:
        return self["eta"]

    @property
    def phi(self) -> Result[float]:
        return self["phi"]

    @property
    def global_index(self) -> Result[int]:
        return self["global_index"]

# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedJetConstituentArrayMethods = JetConstituentArrayMethods.mixin(JetConstituentArrayMethods, ak.JaggedArray)

class JetConstituentArray(JetConstituentArrayMethods, ak.ObjectArray):
    """ Array of jet constituents.

    """
    def __init__(self, pt: Result[float], eta: Result[float], phi: Result[float], global_index: Result[int]) -> None:
        self._init_object_array(ak.Table())
        self["pt"] = pt
        self["eta"] = eta
        self["phi"] = phi
        self["global_index"] = global_index

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedJetConstituentArrayMethods)
    def from_jagged(cls, pt: Result[float], eta: Result[float], phi: Result[float], global_index: Result[int]) -> "JetConstituentArray":
        return cls(pt, eta, phi, global_index)

class Subjet:
    def __init__(self, part_of_iterative_splitting: Result[bool], parent_splitting_index: Result[int],
                 constituents_indices: Result[int]) -> None:
        self._part_of_iterative_splitting = part_of_iterative_splitting
        self._parent_splitting_index = parent_splitting_index
        self._constituents_indices = constituents_indices

    def part_of_iterative_splitting(self) -> Result[bool]:
        return self._part_of_iterative_splitting

    def parent_splitting(self, splittings: Result[Sequence["JetSplitting"]]) -> Result["JetSplitting"]:
        return splittings[self._parent_splitting_index]

    def constituents(self, jet_constituents: Result[Sequence[JetConstituent]]) -> Result[Sequence[JetConstituent]]:
        return jet_constituents[self._constituents_indices]
        #return ak.JaggedArray.fromoffsets(self._constituents_indices.offsets, jet_constituents[s.flatten()])
        #return ak.JaggedArray.fromoffsets(
        #    self._constituents_indices.offsets, ak.JaggedArray.fromoffsets(
        #        self._constituents_indices.offsets, jet_constituents.flatten()[self._constituents_indices.flatten(axis=1)]
        #    )
        #)

    #constituents[subjets._constituents_indices.flatten(axis=1)]

    #def test():
    #    ak.JaggedArray.fromoffsets(
    #        subjets._constituents_indices.offsets, ak.JaggedArray.fromoffsets(
    #            subjets._constituents_indices.flatten().offsets,
    #            constituents[subjets._constituents_indices.flatten(axis=1)].flatten()
    #        )
    #    )

def _convert_jagged_constituents_indicies(constituents_indices: ak.JaggedArray, jagged_indices: ak.JaggedArray) -> ak.JaggedArray:
    return ak.fromiter(
        (ak.JaggedArray.fromoffsets(jagged, indices)
         for jagged, indices in
         zip(jagged_indices, constituents_indices))
    )

class SubjetArrayMethods(ArrayMethods):
    def _init_object_array(self, table: ak.Table) -> None:
        self.awkward.ObjectArray.__init__(
            self, table, lambda row: Subjet(row["part_of_iterative_splitting"], row["parent_splitting_index"], row["constituents_indices"])
        )

    @property
    def _constituents_indices(self) -> Result[ak.JaggedArray]:
        """ Construct constituent indices from stored JaggedArrays.

        Note:
            I can't figure out how to construct a nested JaggedArray, so I have to construct it event by event.
            This is super inefficient. I will try to revise it when I get a better answer.
        """
        return self["constituents_indices"]
        #return _convert_jagged_constituents_indicies(self["constituents_indices"], self["constituents_jagged_indices"])

    def part_of_iterative_splitting(self) -> Result[bool]:
        return self["part_of_iterative_splitting"]

    def parent_splitting(self, splittings: Result[Sequence["JetSplitting"]]) -> Result["JetSplitting"]:
        return splittings[self["parent_splitting_index"]]

    def constituents(self, jet_constituents: Result[Sequence[JetConstituent]]) -> Result[Sequence[JetConstituent]]:
        #return jet_constituents[self._constituents_indices]
        # This isn't super efficient, but I can't seem to broadcast it directly.
        return ak.JaggedArray.fromoffsets(
            self._constituents_indices.offsets, ak.JaggedArray.fromoffsets(
                self._constituents_indices.flatten().offsets,
                jet_constituents[self._constituents_indices.flatten(axis=1)].flatten()
            )
        )

# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedSubjetArrayMethods = SubjetArrayMethods.mixin(SubjetArrayMethods, ak.JaggedArray)

class SubjetArray(SubjetArrayMethods, ak.ObjectArray):
    def __init__(self, part_of_iterative_splitting: Result[bool], parent_splitting_index: Result[int],
                 constituents_indices: Result[int]) -> None:
        self._init_object_array(ak.Table())
        self["part_of_iterative_splitting"] = part_of_iterative_splitting
        self["parent_splitting_index"] = parent_splitting_index
        self["constituents_indices"] = constituents_indices

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedSubjetArrayMethods)
    def from_jagged(cls, part_of_iterative_splitting: Result[bool], parent_splitting_index: Result[int],
                 constituents_indices: Result[int]) -> "SubjetArray":
        # NOTE: This doesn't work because the JaggedArrays are different sizes.
        #       Need to find a new solution! Probably just construct the constituent indices array before passing it in.
        #       See: https://github.com/scikit-hep/uproot/issues/452
        return cls(part_of_iterative_splitting, parent_splitting_index, constituents_indices)

class JetSplitting:
    def __init__(self, kt: Result[float], delta_R: Result[float], z: Result[float], parent_index: Result[int]) -> None:
        self._kt = kt
        self._delta_R = delta_R
        self._z = z
        self._parent_index = parent_index

    @property
    def kt(self) -> Result[float]:
        return self._kt

    @property
    def delta_R(self) -> Result[float]:
        return self._delta_R

    @property
    def z(self) -> Result[float]:
        return self._z

    def iterative_splitting(self, subjets: Result[Sequence[Subjet]]) -> Result[bool]:
        # Determine the parent index of the splittings which are iterative.
        # This indexes the splittings, so we then apply it to the object.
        iterative_splittings = subjets.parent_splitting[subjets.part_of_iterative_splitting]
        return self[iterative_splittings]

class JetSplittingArrayMethods(ArrayMethods):
    def _init_object_array(self, table: ak.Table) -> None:
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

    def iterative_splitting(self, subjets: Result[Sequence[Subjet]]) -> Result[bool]:
        return self.iterative_splitting(subjets)

# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedJetSplittingArrayMethods = JetSplittingArrayMethods.mixin(JetSplittingArrayMethods, ak.JaggedArray)

class JetSplittingArray(JetSplittingArrayMethods, ak.ObjectArray):
    def __init__(self, kt: Result[float], delta_R: Result[float], z: Result[float], parent_index: Result[int]) -> None:
        self._init_object_array(ak.Table())
        self["kt"] = kt
        self["delta_R"] = delta_R
        self["z"] = z
        self["parent_index"] = parent_index

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedJetSplittingArrayMethods)
    def from_jagged(cls, kt: Result[float], delta_R: Result[float], z: Result[float],
                    parent_index: Result[int]) -> "JetSplittingArray":
        return cls(kt, delta_R, z, parent_index)

class SubstructureJet(ak.Methods):
    def __init__(self, jet_pt: Result[float], constituents: Result[Sequence[JetConstituent]],
                 subjets: Result[Sequence[Subjet]], splittings: Result[Sequence[JetSplitting]]) -> None:
        self._jet_pt = jet_pt

        self._constituents: Sequence[JetConstituent] = constituents
        self._subjets: Sequence[Subjet] = subjets
        self._splittings: Sequence[JetSplitting] = splittings

    @property
    def jet_pt(self) -> Result[float]:
        return self._jet_pt

    @property
    def leading_track_pt(self) -> Result[float]:
        return self._constituents.max_pt()

    @property
    def constituents(self) -> Result[Sequence[JetConstituent]]:
        return self._constituents

    def leading_kt(self) -> Result[float]:
        ...

    def soft_drop_kt(self, z_hard_cutoff: float) -> Result[float]:
        ...

    def dynamical_z(self) -> Result[float]:
        ...

    def dynamical_kt(self) -> Result[float]:
        ...

    def dynamical_time(self) -> Result[float]:
        ...


