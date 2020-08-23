""" Uproot4 + awkward1 substructure methods.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import functools
import typing
from pathlib import Path
from typing import Final, Optional, Tuple, TypeVar, cast

import attr
import awkward1 as ak
import numpy as np
import uproot_methods

from jet_substructure.base.helpers import ArrayOrScalar, UprootArray

# Typing helpers
_T = TypeVar("_T")


# Constants
UNFILLED_VALUE: Final[float] = -0.005


@typing.overload
def _dynamical_hardness_measure(
    delta_R: UprootArray[float], z: UprootArray[float], parent_pt: UprootArray[float], R: float, a: float
) -> UprootArray[float]:
    ...


@typing.overload
def _dynamical_hardness_measure(delta_R: float, z: float, parent_pt: float, R: float, a: float) -> float:
    ...


def _dynamical_hardness_measure(delta_R, z, parent_pt, R, a):  # type: ignore
    return z * (1 - z) * parent_pt * (delta_R / R) ** a


dynamical_z = functools.partial(_dynamical_hardness_measure, a=0.1)
dynamical_kt = functools.partial(_dynamical_hardness_measure, a=1.0)
dynamical_time = functools.partial(_dynamical_hardness_measure, a=2.0)


def find_leading(values: UprootArray[_T]) -> Tuple[np.ndarray, UprootArray[int]]:
    """ Calculate hardest value given a set of values.

    Used for dynamical grooming, hardest kt, etc.

    In the case that we don't find a viable max (ie. because there was no splitting), we pad
    to one entry and fill -0.01 (our UNFILLED_VALUE) before flattening. The corresponding index
    will be empty for that event. This way, we can just fill all values, regardless of whether
    the splittings were selected, and we automatically get the right normalization (as long as
    those values are included in the hist...).

    Returns:
        Leading value, index of value.
    """
    arg_max = ak.argmax(values)
    return values[arg_max].pad(1).fillna(UNFILLED_VALUE).flatten(), arg_max

class JetConstituentCommon:
    offset: int = 2000000
    pt: float
    eta: float
    phi: float
    ID: int

    def delta_R(self: _T, other: _T) -> ArrayOrScalar[float]:
        """ Separation between jet constituents. """
        return np.sqrt((self.phi - other.phi) ** 2 + (self.eta - other.eta) ** 2)

class JetConstituent(ak.Record, JetConstituentCommon):
    """ A single jet constituent.

    Args:
        pt: Jet constituent pt.
        eta: Jet constituent eta.
        phi: Jet constituent phi.
        id: Jet constituent identifier. MC label (via GetLabel()) or global index (with offset defined above).
    """

    pt: float
    eta: float
    phi: float
    id: int

    def four_vector(self, mass_hypothesis: float = 0.139) -> uproot_methods.TLorentzVector:
        return uproot_methods.TLorentzVector(self.pt, self.eta, self.phi, mass_hypothesis,)


class JetConstituentArray(ak.Array, JetConstituentCommon):
    """ Methods for operating on jet constituents arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
    """

    pt: UprootArray[float]
    eta: UprootArray[float]
    phi: UprootArray[float]
    id: UprootArray[int]

    @property
    def max_pt(self) -> ArrayOrScalar[float]:
        """ Maximum pt of the stored constituent. """
        return cast(ArrayOrScalar[float], self.pt.max())

    def four_vectors(self, mass_hypothesis: float = 0.139) -> uproot_methods.TLorentzVectorArray:
        mass_hypothesis_array = self.pt * 0 + mass_hypothesis
        return uproot_methods.TLorentzVectorArray.from_ptetaphim(
            self.pt, self.eta, self.phi, mass_hypothesis_array,
        )

# Register behavior
ak.behavior["JetConstituent"] = JetConstituent
ak.behavior["*", "JetConstituent"] = JetConstituentArray

class SubjetCommon:
    """ Common subjet related methods. """

    part_of_iterative_splitting: bool
    parent_splitting_index: int
    constituents: JetConstituentArray

    @typing.overload
    def parent_splitting(self, splittings: UprootArray[JetSplittingArray]) -> JetSplittingArray:
        ...

    @typing.overload
    def parent_splitting(self, splittings: JetSplittingArray) -> JetSplitting:
        ...

    def parent_splitting(self, splittings: JetSplittingArray) -> JetSplitting:
        """ Retrieve the parent splitting of this subjet.

        Args:
            splittings: All of the splittings from the overall jet.
        Returns:
            Splitting which led to this subjet.
        """
        return splittings[self.parent_splitting_index]


class Subjet(ak.Record, SubjetCommon):
    """ Single subjet. """
    part_of_iterative_splitting: bool
    parent_splitting_index: int
    constituents: JetConstituentArray

class SubjetArray(ak.Array, SubjetCommon):
    """ Array of subjets. """
    part_of_iterative_splitting: UprootArray[bool]
    parent_splitting_index: UprootArray[int]
    constituents: JetConstituentArray

# Register behavior
ak.behavior["Subjet"] = Subjet
ak.behavior["*", "Subjet"] = SubjetArray

class JetSplittingCommon:
    """ Common jet splitting related methods. """
    kt: float
    delta_R: float
    z: float
    parent_index: int

    @property
    def parent_pt(self) -> UprootArray[float]:
        """ Pt of the (parent) subjets which lead to the splittings.

        The pt can be calculated from the splitting properties via:

        parent_pt = subleading / z = kt / sin(delta_R) / z

        Args:
            None.
        Returns:
            None.
        """
        # parent_pt = subleading / z = kt / sin(delta_R) / z
        return cast(UprootArray[float], self.kt / np.sin(self.delta_R) / self.z)

    def theta(self, jet_R: float) -> float:
        """ Theta of the splitting.

        This is defined as delta_R normalized by the jet resolution parameter.

        Args:
            jet_R: Jet resolution parameter.
        Returns:
            Theta of the splitting.
        """
        return self.delta_R / jet_R


class JetSplitting(ak.Record, JetSplittingCommon):
    """ Single jet splitting. """

    kt: float
    delta_R: float
    z: float
    parent_index: int

    @property
    def parent_pt(self) -> ArrayOrScalar[float]:
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

    def dynamical_z(self, R: float) -> float:
        """ Dynamical z of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical z of the splitting.
        """
        return dynamical_z(self.delta_R, self.z, self.parent_pt, R)  # type: ignore

    def dynamical_kt(self, R: float) -> float:
        """ Dynamical kt of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical kt of the splitting.
        """
        return dynamical_kt(self.delta_R, self.z, self.parent_pt, R)  # type: ignore

    def dynamical_time(self, R: float) -> float:
        """ Dynamical time of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical time of the splitting.
        """
        return dynamical_time(self.delta_R, self.z, self.parent_pt, R)  # type: ignore

class JetSplittingArray(ak.Array, JetSplittingCommon):
    """ Array of jet splittings. """

    kt: UprootArray[float]
    delta_R: UprootArray[float]
    z: UprootArray[float]
    parent_index: UprootArray[int]

    def dynamical_z(self, R: float) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """ Dynamical z of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical z values, leading dynamical z indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_z(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, self.z.localindex

    def dynamical_kt(self, R: float) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """ Dynamical kt of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical kt values, leading dynamical kt indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_kt(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, self.z.localindex

    def dynamical_time(self, R: float) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """ Dynamical time of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical time values, leading dynamical time indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_time(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, self.z.localindex

    def leading_kt(
        self, z_cutoff: Optional[float] = None
    ) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """ Leading kt of the jet splittings.

        Args:
            z_cutoff: Z cutoff to be applied before calculating the leading kt.
        Returns:
            Leading kt values, leading kt indices, indices of all splittings which pass the cutoff.
        """
        # Need to use the local index because we are going to mask z values. If we index from the masked
        # z values, it it is applied to the unmasked array later, it will give nonsense. So we mask the local index,
        # find the leading, and then apply that index back to the local index, which then gives us the leading index
        # in the unmasked array.
        indices_passing_cutoff = self.z.localindex
        if z_cutoff is not None:
            indices_passing_cutoff = self.z.localindex[self.z > z_cutoff]
        values, indices = find_leading(self.kt[indices_passing_cutoff])
        return values, indices_passing_cutoff[indices], indices_passing_cutoff

    def soft_drop(self, z_cutoff: float) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """ Calculate soft drop of the splittings.

        Note:
            z_g is filled with the `UNFILLED_VALUE` if a splitting wasn't selected. In that case, there is
            no index (ie. an emptry JaggedArray entry), and n_sd = 0.

        Note:
            n_sd can be calculated by using `count_nonzero()` on the indices which pass the cutoff.

        Args:
            z_cutoff: Minimum z for Soft Drop.
        Returns:
            First z passing cutoff (z_g), index of z passing cutoff, indices of all splittings which pass the cutoff.
        """
        z_cutoff_mask = self.z > z_cutoff
        indices_passing_cutoff = self.z.localindex[z_cutoff_mask]
        # We use :1 because this maintains the jagged structure. That way, we can apply it to initial arrays.
        z_index = indices_passing_cutoff[:, :1]
        z_g = self.z[z_index].pad(1).fillna(UNFILLED_VALUE).flatten()

        return z_g, z_index, indices_passing_cutoff

# Register behavior
ak.behavior["JetSplitting"] = JetSplitting
ak.behavior["*", "JetSplitting"] = JetSplittingArray

#def jet_substructure(tree: ak.JaggedArray, prefix: str):
def jet_substructure_initial(tree, prefix: str):

    filename = Path("trains/embedPythia/5966/AnalysisResults.18q.parquet")
    if filename.exists():
        return

    #return ak.Array(
    #    properties={

    #    }
    #)

    # FIXME: Calling array on each is slow!!

    # TODO: Need to play with the depth limit here.
    # TODO: It may make sense to convert the arrays that are defined here to be zip with a limited depth limit...

    #arrays = ak.zip({
    #    f"data.fJetPt": tree[f"{prefix}.fJetPt"].array(),
    #    f"data.fJetConstituents.fPt": tree[f"{prefix}.fJetConstituents.fPt"].array(),
    #})
    arrays = tree.arrays(
        [
            f"{prefix}.fJetPt",
            f"{prefix}.fJetConstituents.fPt",
            f"{prefix}.fJetConstituents.fEta",
            f"{prefix}.fJetConstituents.fPhi",
            f"{prefix}.fJetConstituents.fID",
            f"{prefix}.fJetSplittings.fKt",
            f"{prefix}.fJetSplittings.fDeltaR",
            f"{prefix}.fJetSplittings.fZ",
            f"{prefix}.fJetSplittings.fParentIndex",
            f"{prefix}.fSubjets.fPartOfIterativeSplitting",
            f"{prefix}.fSubjets.fSplittingNodeIndex",
            f"{prefix}.fSubjets.fConstituentIndices",
        ],
        #filter_name=f"{prefix}.fJetConstituents",
        #entry_stop=1000,
    )

    ak.to_parquet(arrays, "trains/embedPythia/5966/AnalysisResults.18q.parquet")

def arrow_to_substructure(prefix: str):

    arrays = ak.from_parquet("trains/embedPythia/5966/AnalysisResults.18q.parquet")

    #jet_constituents = ak.zip(
    #    {
    #        "pt": arrays[f"{prefix}.fJetConstituents.fPt"],
    #        "eta": arrays[f"{prefix}.fJetConstituents.fEta"],
    #        "phi": arrays[f"{prefix}.fJetConstituents.fPhi"],
    #        "id": arrays[f"{prefix}.fJetConstituents.fID"],
    #    },
    #    with_name="JetConstituent",
    #)
    #subjets = ak.zip(
    #    {
    #        "kt": arrays[f"{prefix}.fJetSplittings.fKt"],
    #        "delta_R": arrays[f"{prefix}.fJetSplittings.fDeltaR"],
    #        "z": arrays[f"{prefix}.fJetSplittings.fZ"],
    #        "parent_index": arrays[f"{prefix}.fJetSplittings.fParentIndex"],
    #    },
    #    with_name="Subjet",
    #)
    #return arrays
    return ak.zip({
        "jet_pt": arrays[f"{prefix}.fJetPt"],
        "jet_constituents": ak.zip(
            {
                "pt": arrays[f"{prefix}.fJetConstituents.fPt"],
                "eta": arrays[f"{prefix}.fJetConstituents.fEta"],
                "phi": arrays[f"{prefix}.fJetConstituents.fPhi"],
                "id": arrays[f"{prefix}.fJetConstituents.fID"],
            },
            with_name="JetConstituent",
        ),
        "jet_splittings": ak.zip(
            {
                "kt": arrays[f"{prefix}.fJetSplittings.fKt"],
                "delta_R": arrays[f"{prefix}.fJetSplittings.fDeltaR"],
                "z": arrays[f"{prefix}.fJetSplittings.fZ"],
                "parent_index": arrays[f"{prefix}.fJetSplittings.fParentIndex"],
            },
            with_name="JetSplitting",
        ),
        "subjets": ak.zip(
            {
                "part_of_iterate_splitting": arrays[f"{prefix}.fSubjets.fPartOfIterativeSplitting"],
                "splitting_node_index": arrays[f"{prefix}.fSubjets.fSplittingNodeIndex"],
                "constituent_indices": arrays[f"{prefix}.fSubjets.fConstituentIndices"],
            },
            with_name="Subjet",
        ),
    }, depth_limit=1)



if __name__ == "__main__":
    import uproot4 as uproot
    f = uproot.open("trains/embedPythia/5966/AnalysisResults.18q.repaired.root")
    tree = f["AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"]
    tree.show()
    import time
    start = time.perf_counter()
    jet_substructure_initial(tree, prefix="data")
    arrays = arrow_to_substructure(prefix="data")
    finish = time.perf_counter()
    print(f"Uproot4: Length: {ak.num(arrays, axis=0)}, time: {finish-start}")
    import IPython; IPython.embed()
    f.close()

    #import uproot as uproot3
    #f = uproot3.open("trains/embedPythia/5966/AnalysisResults.18q.root")
    #tree = f["AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"]
    #import time
    #start = time.perf_counter()
    #arrays = jet_substructure(tree, prefix="data")
    #finish = time.perf_counter()
    #print(f"Uproot3: Length: {len(arrays[b'data.fJetPt'])}, time: {finish-start}")
