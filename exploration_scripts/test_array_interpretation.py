#!/usr/bin/env python3

""" Test of array interpretation which may be causing issues.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import awkward as ak
import IPython
import numpy as np
import uproot

class ArrayMethods(ak.Methods):
    # Seems to be required for creating JaggedArray elements within an Array.
    # Otherwise, it will create one object per event, with that object storing arrays of members.
    awkward = ak

class Subjet:
    def __init__(self, part_of_iterative_splitting, parent_splitting_index, constituents_indices, constituents_jagged_indices):
        self._part_of_iterative_splitting = part_of_iterative_splitting
        self._parent_splitting_index = parent_splitting_index
        self._constituents_indices = constituents_indices
        self._constituents_jagged_indices = constituents_jagged_indices

class SubjetArrayMethods(ArrayMethods):
    # Seems to be required for creating JaggedArray elements within an Array.
    # Otherwise, it will create one object per event, with that object storing arrays of members.
    awkward = ak
    def _init_object_array(self, table):
        self.awkward.ObjectArray.__init__(
            self, table, lambda row: Subjet(row["part_of_iterative_splitting"], row["parent_splitting_index"], row["constituents_indices"], row["constituents_jagged_indices"])
        )

# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedSubjetArrayMethods = SubjetArrayMethods.mixin(SubjetArrayMethods, ak.JaggedArray)

class SubjetArray(SubjetArrayMethods, ak.ObjectArray):
    def __init__(self, part_of_iterative_splitting, parent_splitting_index,
                 constituents_indices, constituents_jagged_indices) -> None:
        self._init_object_array(ak.Table())
        self["part_of_iterative_splitting"] = part_of_iterative_splitting
        self["parent_splitting_index"] = parent_splitting_index
        self["constituents_indices"] = constituents_indices
        self["constituents_jagged_indices"] = constituents_jagged_indices

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedSubjetArrayMethods)
    def from_jagged(cls, part_of_iterative_splitting, parent_splitting_index,
                 constituents_indices, constituents_jagged_indices):
        return cls(part_of_iterative_splitting, parent_splitting_index, constituents_indices, constituents_jagged_indices)

def run():
    f = uproot.open("../temp/AnalysisResults.root")
    t = f["AliAnalysisTaskJetDynamicalGrooming_RawTree_Data_ConstSub_Incl"]
    arrays = t.arrays(namedecode="utf-8")

    subjets = SubjetArray.from_jagged(
        arrays["data.fSubjets.fPartOfIterativeSplitting"],
        arrays["data.fSubjets.fSplittingNodeIndex"],
        arrays["data.fSubjets.fSplittingNodeIndex"],
        arrays["data.fSubjets.fSplittingNodeIndex"],
        #arrays["data.fSubjets.fConstituentIndices"],
        #arrays["data.fSubjets.fConstituentJaggedIndices"],
    )

    IPython.start_ipython(user_ns=locals())

if __name__ == "__main__":
    run()
