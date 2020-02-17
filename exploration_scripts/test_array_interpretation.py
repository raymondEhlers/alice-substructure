#!/usr/bin/env python3

""" Test of array interpretation which may be causing issues.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import awkward as ak
import uproot

class ArrayMethods(ak.Methods):
    # Seems to be required for creating JaggedArray elements within an Array.
    # Otherwise, it will create one object per event, with that object storing arrays of members.
    awkward = ak

class MyObject:
    def __init__(self, a, b):
        self._a = a
        self._b = b

class MyObjectArrayMethods(ArrayMethods):
    def _init_object_array(self, table):
        self.awkward.ObjectArray.__init__(
            self, table, lambda row: MyObject(row["a"], row["b"])
        )

# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedMyArrayMethods = MyObjectArrayMethods.mixin(MyObjectArrayMethods, ak.JaggedArray)

class MyArray(MyObjectArrayMethods, ak.ObjectArray):
    def __init__(self, a, b) -> None:
        self._init_object_array(ak.Table())
        self["a"] = a
        self["b"] = b

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedMyArrayMethods)
    def from_jagged(cls, a, b):
        return cls(a, b)

f = uproot.open("../temp/reproducer.root")
t = f["reproducer"]
arrays = t.arrays(namedecode="utf-8")

subjets = MyArray.from_jagged(
    arrays["data.fSubjets.fConstituentIndices"],
    arrays["data.fSubjets.fConstituentJaggedIndices"],
)
