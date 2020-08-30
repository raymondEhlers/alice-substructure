# Reported and resolved at https://github.com/scikit-hep/awkward-1.0/issues/395

import awkward1 as ak
import numba as nb


parquet_arrays = ak.from_parquet("trains/embedPythia/5966/AnalysisResults.18q.parquet")
prefix = "matched"
arrays = ak.zip(
    {
        # "kt": parquet_arrays[f"{prefix}.fJetSplittings.fKt"],
        # "delta_R": parquet_arrays[f"{prefix}.fJetSplittings.fDeltaR"],
        # "z": parquet_arrays[f"{prefix}.fJetSplittings.fZ"],
        "parent_index": parquet_arrays[f"{prefix}.fJetSplittings.fParentIndex"],
    }
)

# @nb.njit
# def access_values(arrays):
#    ...
#    i = 0
#    for values in arrays:
#        if i > 28:
#            break
#        print("i", i, "parent_index", values.parent_index)
#        if len(values.parent_index):
#            print("first entry", values.parent_index[0])
#        i += 1


@nb.njit
def reproduce(arrays):  # type: ignore
    i = 0
    for values in arrays:
        if i > 28:
            return
        print("======== i =", i)
        parent_indices = values.parent_index
        print("parent_indices", parent_indices)
        j = 0
        for p in parent_indices:
            print(j, ":", p)
            assert p == parent_indices[j]
            j += 1
        i += 1


reproduce(arrays[:, :2])
print(arrays.parent_index.layout)
# reproduce(arrays[:30])


@nb.jit
def reproduce_simple(arrays):  # type: ignore
    i = 0
    for values in arrays:
        if i > 28:
            return
        print("======== i =", i)
        parent_indices = values
        print("parent_indices", parent_indices)
        j = 0
        for p in parent_indices:
            print(j, ":", p)
            j += 1
        i += 1


reproduce_simple(ak.Array([[0, 1], [], [3, 4], [0, 1, 2, 3, 4]])[:, :2])
