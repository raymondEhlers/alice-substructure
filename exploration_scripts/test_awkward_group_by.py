"""Tests for performing group-by via sorting.

Heavily relies on https://stackoverflow.com/q/29352511/12907985
"""

from typing import Sequence, Union

import awkward as ak
import numpy as np
import IPython


def _groupby_manual(array: ak.Array) -> ak.Array:
    sort1 = ak.argsort(array[:, 0])
    sort2 = ak.argsort(array[sort1][:, 1], stable=True)
    return array[sort1][sort2]


def test_awkward_groupby_manual(array: ak.Array, desired_result: ak.Array) -> None:
    group_by = _groupby_manual(array=array)

    if not ak.all(ak.flatten(group_by == desired_result, axis=None)):
        print("Wrong output in manual :-(")
    print("Success for manual")

    return group_by


def _groupby_lexsort(array: ak.Array, columns: Sequence[Union[str, int]]) -> ak.Array:
    #sort = np.lexsort((np.asarray(array[:, 0]), np.asarray(array[:, 1])))
    sort = np.lexsort(tuple(np.asarray(array[:, col]) for col in reversed(columns)))
    return array[sort]


def test_awkward_groupby_lexsort(array: ak.Array, desired_result: ak.Array) -> None:
    group_by = _groupby_lexsort(array=array, columns=[1, 0])

    if not ak.all(ak.flatten(group_by == desired_result, axis=None)):
        print("Wrong output in lexsort :-(")
    print("Success for lexsort")

    return group_by


if __name__ == "__main__":
    array = ak.Array([
        [   2.,    1.,    2.,    0.],
        [   4.,    2.,    4.,    0.],
        [   2.,    3.,  100.,    0.],
        [   3.,    1.,    2.,    0.],
        [   3.,    3.,    6.,    0.],
        [   2.,    2.,  100.,    0.],
        [   4.,    1.,    2.,    0.],
        [   3.,    2.,    4.,    0.],
        [   4.,    3.,    6.,    0.]
    ])

    desired_result = ak.Array([
        [   2.,   1.,   2.,   0.],
        [   3.,   1.,   2.,   0.],
        [   4.,   1.,   2.,   0.],
        [   2.,   2., 100.,   0.],
        [   3.,   2.,   4.,   0.],
        [   4.,   2.,   4.,   0.],
        [   2.,   3., 100.,   0.],
        [   3.,   3.,   6.,   0.],
        [   4.,   3.,   6.,   0.],
    ])

    array_manual = test_awkward_groupby_manual(array=array, desired_result=desired_result)
    array_lexsort = test_awkward_groupby_lexsort(array=array, desired_result=desired_result)

    # TODO: These are sorted, but they're not grouped by yet. Need to use ak.run_lengths (and probably better test data for such a scenario)

    IPython.embed()

    # On my laptop:
    # In [4]: %timeit _groupby_lexsort(array)
    # 724 µs ± 4.83 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    # In [5]: %timeit _groupby_manual(array)
    # 1.48 ms ± 46.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
