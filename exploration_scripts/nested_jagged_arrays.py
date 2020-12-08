#!/usr/bin/env python3

""" Simple reproducer for creating nested Jagged Arrays.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import awkward0 as ak

all_jagged_indices = ak.fromiter([[0, 1, 4], [0, 1, 2, 3]])
all_constituents = ak.fromiter([[12, 14, 3, 4], [2, 8, 3]])
output = ak.fromiter(
    (ak.JaggedArray.fromoffsets(jagged_indices, constituents)
     for jagged_indices, constituents in
     zip(all_jagged_indices, all_constituents))
)
expected = ak.fromiter([[[12], [14, 3, 4]], [[2], [8], [3]]])
assert (output == expected).all().all().all()
print(output.tolist())

import IPython
IPython.start_ipython(user_ns=locals())
