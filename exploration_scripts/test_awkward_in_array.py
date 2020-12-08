#!/usr/bin/env python3

""" Test for awkward elements in an array.

Posted as a question to StackOverflow: https://stackoverflow.com/q/61289810/12907985

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import awkward0 as ak0
import awkward as ak
import IPython


# Example arrays:
desired_output = ak0.fromiter([True, False, False])
full_array = ak0.fromiter([[1, 2, 3], [], [0, 1, 2, 3, 4, 5]])
selected_array = ak0.fromiter([[2], [], [7]])
ak1_full_array = ak.from_awkward0(full_array)
ak1_selected_array = ak.from_awkward0(selected_array)

try:
    selected_array in full_array
except ValueError as e:
    print(e)

workaround_array = full_array.ones_like() * selected_array.pad(1).fillna(-1).flatten()
assert (desired_output == (workaround_array == full_array).any()).all()

print(ak1_selected_array in ak1_full_array)

IPython.start_ipython(user_ns=locals())
