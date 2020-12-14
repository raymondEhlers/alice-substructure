""" Tests for the unfolding base.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import numpy as np
import pytest  # noqa: F401

from jet_substructure.analysis import unfolding_base


def test_two_value_asymmetric_error_calcuation() -> None:
    one = np.array([1.0, -1.0, 0.1, -0.1])
    two = np.array([0.5, -1.5, -0.5, 0.5])

    res = unfolding_base.AsymmetricErrors.calculate_errors(one, two)

    np.testing.assert_allclose(res.low, np.array([0, 1.5, 0.5, 0.1]))
    np.testing.assert_allclose(res.high, np.array([1.0, 0, 0.1, 0.5]))


def test_one_sided_asymmetric_error_calculation() -> None:
    one = np.array([1.0, -1.5, 0.1, -0.1])

    res = unfolding_base.AsymmetricErrors.calculate_errors(one)

    np.testing.assert_allclose(res.low, np.array([0, 1.5, 0, 0.1]))
    np.testing.assert_allclose(res.high, np.array([1.0, 0, 0.1, 0]))

    # Verify that it copies the errors correctly.
    res_two_value = unfolding_base.AsymmetricErrors.calculate_errors(one, one)
    assert res == res_two_value
