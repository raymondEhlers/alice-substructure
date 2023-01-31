"""Tests for the unfolding module

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
from pachyderm import yaml

from jet_substructure.base import unfolding as unfolding_base

_test_config = """
nominal_binning:
    default:
        jet_pt:
            "true": [0, 20, 40, 60, 80, 100]
            "smeared": [30, 40, 60, 80, 100]
        kt:
            "true": [1, 1.5, 2, 3, 4, 5, 6]
            "smeared": [1, 2, 3, 4, 5, 6]
            jet_pt:
                # NOTE: 3x default
                "true": [  0.,  60., 120., 180., 240., 300.]
                "smeared": [ 90., 120., 180., 240., 300.]
            var_over_pt:
                "true": [...]
                "smeared": [...]
        delta_R:
            # NOTE: -1 x default (just for convenience)
            "true": [-0.5, -1, -1.5, -2, -3, -4, -5, -6]
            "smeared": [-0.25, -1, -2, -3, -4, -5, -6]
    soft_drop_z_cut_02:
        # NOTE: 2x default!
        kt:
            "true": [ 2.,  3.,  4.,  6.,  8., 10., 12.]
            "smeared": [ 2.,  4.,  6.,  8., 10., 12.]
            jet_pt:
                "true": [  0.,  40.,  80., 120., 160., 200.]
                "smeared": [ 60.,  80., 120., 160., 200.]
settings:
    some_new_settings:
        binning:
            default:
                kt:
                    # NOTE: 4x default
                    "true": [ 4.,  6.,  8., 12., 16., 20., 24.]
                    "smeared": [ 4.,  8., 12., 16., 20., 24.]
                    jet_pt:
                        "true": [  0.,  80., 160., 240., 320., 400.]
                        "smeared": [120., 160., 240., 320., 400.]
                delta_R:
                    NOTE: -2x default
                    "true": [ -1.0, -2.,  -3.,  -4.,  -6.,  -8., -10., -12.]
                    "smeared": [ -0.5, -2.,  -4.,  -6.,  -8., -10., -12.]
                    jet_pt:
                        "true": [  -0.,  -40.,  -80., -120., -160., -200.]
                        "smeared": [ -60.,  -80., -120., -160., -200.]
            soft_drop_z_cut_02:
                # NOTE: 5x default
                kt:
                    "true": [ 5. ,  7.5, 10. , 15. , 20. , 25. , 30. ]
                    "smeared": [ 5., 10., 15., 20., 25., 30.]
                    jet_pt:
                        "true": [  0., 100., 200., 300., 400., 500.]
                        "smeared": [150., 200., 300., 400., 500.]
    new_settings_without_specialized_binning:
        value: "some"
"""

@pytest.fixture
def binning_in_unfolding_config() -> Dict[str, Any]:
    y = yaml.yaml()
    return y.load(_test_config)  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "specialized_settings", ["some_new_settings", "new_settings_without_specialized_binning"]
)
@pytest.mark.parametrize(
    "binning_type", ["smeared", "true"],
)
@pytest.mark.parametrize(
    "grooming_method", ["dynamical_core", "soft_drop_z_cut_02"],
)
@pytest.mark.parametrize(
    "substructure_variable_to_analyze", ["kt", "delta_R"],
)
@pytest.mark.parametrize(
    "variable_to_retrieve", [None, "jet_pt"],
)
def test_loading_binning_from_unfolding_config(
    caplog: Any,
    binning_in_unfolding_config: Any,
    specialized_settings: str,
    binning_type: unfolding_base.BinningType,
    grooming_method: str,
    substructure_variable_to_analyze: str,
    variable_to_retrieve: str | None,
) -> None:
    base_unfolding_config = binning_in_unfolding_config
    unfolding_settings = binning_in_unfolding_config["settings"][specialized_settings]
    binning = unfolding_base.get_binning(
        unfolding_settings=unfolding_settings,
        base_unfolding_config=base_unfolding_config,
        binning_type=binning_type,
        grooming_method=grooming_method,
        substructure_variable_to_analyze=substructure_variable_to_analyze,
        variable_to_retrieve=variable_to_retrieve,
    )


    # Determine the expected value
    _base_true_values = {
        "kt": np.array([1, 1.5, 2, 3, 4, 5, 6], dtype=np.float64),
        # NOTE: The sign is added in with the multiplicative factor
        "delta_R": np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6], dtype=np.float64),
        "jet_pt": np.array([0, 20, 40, 60, 80, 100], dtype=np.float64),
    }
    _base_smeared_values = {
        "kt": np.array([1, 2, 3, 4, 5, 6], dtype=np.float64),
        # NOTE: The sign is added in with the multiplicative factor
        "delta_R": np.array([0.25, 1, 2, 3, 4, 5, 6], dtype=np.float64),
        "jet_pt": np.array([30, 40, 60, 80, 100], dtype=np.float64),
    }
    # First, what variable are we trying to look at?
    _expected_variable_name = substructure_variable_to_analyze
    if variable_to_retrieve:
        _expected_variable_name = variable_to_retrieve
    # And grab those relevant values
    _expected_values = (_base_smeared_values if binning_type == "smeared" else _base_true_values)[_expected_variable_name]
    # Next, we need to figure out what factor to expect
    # This isn't especially sustainable because I'm just enumerating options here, but I don't see a nice way to do it, so we'll just go with this for now...
    # NOTE: Since the smeared and true values are different, we can be confident that we would notice if we mix those up when we grab the bins
    _multiplicative_factors = {
        # specialization, grooming_method, substructure_variable, _expected_variable_name
        # kt
        ("some_new_settings", "dynamical_core", "kt", "kt"): 4,
        ("some_new_settings", "dynamical_core", "kt", "jet_pt"): 4,
        ("some_new_settings", "soft_drop_z_cut_02", "kt", "kt"): 5,
        ("some_new_settings", "soft_drop_z_cut_02", "kt", "jet_pt"): 5,
        ("new_settings_without_specialized_binning", "dynamical_core", "kt", "kt"): 1,
        ("new_settings_without_specialized_binning", "dynamical_core", "kt", "jet_pt"): 3,
        ("new_settings_without_specialized_binning", "soft_drop_z_cut_02", "kt", "kt"): 2,
        ("new_settings_without_specialized_binning", "soft_drop_z_cut_02", "kt", "jet_pt"): 2,
        # delta_R
        ("some_new_settings", "dynamical_core", "delta_R", "delta_R"): -2,
        ("some_new_settings", "dynamical_core", "delta_R", "jet_pt"): -2,
        ("some_new_settings", "soft_drop_z_cut_02", "delta_R", "delta_R"): -2,
        ("some_new_settings", "soft_drop_z_cut_02", "delta_R", "jet_pt"): -2,
        ("new_settings_without_specialized_binning", "dynamical_core", "delta_R", "delta_R"): -1,
        ("new_settings_without_specialized_binning", "dynamical_core", "delta_R", "jet_pt"): 1,
        ("new_settings_without_specialized_binning", "soft_drop_z_cut_02", "delta_R", "delta_R"): -1,
        ("new_settings_without_specialized_binning", "soft_drop_z_cut_02", "delta_R", "jet_pt"): 1,
    }
    _expected_values *= _multiplicative_factors[
        (specialized_settings, grooming_method, substructure_variable_to_analyze, _expected_variable_name)
    ]

    # NOTE: This directly asserts, so we don't need an assertion
    np.testing.assert_allclose(binning, _expected_values)
