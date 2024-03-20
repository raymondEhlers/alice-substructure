""" Unfolding results base functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from functools import reduce
from typing import Any, TypeVar

import attrs
import numpy as np
import numpy.typing as npt

from jet_substructure.base import helpers
from mammoth.framework.analysis import array_helpers
from pachyderm import binned_data

logger = logging.getLogger(__name__)


@attrs.define(eq=False)
class AsymmetricErrors:
    low: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]

    def __eq__(self, other: Any) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)

    def smooth(self, n_times: int = 1) -> None:
        """Smooth the uncertainties.

        Args:
            n_times: Number of iterations to use to smooth the uncertainties.
        """
        # Setup
        # We'll have to treat it differently for one-sided and two-sided
        one_sided = False
        low_is_zero_array = np.isclose(self.low, np.zeros_like(self.low), atol=1e-17)
        high_is_zero_array = np.isclose(self.high, np.zeros_like(self.high), atol=1e-17)
        if np.all(low_is_zero_array or high_is_zero_array):
            one_sided = True

        if one_sided:
            values = np.where(low_is_zero_array, self.high, self.low)
            smoothed_values = array_helpers.smooth_array(values, n_times=n_times)
            self.low[~low_is_zero_array] = smoothed_values[~low_is_zero_array]
            self.high[low_is_zero_array] = smoothed_values[low_is_zero_array]
        else:
            # Smooth the uncertainties
            self.low = array_helpers.smooth_array(self.low, n_times=n_times)
            self.high = array_helpers.smooth_array(self.high, n_times=n_times)

    @classmethod
    def calculate_errors(
        cls: type[AsymmetricErrors], errors_one: npt.NDArray[np.float64], errors_two: npt.NDArray[np.float64] | None = None
    ) -> AsymmetricErrors:
        """Calculate asymmetric errors from given errors.

        Note:
            This returns positive, absolute errors in each direction.

        Args:
            errors_one: First error array. Doesn't matter if it's the upper or lower value.
            errors_two: Second error array. Doesn't matter if it's the upper or lower value.
                Default: None, in which case the first passed errors are duplicated. This duplication
                is used for single valued asymmetric errors.
        Returns:
            Asymmetric errors calculated based on the given errors. See the function for the
                precise algorithm.
        """
        # Validation
        # This allows us to calculate single value asymmetric errors.
        # This is equivalent to just pass the same error values twice, but this
        # is a cleaner interface.
        one_sided = False
        if errors_two is None:
            errors_two = np.array(errors_one, copy=True)
            one_sided = True

        # Determine when the errors are positive.
        # True if positive, false if negative
        positive_one = np.sign(errors_one) == 1
        positive_two = np.sign(errors_two) == 1
        # Calculate once for convenience.
        errors_one_abs = np.abs(errors_one)
        errors_two_abs = np.abs(errors_two)

        # Output arrays
        low = np.zeros_like(errors_one)
        high = np.zeros_like(errors_one)

        # First, handle when they have the same sign.
        # For this case, we take the maximum of either error, and assign that asymmetrically to the side of the sign
        # positive -> high
        # negative -> low
        same_sign = positive_one == positive_two
        # Both positive.
        mask = same_sign & positive_one
        # Don't need to set low because it's already zero for these points.
        high[mask] = np.maximum(errors_one_abs, errors_two_abs)[mask]
        # Both negative.
        mask = same_sign & ~positive_one
        # Don't need to set high because it's already zero for these points.
        low[mask] = np.maximum(errors_one_abs, errors_two_abs)[mask]

        # Next, handle opposite signs.
        # For this case, we assign the errors based on the sign.
        # positive -> high
        # negative -> low
        opposite_sign = positive_one != positive_two
        # one positive, two negative
        mask = opposite_sign & positive_one
        low[mask] = errors_two_abs[mask]
        high[mask] = errors_one_abs[mask]
        # one negative, two positive
        mask = opposite_sign & positive_two
        low[mask] = errors_one_abs[mask]
        high[mask] = errors_two_abs[mask]

        # Cross checks. We almost certainly will never have 0 in both bins.
        low_is_all_zero = np.allclose(low, np.zeros(len(low)), atol=1e-17)
        high_is_all_zero = np.allclose(high, np.zeros(len(high)), atol=1e-17)
        if np.any(low_is_all_zero & high_is_all_zero):
            logger.warning("Errors are all identically zero for this calculation! Check this carefully!")
        # If it's one sided, then we always should have only one non-zero error.
        if one_sided:
            low_is_zero_array = np.isclose(low, np.zeros(len(low)), atol=1e-17)
            high_is_zero_array = np.isclose(high, np.zeros(len(high)), atol=1e-17)
            # `not` is needed because assert needs to be False for the assertion to fail.
            assert not np.any(
                ~low_is_zero_array & ~high_is_zero_array
            ), f"One sided errors should only have one non-zero value. low: {low}, high: {high}, test: {np.where(~low_is_zero_array & ~high_is_zero_array)}"

        return cls(low=low, high=high)


@attrs.define
class ErrorInput:
    value: npt.NDArray[np.float64]
    error: npt.NDArray[np.float64]


def relative_error(*inputs: ErrorInput) -> npt.NDArray[np.float64]:
    """ Specifically for ratios or individual values. """
    if len(inputs) == 0:
        _msg = "Must pass at least one ErrorInput"
        raise ValueError(_msg)
    if len(inputs) > 1:
        relative_error_squared: npt.NDArray[np.float64] = reduce(lambda x, y: ((x.error / x.value) ** 2) + ((y.error / y.value) ** 2), inputs)  # type: ignore[arg-type, no-any-return, attr-defined]
    else:
        relative_error_squared = (inputs[0].error / inputs[0].value) ** 2
    return np.sqrt(relative_error_squared)


_T_RangeSelector = TypeVar("_T_RangeSelector", bound=helpers.RangeSelector)

def determine_overlapping_range(current_range: _T_RangeSelector, reference: _T_RangeSelector) -> _T_RangeSelector:
    range_min, range_max = tuple(current_range)
    if range_min < reference.min:
        range_min = reference.min
    if range_max > reference.max:
        range_max = reference.max
    return type(current_range)(range_min, range_max)


def select_hist_range(hist: binned_data.BinnedData, x_range: helpers.RangeSelector, allow_range_broader_than_bin_edges: bool = False) -> binned_data.BinnedData:
    """Select a histogram with a new range, including the systematics.

    NOTE:
        This doesn't belong in binned_data precisely because it uses our systematic uncertainty conventions here.
        Otherwise, it's just relying on standard functionality.
    """
    # Cross check
    if len(hist.axes) > 1:
        _msg = "Can only handle 1D histogram"
        raise ValueError(_msg)

    # Validation
    low, high = tuple(x_range)
    if x_range.min < hist.axes[0].bin_edges[0]:
        msg = "Range is smaller than the lower bin edge!"
        if allow_range_broader_than_bin_edges:
            low = hist.axes[0].bin_edges[0]
            logger.debug(msg)
        else:
            msg += " You need to allow this explicitly if it's okay"
            raise ValueError(msg)
    if x_range.max > hist.axes[0].bin_edges[-1]:
        msg = "Range is larger than the upper bin edge!"
        if allow_range_broader_than_bin_edges:
            high = hist.axes[0].bin_edges[-1]
        else:
            msg += " You need to allow this explicitly if it's okay"
            raise ValueError(msg)

    # Setup
    # We'll use a consistent slice throughout. This does mean that we have to create additional binned_data objects
    # just to do the selection, but I think that tradeoff is worth it to use a consistent code path for all selections.
    # NOTE: We could refactor the selection, but we would need the binning and the values, and so we're already most of
    #       the way there. By using the existing code, we don't have to worry about any subtle issues.
    selected_range = slice(low * 1j, high * 1j)

    # First, we handle the main hist
    h = hist[selected_range]

    # Then handle the metadata
    metadata: dict[str, Any] = {}
    # These will be the same for each case, so no need to repeatedly recreate the objects
    _axes = binned_data.Axis(hist.axes[0].bin_edges)
    _empty_variances = np.zeros(len(hist.values))
    for k, v in hist.metadata.items():
        if k == "y_systematic":
            y_systematic = {}
            for k_sys, v_sys in v.items():
                if isinstance(v_sys, AsymmetricErrors):
                    y_systematic[k_sys] = AsymmetricErrors(
                        low=binned_data.BinnedData(
                            axes=_axes,
                            values=v_sys.low,
                            variances=_empty_variances,
                        )[selected_range].values,
                        high=binned_data.BinnedData(
                            axes=_axes,
                            values=v_sys.high,
                            variances=_empty_variances,
                        )[selected_range].values,
                    )
            metadata["y_systematic"] = y_systematic
        else:
            if isinstance(v, AsymmetricErrors):
                metadata[k] = AsymmetricErrors(
                    low=binned_data.BinnedData(
                        axes=_axes,
                        values=v.low,
                        variances=_empty_variances,
                    )[selected_range].values,
                    high=binned_data.BinnedData(
                        axes=_axes,
                        values=v.high,
                        variances=_empty_variances,
                    )[selected_range].values,
                )
    h.metadata = metadata

    return h


def rebin_bin_width_scaled_hist(h_to_rebin: binned_data.BinnedData, h_target_axis: binned_data.Axis, okay_for_systematic_not_to_exist: bool = False) -> binned_data.BinnedData:
    # Validation and setup
    has_systematic = "y_systematic" in h_to_rebin.metadata and "quadrature" in h_to_rebin.metadata["y_systematic"]
    if not has_systematic and not okay_for_systematic_not_to_exist:
        _msg = f"Expected y systematic for rebinning, but it wasn't found. Please check input: {h_to_rebin=}"
        raise ValueError(_msg)

    # NOTE: This may not be the 100% most efficient way, but it's conceptually simple
    # First, undo the bin width scaling
    # For the main hist
    h_to_rebin *= h_to_rebin.axes[0].bin_widths
    # And the metadata
    if has_systematic:
        _ratio_y_systematic = h_to_rebin.metadata["y_systematic"]["quadrature"]
        _ratio_y_systematic = AsymmetricErrors(
            low=_ratio_y_systematic.low * h_to_rebin.axes[0].bin_widths,
            high=_ratio_y_systematic.high * h_to_rebin.axes[0].bin_widths,
        )
        #ratio_reference_hist.metadata["y_systematic"]["quadrature"] = _ratio_y_systematic
        # Next, rebin this hist
        # First, the systematic, so we don't lose track of it in reassigning the main object
        # We construct an additional hist with the uncertainties so that they're handled properly
        # NOTE: It would be really nice if we could do this more gracefully! This is really hacky
        _ratio_systematic_low = binned_data.BinnedData(
            axes=binned_data.Axis(h_to_rebin.axes[0].bin_edges),
            values=_ratio_y_systematic.low,
            variances=np.ones(len(h_to_rebin.values)),
        )[::h_target_axis.bin_edges].values
        _ratio_systematic_high = binned_data.BinnedData(
            axes=binned_data.Axis(h_to_rebin.axes[0].bin_edges),
            values=_ratio_y_systematic.high,
            variances=np.ones(len(h_to_rebin.values)),
        )[::h_target_axis.bin_edges].values
        # Store the update systematics and scale back down by the bin widths
        _ratio_y_systematic = AsymmetricErrors(
            _ratio_systematic_low / h_target_axis.bin_widths,
            _ratio_systematic_high / h_target_axis.bin_widths,
        )

    # And finally rebin the main data
    h_to_rebin = h_to_rebin[::h_target_axis.bin_edges]
    # And scale back by the bin width
    h_to_rebin /= h_to_rebin.axes[0].bin_widths
    # And store the updated systematic
    if has_systematic:
        h_to_rebin.metadata["y_systematic"] = {
            "quadrature": _ratio_y_systematic
        }

    return h_to_rebin

def n_sigma_stat_from_unity(
    values: npt.NDArray[np.float64],
    stat_uncertainty: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate the number of sigma from unity for a given value and its statistical uncertainty.

    Args:
        values: The values for which to calculate the number of sigma from unity.
        stat_uncertainty: The statistical uncertainty on the values.

    Returns:
        The number of sigma from unity for each value.
    """
    return np.abs(values - 1) / stat_uncertainty  # type: ignore[no-any-return]
