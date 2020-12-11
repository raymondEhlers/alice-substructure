from functools import reduce

import attr
import numpy as np
from pachyderm import binned_data

from jet_substructure.base import helpers


@attr.s
class AsymmetricErrors:
    low: np.ndarray = attr.ib()
    high: np.ndarray = attr.ib()


@attr.s
class ErrorInput:
    value: np.ndarray = attr.ib()
    error: np.ndarray = attr.ib()


def relative_error(*inputs: ErrorInput) -> np.ndarray:
    if len(inputs) == 0:
        raise ValueError("Must pass at least one ErrorInput")
    if len(inputs) > 1:
        relative_error_squared = reduce(lambda x, y: ((x.error / x.value) ** 2) + ((y.error / y.value) ** 2), inputs)  # type: ignore
    else:
        relative_error_squared = (inputs[0].error / inputs[0].value) ** 2
    return np.sqrt(relative_error_squared)


def select_hist_range(hist: binned_data.BinnedData, x_range: helpers.RangeSelector) -> binned_data.BinnedData:
    # Sanity check
    if len(hist.axes) > 1:
        raise ValueError("Can only handle 1D histogram")

    bin_center_mask = (hist.axes[0].bin_centers >= x_range.min) & (hist.axes[0].bin_centers <= x_range.max)
    first_bin_edge = np.where(bin_center_mask)[0][0]
    last_bin_edge = -1 * np.where(bin_center_mask[::-1])[0][0]

    # Handle metadata
    metadata = {}
    for k, v in hist.metadata.items():
        if isinstance(v, AsymmetricErrors):
            metadata[k] = AsymmetricErrors(
                low=v.low[bin_center_mask],
                high=v.high[bin_center_mask],
            )

    return binned_data.BinnedData(
        axes=[hist.axes[0].bin_edges[first_bin_edge:last_bin_edge]],
        values=hist.values[bin_center_mask],
        variances=hist.variances[bin_center_mask],
        metadata=metadata,
    )
