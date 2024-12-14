""" Additional analysis methods for jet substructure.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
import typing
from typing import Any

import numpy as np
import numpy.typing as npt

import pachyderm.fit
from jet_substructure.base.helpers import UprootArray
from pachyderm import binned_data, histogram

logger = logging.getLogger(__name__)


@typing.overload
def power_law(x: float, a: float, n: float) -> float:
    ...


@typing.overload
def power_law(x: UprootArray[float], a: float, n: float) -> UprootArray[float]:
    ...


def power_law(
    x: UprootArray[float] | npt.NDArray[np.float64] | float, a: float, n: float
) -> UprootArray[float] | npt.NDArray[np.float64] | float:
    return (a * x) ** -n


class PowerLaw(pachyderm.fit.Fit):
    """ Power law fit. """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.fit_function = power_law  # type: ignore[assignment]

    def _post_init_validation(self) -> None:
        """ Validate that the fit object was setup properly. """

    def _setup(self, h: histogram.Histogram1D) -> tuple[histogram.Histogram1D, pachyderm.fit.T_FitArguments]:
        """ Setup the histogram and arguments for the fit. """
        logger.debug(f"h.x: {h.x}")
        return h, {"n": 1, "limit_n": (0.001, 10), "error_n": 0.1, "a": 1, "limit_a": (1e-4, 1e4), "error_a": 0.1}


def fit_kt_spectrum(kt_spectra: binned_data.BinnedData) -> PowerLaw:
    """Fit a given kt spectrum to a power law.

    Args:
        kt_spectrum: kt spectrum to be fit.
    Returns:
        Fit object (containing the fit result)
    """
    power_law = PowerLaw(use_log_likelihood=True)
    power_law.fit_result = power_law.fit(kt_spectra.to_histogram1D())
    logger.info(
        fr"kt^({power_law.fit_result.values_at_minimum['p']:.02} \pm {power_law.fit_result.errors_on_parameters['p']:.02})"
    )

    return power_law
