""" Functionality related to fitting the paper plots.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import functools
import inspect
import logging
from pathlib import Path
from typing import Any, Protocol

import attrs
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from pachyderm import binned_data
from scipy import optimize

logger = logging.getLogger(__name__)


def _quadratic_polynomial(
    x: npt.NDArray[np.float64] | float, amplitude: float, shift: float, intercept: float
) -> npt.NDArray[np.float64] | float:
    """Simplified quadratic polynomial.

    ..math::

        f = A * (x - shift)^2 + intercept

    Args:
        x: Value(s) where the polynomial should be evaluated.
        amplitude: Amplitude of the polynomial.
        shift: Shift of the $x$ argument in the quadratic term.
        intercept: Intercept of the polynomial.
    Returns:
        Evaluated function
    """
    return amplitude * (x - shift) ** 2 + intercept


def _power_law(
    x: npt.NDArray[np.float64] | float, power: float, amplitude: float, intercept: float
) -> npt.NDArray[np.float64] | float:
    """Power law

    ..math::

        f = (amplitude * x)^-power + intercept

    Args:
        x: Value(s) where the power law should be evaluated.
        power: Power of the power law.
        amplitude: Amplitude of the power law.
        intercept: Intercept of the power law.
    Returns:
        Evaluated function
    """
    return (amplitude * x) ** -power + intercept


def _tanh_scale(x: npt.NDArray[np.float64] | float, x0: float, scale: float) -> npt.NDArray[np.float64] | float:
    """Tanh scaling function.

    ..math::

            f = (1 + \tanh((x - x0) / scale)) / 2

    Args:
        x: Value(s) where the tanh scaling function should be evaluated.
        x0: Location of the tanh scaling turnon.
        scale: Scale of the tanh scaling function. Sharpness of the switch from on to off.
            To flip the scaling, use a negative value.
    Returns:
        Evaluated function
    """
    return (1 + np.tanh((x - x0) / scale)) / 2


def spectra_fit_function_with_y_scale(
    x: npt.NDArray[np.float64] | float,
    amplitude: float,
    shift: float,
    intercept: float,
    power_law: float,
    power_law_amp: float,
    x0: float = 1.25,
    tanh_transition_scale: float = 0.1,
) -> npt.NDArray[np.float64] | float:
    """Nominal fit function with a y-scale parameter.

    For simple comparisons, this seems to be able to describe the high kt range more effectively than the nominal.

    Args:
        x: Value(s) where the fit function should be evaluated.
        amplitude: Amplitude of the quadratic polynomial.
        shift: Shift of the $x$ argument in the quadratic term.
        intercept: Intercept of the quadratic polynomial. It serves as the y-scale.
        power_law: Power of the power law.
        power_law_amp: Amplitude of the power law.
        x0: Location of the tanh scaling turnon.
        tanh_transition_scale: Scale of the tanh scaling function. Sharpness of the switch from on to off.
            To flip the scaling, use a negative value.

    Returns:
        Evaluated function
    """
    # NOTE: We want to the negative on the _tanh_scale for the quadratic polynomial because it needs to contribute
    #       at low ktg and then turn off at high ktg.
    # NOTE: We consistently pass intercept = 0 because it's convenient conceptually to consider the intercept as
    #       being related to the anchor point at x0. This is what I refer to as the "y-scale" in the function title, etc.
    return (
        _quadratic_polynomial(x, amplitude, shift, intercept=0)
        + (intercept - _quadratic_polynomial(x0, amplitude, shift, intercept=0))
    ) * _tanh_scale(x, x0, -tanh_transition_scale) + (
        _power_law(x, power_law, power_law_amp, intercept=0)
        + (intercept - _power_law(x0, power_law, power_law_amp, intercept=0))
    ) * _tanh_scale(x, x0, tanh_transition_scale)


def spectra_fit_function_without_y_scale(
    x: npt.NDArray[np.float64] | float,
    amplitude: float,
    shift: float,
    intercept: float,
    power_law: float,
    power_law_amp: float,
    x0: float = 1.25,
    tanh_transition_scale: float = 0.4,
) -> npt.NDArray[np.float64] | float:
    """Alternative spectra fit function.

    Somewhat simpler than the other functional form, but it doesn't seem to describe the high kt range quite as well.

    Args:
        x: Value(s) where the fit function should be evaluated.
        amplitude: Amplitude of the quadratic polynomial.
        shift: Shift of the $x$ argument in the quadratic term.
        intercept: Intercept of the quadratic polynomial.
        power_law: Power of the power law.
        power_law_amp: Amplitude of the power law.
        x0: Location of the tanh scaling turnon.
        tanh_transition_scale: Scale of the tanh scaling function. Sharpness of the switch from on to off.
            To flip the scaling, use a negative value.

    Returns:
        Evaluated function
    """
    # NOTE: Passing intercept=0 to the power_law is intentional! Otherwise, the coupled arguments don't seem
    #       to work properly (see the y_scale version for an alternative which uses the argument)
    # NOTE: We want to the negative on the _tanh_scale for the quadratic polynomial because it needs to contribute
    #       at low ktg and then turn off at high ktg.
    return (_quadratic_polynomial(x, amplitude, shift, intercept)) * _tanh_scale(x, x0, -tanh_transition_scale) + (
        _power_law(x, power_law, power_law_amp, intercept=0)
    ) * _tanh_scale(x, x0, tanh_transition_scale)


class F(Protocol):
    __name__: str
    def __call__(self, x: npt.NDArray[np.float64] | float, **kwargs: Any) -> npt.NDArray[np.float64] | float: ...


@attrs.define
class FitResult:
    f: F
    parameters: dict[str, float]
    metadata: dict[str, str] = attrs.field(factory=dict)

    @property
    def popt(self) -> list[float]:
        return list(self.parameters.values())

    def kw_parameters_by_name(self, names: list[str]) -> dict[str, float]:
        return {
            k: self.parameters[k]
            for k in names
        }

    def parameters_by_name(self, names: list[str]) -> list[float]:
        return list(self.kw_parameters_by_name(names=names).values())

    def __call__(self, x: npt.NDArray[np.float64] | float, **kwargs: Any) -> npt.NDArray[np.float64] | float:
        # Allow the possibility of replacing arguments
        params = self.parameters | kwargs
        return self.f(x, **params)


@attrs.define
class FitFunction:
    f: F
    initial_arguments: dict[str, float]
    metadata: dict[str, Any] = attrs.field(factory=dict)

    def fit_to_histogram(self, h: binned_data.BinnedData) -> FitResult:
        popt, _ = optimize.curve_fit(
            self.f,
            h.axes[0].bin_centers,
            h.values,
            p0=list(self.initial_arguments.values()),
            maxfev=500000,
        )
        function_argument_names = [
            v.name
            for v in inspect.signature(self.f).parameters.values()
            if v.default is v.empty
        ]
        # Skip the x argument, and then everything after we optimized.
        function_argument_names = function_argument_names[1:]
        # Cross check that we have the arguments that we expect.
        assert list(self.initial_arguments) == function_argument_names, f"Argument mismatch! Expected {list(self.initial_arguments)=}, but got {function_argument_names=}"

        return FitResult(
            f=self.f,
            parameters={
                k: float(v)
                for k, v in zip(function_argument_names, popt)
            },
            metadata=self.metadata
        )


def create_fit_function(
    x0: float,
    tanh_transition_scale: float,
    h: binned_data.BinnedData,
    initial_arguments: dict[str, float],
    disable_y_scale: bool = False,
) -> FitFunction:
    """Determine the fit function and initial arguments based on the histogram.

    Args:
        x0: Location of the tanh scaling turnon.
        tanh_transition_scale: Scale of the tanh scaling function. Sharpness of the switch from on to off.
        h: Histogram to fit.
        initial_arguments: Initial arguments for the fit. The keys are the argument names, and the values are
            the initial values.
        disable_y_scale: If true, disable the y-scale parameter. Default: False.

    Returns:
        Fit function class based on the provided configuration.
    """
    using_power_law_only = False
    if min(h.axes[0].bin_centers) > x0:
        logger.info("Using power law only")
        using_power_law_only = True
        # Fix up the args for the power law only case
        initial_arguments = {
            "power": initial_arguments["power_law"],
            "amplitude": initial_arguments["power_law_amp"],
            "intercept": initial_arguments["intercept"],
        }
        # Only use the power low. We need to reduce the number of parameters since there
        # are a restricted number of points.
        fit_func: F = _power_law  # type: ignore[assignment]
        if disable_y_scale:
            initial_arguments["intercept"] = 0
    else:
        if disable_y_scale:  # noqa: SIM108
            fit_func = spectra_fit_function_without_y_scale  # type: ignore[assignment]
        else:
            fit_func = spectra_fit_function_with_y_scale  # type: ignore[assignment]
        original_name = fit_func.__name__
        fit_func = functools.partial(fit_func, x0=x0, tanh_transition_scale=tanh_transition_scale)  # type: ignore[assignment]
        # This is a hack, but it lets us keep the original name for the fit function,
        # which is what we really need. So good enough.
        fit_func.__name__ = original_name
    return FitFunction(
        f=fit_func,
        initial_arguments=initial_arguments,
        metadata={"using_power_law_only": using_power_law_only}
    )


def fit_spectra(
    x0: float,
    tanh_transition_scale: float,
    h: binned_data.BinnedData,
    initial_arguments: dict[str, float],
    disable_y_scale: bool,
) -> FitResult:
    """Fit a function to a histogram.

    Args:
        x0: Location of the tanh scaling turnon.
        tanh_transition_scale: Scale of the tanh scaling function. Sharpness of the switch from on to off.
        h: Histogram to fit.
        fit_func: Function to fit to the histogram.
        initial_arguments: Initial arguments for the fit.

    Returns:
        Fit parameters.
    """
    fit_function = create_fit_function(
        x0=x0,
        tanh_transition_scale=tanh_transition_scale,
        h=h,
        initial_arguments=initial_arguments,
        disable_y_scale=disable_y_scale,
    )
    return fit_function.fit_to_histogram(h=h)


def fit_and_plot(
    x0: float,
    tanh_transition_scale: float,
    h: binned_data.BinnedData,
    disable_y_scale: bool,
    initial_arguments: dict[str, float],
    ax: mpl.axes.Axes,
    x_for_plotting: npt.NDArray[np.float64],
    plot_label: str,
    fit_result: FitResult | None = None,
    ax_ratio: mpl.axes.Axes | None = None,
    plot_components: bool = False,
    plot_components_without_y_scale: bool = False,
) -> FitResult:
    """Fit a function to a histogram and plot the result.

    Practically used for calibrating the fit functions and their parameters. If using for the paper,
    probably better to perform the fit manually.

    Args:
        x0: Location of the tanh scaling turnon.
        tanh_transition_scale: Scale of the tanh scaling function. Sharpness of the switch from on to off.
        h: Histogram to fit.
        fit_func: Function to fit to the histogram.
        initial_arguments: Initial arguments for the fit.
        ax: Axes to plot the fit on.
        x_for_plotting: x values to use for plotting the fit.
        plot_label: Label to use for the fit.
        fit_result: Fit result. If None, the fit is performed here.
        ax_ratio: Axes to plot the ratio on. If None, no ratio is plotted.
        plot_components: If true, plot the individual components of the fit function.
        plot_components_without_y_scale: If true, plot the individual components of the fit function without
            the y-scaling. Default: False.

    Returns:
        Fit parameters.
    """
    # Setup
    fit_func = create_fit_function(
        x0=x0,
        tanh_transition_scale=tanh_transition_scale,
        h=h,
        initial_arguments=initial_arguments,
        disable_y_scale=disable_y_scale,
    )
    # Allow the user to pass in the fit parameters (eg. then this function is plotting only),
    # or to perform the fit here.
    if fit_result is None:
        # For these functions, some reasonable initial arguments are: [-1, 1, 1, 3, 1],
        fit_result = fit_func.fit_to_histogram(h=h)

    # Main plot
    p = ax.plot(
        x_for_plotting,
        fit_result(x_for_plotting),
        linestyle="--",
        linewidth=3,
        zorder=10,
        label=plot_label,
    )
    # Plot components
    if plot_components:
        using_power_law_only = fit_result.metadata["using_power_law_only"]
        # Named under the combined function
        power_law_kw_args = ["power_law", "power_law_amp"]
        if using_power_law_only:
            # Name when using the power law only
            power_law_kw_args = ["power", "amplitude"]

        if plot_components_without_y_scale:
            # Quadratic term
            if not using_power_law_only:
                ax.plot(
                    x_for_plotting,
                    _quadratic_polynomial(x_for_plotting, **fit_result.kw_parameters_by_name(["amplitude", "shift"]), intercept=0),
                    linestyle="--",
                    linewidth=3,
                    zorder=10,
                    label=f"{plot_label} (Poly)",
                )
            # Power law term
            ax.plot(
                x_for_plotting,
                _power_law(x_for_plotting, **fit_result.kw_parameters_by_name(power_law_kw_args), intercept=0),
                linestyle="--",
                linewidth=3,
                zorder=10,
                label=f"{plot_label} (PL)",
            )
        else:
            # Quadratic term
            if not using_power_law_only:
                ax.plot(
                    x_for_plotting,
                    (
                        _quadratic_polynomial(x_for_plotting, **fit_result.kw_parameters_by_name(["amplitude", "shift"]), intercept=0)
                        + fit_result.parameters_by_name(["intercept"])  # type: ignore[operator]
                        - _quadratic_polynomial(x0, **fit_result.kw_parameters_by_name(["amplitude", "shift"]), intercept=0)
                    ),
                    linestyle="--",
                    linewidth=3,
                    zorder=10,
                    label=f"{plot_label} (Poly)",
                )
            ax.plot(
                x_for_plotting,
                (
                    _power_law(x_for_plotting, *fit_result.parameters_by_name(power_law_kw_args), intercept=0)  # type: ignore[misc]
                    + fit_result.parameters_by_name(["intercept"])  # type: ignore[operator]
                    - _power_law(x0, *fit_result.parameters_by_name(power_law_kw_args), intercept=0)  # type: ignore[misc]
                ),
                linestyle="--",
                linewidth=3,
                zorder=10,
                label=f"{plot_label} (PL)",
            )

    # Ratio plot
    if ax_ratio:
        ax_ratio.plot(
            h.axes[0].bin_centers,
            fit_result(h.axes[0].bin_centers) / h.values,
            marker="o",
            linewidth=0,
            zorder=10,
            label=plot_label,
            color=p[0].get_color(),
        )

    return fit_result


def write_fit_result(
    fit_result: FitResult,
    x0: float,
    tanh_transition_scale: float,
    output_path: Path,
) -> None:
    """Write fit results to a file.

    Args:
        fit_func: Function to fit to the histogram.
        popt: Fit parameters.
        x0: Center of the tanh scaling function. Location of the switch from on to off.
        tanh_transition_scale: Scale of the tanh scaling function. Sharpness of the switch from on to off.
            To flip the scaling, use a negative value.
        output_path: Path to write the results to.
    """
    import pachyderm.yaml

    y = pachyderm.yaml.yaml()
    fit_params = {
        "x0": x0,
        "tanh_transition_scale": tanh_transition_scale,
        **fit_result.parameters,
    }
    with output_path.open("w") as f:
        f.write(f"# Fit function: {fit_result.f.__name__}\n")
        y.dump(fit_params, f)
