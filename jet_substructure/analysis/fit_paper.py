""" Functionality related to fitting the paper plots.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

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


def fit_spectra(
    x0: float,
    tanh_transition_scale: float,
    h: binned_data.BinnedData,
    fit_func: callable[[npt.NDArray[np.float64] | float, ...], npt.NDArray[np.float64] | float],
    initial_arguments: list[float],
) -> npt.NDArray[np.float64]:
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
    popt, _ = optimize.curve_fit(
        functools.partial(fit_func, x0=x0, tanh_transition_scale=tanh_transition_scale),
        h.axes[0].bin_centers,
        h.values,
        p0=initial_arguments,
        maxfev=500000,
    )
    return popt


def fit_and_plot(
    x0: float,
    tanh_transition_scale: float,
    h: binned_data.BinnedData,
    fit_func: callable[[npt.NDArray[np.float64] | float, ...], npt.NDArray[np.float64] | float],
    initial_arguments: list[float],
    ax: mpl.axes.Axes,
    x_for_plotting: npt.NDArray[np.float64],
    plot_label: str,
    ax_ratio: mpl.axes.Axes | None = None,
    plot_components: bool = False,
    plot_components_without_y_scale: bool = False,
    popt: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
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
        ax_ratio: Axes to plot the ratio on. If None, no ratio is plotted.
        plot_components: If true, plot the individual components of the fit function.
        plot_components_without_y_scale: If true, plot the individual components of the fit function without
            the y-scaling. Default: False.
        popt: Fit parameters. If None, the fit is performed here.

    Returns:
        Fit parameters.
    """
    # Allow the user to pass in the fit parameters (eg. then this function is plotting only),
    # or to perform the fit here.
    if popt is None:
        # For these functions, some reasonable initial arguments are: [-1, 1, 1, 3, 1],
        popt = fit_spectra(
            x0=x0,
            tanh_transition_scale=tanh_transition_scale,
            h=h,
            fit_func=fit_func,
            initial_arguments=initial_arguments,
        )

    # Main plot
    p = ax.plot(
        x_for_plotting,
        fit_func(x_for_plotting, *popt),
        linestyle="--",
        linewidth=3,
        zorder=10,
        label=plot_label,
    )
    # Plot components
    if plot_components:
        if plot_components_without_y_scale:
            ax.plot(
                x_for_plotting,
                _quadratic_polynomial(x_for_plotting, *popt[0:2], intercept=0),
                linestyle="--",
                linewidth=3,
                zorder=10,
                label=f"{plot_label} (Poly)",
            )
            ax.plot(
                x_for_plotting,
                _power_law(x_for_plotting, *popt[3:5], intercept=0),
                linestyle="--",
                linewidth=3,
                zorder=10,
                label=f"{plot_label} (PL)",
            )
        else:
            ax.plot(
                x_for_plotting,
                _quadratic_polynomial(x_for_plotting, *popt[0:2], intercept=0)
                + popt[2]
                - _quadratic_polynomial(x0, *popt[0:2], intercept=0),
                linestyle="--",
                linewidth=3,
                zorder=10,
                label=f"{plot_label} (Poly)",
            )
            ax.plot(
                x_for_plotting,
                _power_law(x_for_plotting, *popt[3:5], intercept=0) + popt[2] - _power_law(x0, *popt[3:5], intercept=0),
                linestyle="--",
                linewidth=3,
                zorder=10,
                label=f"{plot_label} (PL)",
            )

    # Ratio plot
    if ax_ratio:
        ax_ratio.plot(
            h.axes[0].bin_centers,
            fit_func(h.axes[0].bin_centers, *popt) / h.values,
            marker="o",
            linewidth=0,
            zorder=10,
            label=plot_label,
            color=p[0].get_color(),
        )

    return popt


def write_fit_result(
    fit_func: callable[[npt.NDArray[np.float64] | float, ...], npt.NDArray[np.float64] | float],
    popt: npt.NDArray[np.float64],
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
        "amplitude": popt[0],
        "shift": popt[1],
        "intercept": popt[2],
        "power_law": popt[3],
        "power_law_amp": popt[4],
    }
    with output_path.open("w") as f:
        f.write(f"# Fit function: {fit_func.__name__}\n")
        y.dump(fit_params, f)
