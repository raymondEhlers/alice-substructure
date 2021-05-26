import time
from typing import Any, Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import seaborn as sns
from scipy.interpolate import interp1d


pachyderm.plot.configure()


def inverse_sample_decorator(
    distribution: Callable[..., Union[np.ndarray, float]]
) -> Callable[..., Union[float, np.ndarray]]:
    """Decorator to perform inverse transform sampling.

    Based on: https://stackoverflow.com/a/64288861/12907985
    """

    def wrapper(
        n_samples: int, x_min: float, x_max: float, n_distribution_samples: int = 100_000, **kwargs: Any
    ) -> Union[np.ndarray, float]:
        x = np.linspace(x_min, x_max, int(n_distribution_samples))
        cumulative = np.cumsum(distribution(x, **kwargs))
        cumulative -= cumulative.min()
        # This is an inverse of the CDF
        # See: https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
        f = interp1d(cumulative / cumulative.max(), x)
        return f(np.random.random(n_samples))  # type: ignore

    return wrapper


def gaussian(x: Union[np.ndarray, float], mean: float, sigma: float) -> Union[np.ndarray, float]:
    r"""Normalized gaussian.

    .. math::

        f = 1 / \sqrt{2 * \pi * \sigma^{2}} * \exp{-\frac{(x - \mu)^{2}}{(2 * \sigma^{2}}}

    Args:
        x: Value(s) where the gaussian should be evaluated.
        mean: Mean of the gaussian distribution.
        sigma: Width of the gaussian distribution.
    Returns:
        Calculated gaussian value(s).
    """
    return 1.0 / np.sqrt(2 * np.pi * np.square(sigma)) * np.exp(-1.0 / 2.0 * np.square((x - mean) / sigma))  # type: ignore


sample_gaussian = inverse_sample_decorator(gaussian)


def test_against_root_gaussian() -> None:
    import ROOT

    n_samples = 5000
    mean = 1
    sigma = 0.5
    x_min = -10
    x_max = 10

    py_start = time.time()
    py_sample_gaussian = sample_gaussian(mean=mean, sigma=sigma, n_samples=n_samples, x_min=x_min, x_max=x_max)
    print(f"Python sampling took {time.time() - py_start}")

    root_start = time.time()
    root_gauss = ROOT.TF1("f1", "gaus(0)", x_min, x_max)
    # Amplitude
    root_gauss.SetParameter(0, 1)
    root_gauss.SetParameter(1, mean)
    root_gauss.SetParameter(2, sigma)
    # root_gauss.Print()
    root_random = [root_gauss.GetRandom(x_min, x_max) for _ in range(n_samples)]
    print(f"Root sampling took {time.time() - root_start}")

    with sns.color_palette("Set2"):
        fig, ax = plt.subplots(figsize=(8, 6))
        binning = np.linspace(-10, 10, 161)

        # First, the python
        ax.hist(
            py_sample_gaussian,
            bins=binning,
            alpha=0.7,
            density=True,
            label="python",
        )
        ax.hist(
            root_random,
            bins=binning,
            alpha=0.7,
            density=True,
            label="ROOT",
        )
        x = np.linspace(-10, 10, 321)
        ax.plot(
            x,
            gaussian(x, mean=mean, sigma=sigma),
            label="true",
        )

        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig("inverse_transform_sampling.pdf")


def toy_func(x: Union[float, np.ndarray]) -> np.ndarray:
    return x * np.exp(-x / 0.3)  # type: ignore


sample_toy = inverse_sample_decorator(toy_func)


def test_against_root_toy() -> None:
    import ROOT

    n_samples = int(5e5)
    x_min = 0
    x_max = 400

    py_start = time.time()
    py_sample_toy = sample_toy(n_samples=n_samples, x_min=x_min, x_max=x_max)
    print(f"Python sampling took {time.time() - py_start}")

    root_start = time.time()
    root_func = ROOT.TF1("f_pT", "x*exp(-x/0.3)", x_min, x_max)
    root_func.SetNpx(40000)
    root_random = [root_func.GetRandom(x_min, x_max) for _ in range(n_samples)]
    print(f"Root sampling took {time.time() - root_start}")

    with sns.color_palette("Set2"):
        fig, ax = plt.subplots(figsize=(8, 6))
        binning = np.linspace(0, 20, 161)

        # First, the python
        ax.hist(
            py_sample_toy,
            bins=binning,
            alpha=0.7,
            density=True,
            label="python",
        )
        ax.hist(
            root_random,
            bins=binning,
            alpha=0.7,
            density=True,
            label="ROOT",
        )
        # Skip 0 to avoid going to 0 on the scale.
        x = np.linspace(0.03125, 5, 160)
        values = toy_func(x)
        ax.plot(
            x,
            # Normalize to put on the same scale as the normalized hists.
            # Need the overall integral, which is the sum times integral bin width.
            values / np.sum(values) / (x[1] - x[0]),
            label="true",
        )

        ax.set_yscale("log")
        ax.set_xlim([None, 10.5])
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig("inverse_transform_sampling_toy.pdf")


if __name__ == "__main__":
    # test_against_root_gaussian()
    test_against_root_toy()
