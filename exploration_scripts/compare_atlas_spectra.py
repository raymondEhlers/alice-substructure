""" Compare ATLAS hadron RAA spectra.

Plot on a log-log scale so we can compare more easily.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import pandas
from pachyderm import binned_data


pachyderm.plot.configure()


def read_data(base_dir: Path, filename: str) -> binned_data.BinnedData:
    df = pandas.read_csv(base_dir / filename)
    df.columns = ["x", "y", "-dx", "+dx", "-dy", "+dy"]

    bin_edges = list(df["x"] - df["-dx"]) + [list(df["x"] + df["+dx"])[-1]]
    return binned_data.BinnedData(
        axes=[bin_edges],
        values=df["y"],
        variances=np.zeros_like(df["y"]),
        metadata={"y_systematic_errors": df["-dy"].to_numpy()},
    )


def plot() -> None:
    base_dir = Path("../../jetscape/js-an/inputData/atlas/")
    hadron_RAA_central = read_data(base_dir=base_dir, filename="hadronRaa0-5.csv")
    hadron_RAA_semi_central = read_data(base_dir=base_dir, filename="hadronRaa30-40.csv")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use error boxes instead to match style...
    p_central = ax.errorbar(
        hadron_RAA_central.axes[0].bin_centers,
        hadron_RAA_central.values,
        # yerr=hadron_RAA_central.metadata["y_systematic_errors"],
        # xerr=hadron_RAA_central.axes[0].bin_widths / 2,
        color="red",
        marker=".",
        linestyle="",
        label=r"0-5%",
    )
    pachyderm.plot.error_boxes(
        ax=ax,
        x_data=hadron_RAA_central.axes[0].bin_centers,
        y_data=hadron_RAA_central.values,
        x_errors=hadron_RAA_central.axes[0].bin_widths / 2,
        y_errors=hadron_RAA_central.metadata["y_systematic_errors"],
        color=p_central[0].get_color(),
        linewidth=0,
        # label = "Background", color = plot_base.AnalysisColors.fit,
    )
    p_semi_central = ax.errorbar(
        hadron_RAA_semi_central.axes[0].bin_centers,
        hadron_RAA_semi_central.values,
        # yerr=hadron_RAA_semi_central.metadata["y_systematic_errors"],
        # xerr=hadron_RAA_semi_central.axes[0].bin_widths / 2,
        color="green",
        marker=".",
        linestyle="",
        label=r"30-40%",
    )
    pachyderm.plot.error_boxes(
        ax=ax,
        x_data=hadron_RAA_semi_central.axes[0].bin_centers,
        y_data=hadron_RAA_semi_central.values,
        x_errors=hadron_RAA_semi_central.axes[0].bin_widths / 2,
        y_errors=hadron_RAA_semi_central.metadata["y_systematic_errors"],
        color=p_semi_central[0].get_color(),
        linewidth=0,
        # label = "Background", color = plot_base.AnalysisColors.fit,
    )

    ax.set_xlabel(r"$p_{\text{T}}\:(\text{GeV}/c)$")
    ax.set_ylabel(r"$R_{\text{AA}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1, 300])
    ax.set_ylim([0.09, 1.5])
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig("atlas_hadron_RAA_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot()
