""" Plot existing unfolded results.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from functools import reduce
from pathlib import Path
from typing import Dict, Sequence

import attr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import uproot
from pachyderm import binned_data

import jet_substructure.analysis.plot_base as pb
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


@attr.s
class Result:
    data: binned_data.BinnedData = attr.ib()
    pythia: binned_data.BinnedData = attr.ib()


@attr.s
class AsymmetricErrors:
    low: np.ndarray = attr.ib()
    high: np.ndarray = attr.ib()


@attr.s
class ErrorInput:
    value: np.ndarray = attr.ib()
    error: np.ndarray = attr.ib()


def relative_error(*inputs: ErrorInput) -> np.ndarray:
    relative_error_squared = reduce(lambda x: (x.error / x.value) ** 2, inputs)  # type: ignore
    return np.sqrt(relative_error_squared)


def select_hist_in_range(hist: binned_data.BinnedData, x_range: helpers.RangeSelector) -> binned_data.BinnedData:
    bin_center_mask = (hist.axes[0].bin_centers >= x_range.min) & (hist.axes[0].bin_centers <= x_range.max)
    first_bin_edge = np.where(bin_center_mask)[0][0]
    last_bin_edge = -1 * np.where(bin_center_mask[::-1])[0][0]

    # Handle metadata
    metadata = {}
    for k, v in hist.metadata.items():
        if isinstance(v, AsymmetricErrors):
            metadata[k] = AsymmetricErrors(low=v.low[bin_center_mask], high=v.high[bin_center_mask],)

    return binned_data.BinnedData(
        axes=[hist.axes[0].bin_edges[first_bin_edge:last_bin_edge]],
        values=hist.values[bin_center_mask],
        variances=hist.variances[bin_center_mask],
        metadata=metadata,
    )


def get_unfolded_pp_data_leticia(grooming_method: str, x_range: helpers.RangeSelector) -> Result:
    # Map to Leticia's filenames.
    grooming_method_map = {
        "dynamical_kt": "dynamickt",
        "dynamical_time": "dynamictf",
        "leading_kt": "leadingktnocut",
        "leading_kt_z_cut_02": "leadingktzcut02",
        "leading_kt_z_cut_04": "leadingktzcut04",
    }

    input_path = Path("output/pp/unfolding/leticia")
    f = uproot.open(input_path / f"result_{grooming_method_map[grooming_method]}.root")

    # This is the main data, giving the statistical uncertaintines.
    # For some reason, the name sometimes changes (apparently for the z_cut)
    hist_name = "shape_0" if "shape_0" in f else "shapeR"
    hist = f[hist_name]
    data = binned_data.BinnedData.from_existing_data(hist)

    # Now, extract the errors from the graph and store them with the data.
    systematics = f["Graph"]
    # The x errors are just the bin widths, so we don't extract them here.
    # The y errors are asymmetric.
    data.metadata["y_systematic_errors"] = AsymmetricErrors(low=systematics.yerrorslow, high=systematics.yerrorshigh)

    # Only keep the data within the selected range.
    data = select_hist_in_range(hist=data, x_range=x_range)

    # Next, get pythia (for conveience, we've just stored it in the same file. Later we can do better).
    pythia = binned_data.BinnedData.from_existing_data(f["true1"])
    pythia = select_hist_in_range(hist=pythia, x_range=x_range)

    return Result(data=data, pythia=pythia)


def plot_comparison_pythia(
    hists: Dict[str, Result], grooming_methods: Sequence[str], plot_config: pb.PlotConfig, output_dir: Path,
) -> None:
    # Setup
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    grooming_styles = pb.define_grooming_styles()

    for grooming_method in grooming_methods:
        # Setup
        style = grooming_styles[grooming_method]
        result = hists[grooming_method]
        plotting_last_method = grooming_method == grooming_methods[-1]

        # Plot options
        kwargs = {
            "markerfacecolor": "white" if style.fillstyle == "none" else style.color,
            "alpha": 1 if style.fillstyle == "none" else 0.8,
        }
        if style.fillstyle != "none":
            kwargs["markeredgewidth"] = 0

        ax.errorbar(
            result.data.axes[0].bin_centers,
            result.data.values,
            yerr=result.data.errors,
            xerr=result.data.axes[0].bin_widths / 2,
            color=style.color,
            marker=style.marker,
            fillstyle=style.fillstyle,
            linestyle="",
            label=style.label,
            zorder=style.zorder,
            **kwargs,
        )
        # Systematic errors.
        y_systematic_errors = result.data.metadata["y_systematic_errors"]
        pachyderm.plot.error_boxes(
            ax=ax,
            x_data=result.data.axes[0].bin_centers,
            y_data=result.data.values,
            x_errors=result.data.axes[0].bin_widths / 2,
            y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
            color=style.color,
            linewidth=0,
            # label = "Background", color = plot_base.AnalysisColors.fit,
        )

        # Pythia comparison. Use the compare style
        pythia_style = grooming_styles[f"{grooming_method}_compare"]

        pythia = result.pythia

        # Plot options
        # kwargs = {
        #    "markerfacecolor": "white" if pythia_style.fillstyle == "none" else pythia_style.color,
        #    "alpha": 1 if pythia_style.fillstyle == "none" else 0.8,
        # }
        # if pythia_style.fillstyle != "none":
        #    kwargs["markeredgewidth"] = 0

        # ax.errorbar(
        #    pythia.axes[0].bin_centers,
        #    pythia.values,
        #    # yerr=pythia.errors,
        #    # xerr=pythia.axes[0].bin_widths / 2,
        #    color=pythia_style.color,
        #    # marker=pythia_style.marker,
        #    fillstyle=pythia_style.fillstyle,
        #    # linestyle="",
        #    linewidth=3,
        #    #label=f"Pythia, {pythia_style.label}",
        #    label="Pythia 8 Monash 2013" if plotting_last_method else None,
        #    zorder=pythia_style.zorder,
        #    alpha=0.7,
        # )
        ax.errorbar(
            pythia.axes[0].bin_centers,
            pythia.values,
            # yerr=pythia.errors,
            # xerr=pythia.axes[0].bin_widths / 2,
            color=style.color,
            # marker=style.marker,
            fillstyle=style.fillstyle,
            # linestyle="",
            linewidth=3,
            # label=f"Pythia, {pythia_style.label}",
            label="Pythia 8 Monash 2013" if plotting_last_method else None,
            zorder=pythia_style.zorder,
            alpha=0.7,
        )

        # TODO: Systematic errors for ratios.
        ratio = pythia / result.data
        ax_ratio.errorbar(
            ratio.axes[0].bin_centers,
            ratio.values,
            yerr=ratio.errors,
            xerr=ratio.axes[0].bin_widths / 2,
            color=style.color,
            marker=style.marker,
            fillstyle=style.fillstyle,
            linestyle="",
            zorder=style.zorder,
            **kwargs,
        )

    # Reference value for ratio
    ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Apply the PlotConfig
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])
    # A few additional tweaks.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    # filename = f"{plot_config.name}_{jet_pt_bin}{grooming_methods_filename_label}_{identifiers}_iterative_splittings"
    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_comparison(hists: Dict[str, Result], grooming_methods: Sequence[str], comparison_grooming_method: str) -> None:
    ...
    fig, ax = plt.subplots(figsize=(8, 6))


def run() -> None:
    # Setup
    helpers.setup_logging()
    output_dir = Path("output/pp/unfolding/leticia/plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, retrieve the have the grooming methods.
    x_range = helpers.RangeSelector(0.5, 8)
    dynamical_kt = get_unfolded_pp_data_leticia(grooming_method="dynamical_kt", x_range=x_range)
    dynamical_time = get_unfolded_pp_data_leticia(grooming_method="dynamical_time", x_range=x_range)
    leading_kt = get_unfolded_pp_data_leticia(grooming_method="leading_kt", x_range=x_range)
    leading_kt_z_cut_02 = get_unfolded_pp_data_leticia(grooming_method="leading_kt_z_cut_02", x_range=x_range)
    hists = {
        "dynamical_kt": dynamical_kt,
        "dynamical_time": dynamical_time,
        "leading_kt": leading_kt,
        "leading_kt_z_cut_02": leading_kt_z_cut_02,
    }

    logger.info(locals())
    # import IPython; IPython.embed()

    jet_pt_bin = helpers.RangeSelector(60, 80)
    text = pb.label_to_display_string["ALICE"]["preliminary"]
    text += "\n" + pb.label_to_display_string["collision_system"]["pp_5TeV"]
    text += "\n" + pb.label_to_display_string["jets"]["R04"]
    text += "\n" + fr"${jet_pt_bin.display_str(label='ch')}\:\text{{GeV}}/c$"
    plot_comparison_pythia(
        hists=hists,
        grooming_methods=list(hists.keys()),
        plot_config=pb.PlotConfig(
            name="kt_pythia_comparison",
            panels=[
                # Main axis.
                pb.Panel(
                    axes=pb.AxisConfig(
                        "y",
                        label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                        log=True,
                        range=(7e-3, 1),
                    ),
                    text=pb.TextConfig(x=0.96, y=0.96, text=text),
                    legend=pb.LegendConfig(location="lower left", anchor=(0.02, 0.02)),
                ),
                # Ratio.
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(-0.1, 8.6)),
                        pb.AxisConfig("y", label="Pythia/data", range=(0.65, 1.35)),
                    ]
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.12, "top": 0.975, "right": 0.975}),
        ),
        output_dir=output_dir,
    )
    plot_comparison(hists=hists, grooming_methods=list(hists.keys()), comparison_grooming_method="leading_kt")


if __name__ == "__main__":
    run()
