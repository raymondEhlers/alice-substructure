#!/usr/bin/env python3

""" One-off plots for Hard Probes 2020.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict

import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import uproot
from pachyderm import binned_data

import jet_substructure.analysis.plot_base as pb
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


def plot_response(
    hists: Dict[str, binned_data.BinnedData],
    grooming_method: str,
    matching_suffix: str,
    hybrid_jet_pt_bin: helpers.RangeSelector,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    # Setup
    logger.debug(
        # f"Plotting {label} {response_type} response for {grooming_method}, {matching_type}, hybrid: {hybrid_jet_pt_bin}"
        f"Plotting kt response for {grooming_method}, hybrid: {hybrid_jet_pt_bin}"
    )

    hist_name = "hHybridTrueKtResponse"
    if matching_suffix:
        hist_name += matching_suffix

    # matches_label = " ".join(matching_type.split("_")).capitalize()
    # bh_input_hist = hists[f"{grooming_method}_{response_type}_{label}_response_matching_type_{matching_type}"]
    h = binned_data.BinnedData.from_existing_data(hists[hist_name])

    # Select the variables (for the example of kt)
    # Axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt
    # NOTE: We already applied the 40 < hybrid jet pt < 120 cut, so it doesn't need an additional selection.
    # h = binned_data.BinnedData(
    #    axes=[h_input.axes[1], h_input.axes[3]],
    #    values=np.sum(h_input.values, axis=(0, 2)),
    #    variances=np.sum(h_input.variances, axis=(0, 2)),
    # )

    # Normalize the response.
    normalization_values = h.values.sum(axis=0, keepdims=True)
    h.values = np.divide(h.values, normalization_values, out=np.zeros_like(h.values), where=normalization_values != 0)

    # Finish setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        # "vmin": h_proj.values[h_proj.values > 0].min(),
        "vmin": 1e-4,
        # "vmax": h.values.max(),
        "vmax": 1,
    }

    # Plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T, h.axes[1].bin_edges.T, h.values.T, norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Labeling and presentation
    # Help out mypy...
    assert plot_config.panels[0].text is not None
    # plot_config.panels[0].text.text += "\n" + matches_label + " matches"
    plot_config.apply(fig=fig, ax=ax)

    # Store and cleanup
    matching_type = "matching_type_"
    if matching_suffix:
        matching_type += matching_suffix
    else:
        matching_type += "all"
    filename = f"{plot_config.name}_iterative_splittings_{grooming_method}_{matching_type}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_1D_comparison(
    hists: Dict[str, binned_data.BinnedData],
    grooming_method: str,
    matching_suffix: str,
    hybrid_jet_pt_bin: helpers.RangeSelector,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    # Setup
    logger.debug(
        # f"Plotting {label} {response_type} response for {grooming_method}, {matching_type}, hybrid: {hybrid_jet_pt_bin}"
        f"Plotting 1D kt for {grooming_method}, hybrid: {hybrid_jet_pt_bin}"
    )
    grooming_styles = pb.define_grooming_styles()

    hist_name = "hHybridTrueKtResponse"
    if matching_suffix:
        hist_name += matching_suffix

    # Get hist and project.
    bh_hist = hists[hist_name].to_boost_histogram()
    bh_det_level = bh_hist[:: bh.sum, ::]
    h_det_level_normalization = np.sum(bh_det_level.view().value)
    h_det_level = binned_data.BinnedData.from_existing_data(bh_det_level[bh.loc(0) :: bh.rebin(2)])  # noqa: E203
    bh_hybrid = bh_hist[::, :: bh.sum]
    h_hybrid_normalization = np.sum(bh_hybrid.view().value)
    h_hybrid = binned_data.BinnedData.from_existing_data(bh_hybrid[bh.loc(0) :: bh.rebin(2)])  # noqa: E203

    # Normalize
    h_det_level /= h_det_level_normalization
    h_det_level /= h_det_level.axes[0].bin_widths
    h_hybrid /= h_hybrid_normalization
    h_hybrid /= h_hybrid.axes[0].bin_widths

    # Finish setup
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)

    style = grooming_styles[grooming_method]
    for hist, label, marker, fillstyle in [
        (h_hybrid, "Embedded Pythia", "o", "full"),
        (h_det_level, "PYTHIA Det. Level", "s", "none"),
    ]:
        # Setup
        # Plot options
        kwargs = {
            "markerfacecolor": "white" if fillstyle == "none" else style.color,
            "alpha": 1 if fillstyle == "none" else 0.8,
        }
        if fillstyle != "none":
            kwargs["markeredgewidth"] = 0

        ax.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            yerr=hist.errors,
            xerr=hist.axes[0].bin_widths / 2,
            color=style.color,
            marker=style.marker,
            fillstyle=fillstyle,
            linestyle="",
            label=label,
            zorder=style.zorder,
            **kwargs,
        )

    # Ratio
    ratio = h_hybrid / h_det_level
    ax_ratio.errorbar(
        ratio.axes[0].bin_centers,
        ratio.values,
        yerr=ratio.errors,
        xerr=ratio.axes[0].bin_widths / 2,
        color=style.color,
        marker=style.marker,
        fillstyle=fillstyle,
        linestyle="",
        zorder=style.zorder,
        **kwargs,
    )

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])

    # Store and cleanup
    matching_type = "matching_type_"
    if matching_suffix:
        matching_type += matching_suffix
    else:
        matching_type += "all"
    filename = f"{plot_config.name}_iterative_splittings_{grooming_method}_{matching_type}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot(grooming_method: str, output_dir: Path) -> None:
    f = uproot.open(f"embeddingResponse_kt_grooming_method_{grooming_method}.root")
    temp_hists = {k.decode("utf-8"): binned_data.BinnedData.from_existing_data(f[k]) for k in f.keys()}
    hists = {}
    # Remove the cycle, which we don't care about.
    for k, v in temp_hists.items():
        k = k[: k.find(";")]
        hists[k] = v
    print(hists)

    # Just for conveneice to get the grooming label right...
    grooming_styles = pb.define_grooming_styles()
    grooming_method_label = grooming_styles[grooming_method].label

    hybrid_jet_pt_bin = helpers.RangeSelector(min=40, max=120)
    response_type = "hybrid_det_level"
    # Improve the display of labels (such as "det_level" -> "det"
    # measured_like_label = response_type.measured_like.replace("_level", "")
    # generator_like_label = response_type.generator_like.replace("_level", "")
    measured_like_label = "hybrid"
    generator_like_label = "det"
    for matching_suffix in ["", "PureMatches"]:
        text = pb.label_to_display_string["ALICE"]["simulation"]
        text += "\n" + f"{grooming_method_label} iterative splittings"
        text += "\n" + ("Pure" if matching_suffix else "All") + " subjet matches"
        text += "\n" + f"${hybrid_jet_pt_bin.display_str(label='hybrid')}$"
        text += "\n" + pb.label_to_display_string["collision_system"]["embedPythia"].format(main_system=r"30-50\%")
        plot_response(
            hists=hists,
            grooming_method=grooming_method,
            matching_suffix=matching_suffix,
            hybrid_jet_pt_bin=hybrid_jet_pt_bin,
            plot_config=pb.PlotConfig(
                name=f"response_kt_{response_type}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x", label=fr"$k_{{\text{{T}}}}^{{\text{{{measured_like_label}}}}}\:(\text{{GeV}}/c)$"
                        ),
                        pb.AxisConfig(
                            "y", label=fr"$k_{{\text{{T}}}}^{{\text{{{generator_like_label}}}}}\:(\text{{GeV}}/c)$"
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.03, text=text, color="w"),
                ),
                figure=pb.Figure(edge_padding=dict(left=0.10, bottom=0.12, right=1.01)),
            ),
            output_dir=output_dir,
        )

    matching_suffix = ""
    text = pb.label_to_display_string["ALICE"]["simulation"]
    text += "\n" + pb.label_to_display_string["collision_system"]["embedPythia"].format(main_system=r"30-50\%")
    text += "\n" + f"{grooming_method_label} iterative splittings"
    text += "\n" + ("Pure" if matching_suffix else "All") + " subjet matches"
    text += "\n" + f"${hybrid_jet_pt_bin.display_str(label='hybrid')}$"
    plot_1D_comparison(
        hists=hists,
        grooming_method=grooming_method,
        matching_suffix=matching_suffix,
        hybrid_jet_pt_bin=hybrid_jet_pt_bin,
        plot_config=pb.PlotConfig(
            name=f"kt_spectra_{response_type}",
            panels=[
                # Main axis.
                pb.Panel(
                    axes=pb.AxisConfig(
                        "y",
                        label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                        log=True,
                        range=(10e-8, 5),
                    ),
                    text=pb.TextConfig(x=0.96, y=0.96, text=text),
                    legend=pb.LegendConfig(location="lower left"),
                ),
                # Ratio.
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig("y", label="Embed./Det."),
                    ]
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.12}),
        ),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    helpers.setup_logging()

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot(grooming_method="leading_kt", output_dir=output_dir)
    plot(grooming_method="leading_kt_z_cut_02", output_dir=output_dir)
    plot(grooming_method="dynamical_kt", output_dir=output_dir)
    plot(grooming_method="dynamical_time", output_dir=output_dir)
