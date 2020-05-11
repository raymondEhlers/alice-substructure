#!/usr/bin/env python3

""" Plotting for the tree skim.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import attr
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
from pachyderm import binned_data

from jet_substructure.analysis.plot_base import (
    AxisConfig,
    Figure,
    LegendConfig,
    Panel,
    PlotConfig,
    TextConfig,
    define_grooming_styles,
)
from jet_substructure.base import analysis_objects, helpers, skim_analysis_objects


logger = logging.getLogger(__name__)

pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True


@attr.s
class PlotHists:
    hists: Mapping[str, bh.Histogram] = attr.ib()
    prefix: str = attr.ib()
    identifier: str = attr.ib()
    display_label: str = attr.ib()


def _project_matching(bh_hist: bh.Histogram, axis_to_keep: int) -> binned_data.BinnedData:
    # Axes: 0 = measured_pt, 1 = measured_kt, 2 = detector_pt , 3 = detector_kt
    selections = [
        slice(None, None, bh.sum),
        slice(None, None, bh.sum),
        slice(None, None, bh.sum),
        slice(None, None, bh.sum),
    ]
    selections[axis_to_keep] = slice(None)

    bh_hist = bh_hist[tuple(selections)]

    return binned_data.BinnedData.from_existing_data(bh_hist)


def _plot_subjet_matching(
    hists: Mapping[str, bh.Histogram],
    axis_parameter: str,
    grooming_method: str,
    matching_types: Sequence[str],
    hybrid_jet_pt_bin: helpers.RangeSelector,
    plot_config: PlotConfig,
    output_dir: Path,
    min_hybrid_kt: float = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    axis_to_keep_map = {"pt": 2, "kt": 3}
    axis_to_keep = axis_to_keep_map[axis_parameter]
    matching_type_label_map = {
        "pure": "Pure matches",
        "leading_untagged_subleading_correct": "Leading unmatched, subleading matched",
        "leading_correct_subleading_untagged": "Leading matched, subleading unmatched",
        "leading_untagged_subleading_mistag": "Leading unmatched, subleading in leading",
        "leading_mistag_subleading_untagged": "Leading in subleading, subleading unmatched",
        "swap": "Swaps",
        "both_untagged": "Leading, subleading unmatched",
    }

    normalization = _project_matching(
        bh_hist=hists[f"{grooming_method}_hybrid_det_level_kt_response_matching_type_all"], axis_to_keep=axis_to_keep
    )

    for matching_type in matching_types:
        if matching_type == "all":
            continue
        logger.debug(
            f"Plotting {axis_parameter} residual for {grooming_method}, {matching_type}, min_hybrid_kt: {min_hybrid_kt}"
        )
        h = _project_matching(
            hists[f"{grooming_method}_hybrid_det_level_kt_response_matching_type_{matching_type}"],
            axis_to_keep=axis_to_keep,
        )

        h /= normalization
        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=matching_type_label_map[matching_type],
        )

    # Labeling
    text = "Iterative splittings"
    text += "\n" + f"${helpers.RangeSelector(40, 120).display_str(label='hybrid')}$"
    text += "\n" + " ".join(grooming_method.split("_")).capitalize()
    ax.text(
        0.975,
        0.55,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="center",
        multialignment="right",
    )

    # Presentation
    # Axis labels
    x_axis_label = fr"${axis_parameter[0]}" + r"_{\text{T}}^{\text{det}}\:(\text{GeV}/c)$"
    ax.set_xlabel(x_axis_label)
    # Apply the PlotConfig
    plot_config.apply(fig=fig, axes=[ax])

    # Store and reset
    fig.savefig(
        output_dir
        / f"{plot_config.name}_{axis_parameter}_hybrid_{hybrid_jet_pt_bin}_{grooming_method}_single_figure.pdf"
    )
    plt.close(fig)


def plot_prong_matching(
    hists: Mapping[str, bh.Histogram], grooming_methods: Sequence[str], matching_types: Sequence[str], output_dir: Path
) -> None:
    hybrid_jet_pt_bin = helpers.RangeSelector(min=40, max=120)
    for grooming_method in grooming_methods:
        _plot_subjet_matching(
            hists=hists,
            grooming_method=grooming_method,
            matching_types=matching_types,
            axis_parameter="pt",
            hybrid_jet_pt_bin=hybrid_jet_pt_bin,
            plot_config=PlotConfig(
                name="subjet_matching",
                panels=Panel(
                    axes=[AxisConfig("y", label="Tagging Fraction", log=True, range=(1e-3, 10))],
                    legend=LegendConfig(location="upper left", ncol=2, font_size=14),
                ),
                figure=Figure(edge_padding=dict(right=0.99, top=0.96)),
            ),
            output_dir=output_dir,
        )


def _plot_residual_by_matching_type(
    hists: Mapping[str, bh.Histogram],
    label: str,
    grooming_method: str,
    matching_types: Sequence[str],
    plot_config: PlotConfig,
    output_dir: Path,
    min_hybrid_kt: float = 0,
) -> None:
    """

    Note:
        The min_hybrid_kt is only meaningful for the kt residual because it has the kt axis...

    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig_simplified, ax_simplified = plt.subplots(figsize=(8, 6))

    for matching_type in matching_types:
        logger.debug(
            f"Plotting {label} residual for {grooming_method}, {matching_type}, min_hybrid_kt: {min_hybrid_kt}"
        )

        matches_label = " ".join(matching_type.split("_")).capitalize()
        bh_hist = hists[f"{grooming_method}_hybrid_det_level_{label}_residuals_matching_type_{matching_type}"]
        h = binned_data.BinnedData.from_existing_data(bh_hist)

        selection_list = [slice(None), slice(None), slice(None)]
        if min_hybrid_kt:
            # Apply a hybrid kt cut.
            selection_list[0] = slice(h.axes[0].find_bin(min_hybrid_kt), None,)
        # Must be a tuple to be used for indexing, but need a list for reassignment.
        selection = tuple(selection_list)

        # Axes: hybrid, det, residual
        # For example, for jet pt, it's: Axes: hybrid_jet_pt, det_level_jet_pt, residual
        h_residual = binned_data.BinnedData(
            axes=[h.axes[2]],
            values=np.sum(h.values[selection], axis=(0, 1)),
            variances=np.sum(h.variances[selection], axis=(0, 1)),
        )

        # Normalize
        h_residual /= np.sum(h_residual.values)

        if matching_type in ["all", "pure", "swap", "leading_correct_subleading_untagged"]:
            ax_simplified.errorbar(
                h_residual.axes[0].bin_centers,
                h_residual.values,
                yerr=h_residual.errors,
                xerr=h_residual.axes[0].bin_widths / 2,
                marker=".",
                linestyle="",
                label=matches_label,
            )
        else:
            # Let's rebin otherwise to reduce error bar size for some other the other methods.
            h_residual = binned_data.BinnedData.from_existing_data(h_residual.to_boost_histogram()[:: bh.rebin(4)])
            # Normalize again
            h_residual /= np.sum(h_residual.values)

        ax.errorbar(
            h_residual.axes[0].bin_centers,
            h_residual.values,
            yerr=h_residual.errors,
            xerr=h_residual.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=matches_label,
        )

    # Labeling
    for a, f in [(ax, fig), (ax_simplified, fig_simplified)]:
        # Presentation
        if min_hybrid_kt:
            # Help out mypy...
            assert plot_config.panels[0].text is not None
            plot_config.panels[0].text.text += (
                "\n" + fr"$k_{{\text{{T}}}}^{{\text{{hybrid}}}} > {min_hybrid_kt}\:\text{{GeV}}/c$"
            )
        plot_config.apply(fig=f, axes=[a])
    # Set range so that it's consistent.
    ax_simplified.set_ylim([-0.01, 0.18])

    # Store and cleanup
    filename = f"{plot_config.name}_iterative_splittings_{grooming_method}"
    if min_hybrid_kt:
        filename += f"_min_hybrid_kt_{min_hybrid_kt}"
    fig.savefig(output_dir / f"{filename}_matching.pdf")
    plt.close(fig)
    fig_simplified.savefig(output_dir / f"{filename}_matching_simplified.pdf")
    plt.close(fig_simplified)


def plot_residuals_by_matching_type(
    hists: Mapping[str, bh.Histogram], grooming_methods: Sequence[str], matching_types: Sequence[str], output_dir: Path
) -> None:
    for grooming_method in grooming_methods:
        # Define shared text.
        text = "Iterative splittings"
        text += "\n" + f"${helpers.RangeSelector(40, 120).display_str(label='hybrid')}$"
        text += "\n" + " ".join(grooming_method.split("_")).capitalize()

        _plot_residual_by_matching_type(
            hists=hists,
            label="jet_pt",
            grooming_method=grooming_method,
            matching_types=matching_types,
            plot_config=PlotConfig(
                name="jet_pt_residual_hybrid_det_level",
                panels=Panel(
                    axes=[
                        AxisConfig(
                            "x",
                            label=r"$(p_{\text{T}}^{\text{hybrid}} - p_{\text{T}}^{\text{det}}) / p_{\text{T}}^{\text{det}}$",
                        ),
                        AxisConfig("y", label=""),
                    ],
                    text=TextConfig(x=0.97, y=0.97, text=text),
                    legend=LegendConfig(location="upper left", font_size=14),
                ),
                figure=Figure(edge_padding=dict(bottom=0.12)),
            ),
            output_dir=output_dir,
        )
        _plot_residual_by_matching_type(
            hists=hists,
            label="kt",
            grooming_method=grooming_method,
            matching_types=matching_types,
            plot_config=PlotConfig(
                name="kt_residual_hybrid_det_level",
                panels=Panel(
                    axes=[
                        AxisConfig(
                            "x",
                            label=r"$(k_{\text{T}}^{\text{hybrid}} - k_{\text{T}}^{\text{det}}) / k_{\text{T}}^{\text{det}}$",
                        ),
                        AxisConfig("y", label=""),
                    ],
                    text=TextConfig(x=0.97, y=0.97, text=text),
                    legend=LegendConfig(location="upper left", font_size=14),
                ),
                figure=Figure(edge_padding=dict(bottom=0.12)),
            ),
            output_dir=output_dir,
        )
        _plot_residual_by_matching_type(
            hists=hists,
            label="kt",
            grooming_method=grooming_method,
            matching_types=matching_types,
            plot_config=PlotConfig(
                name="kt_residual_hybrid_det_level",
                panels=Panel(
                    axes=[
                        AxisConfig(
                            "x",
                            label=r"$(k_{\text{T}}^{\text{hybrid}} - k_{\text{T}}^{\text{det}}) / k_{\text{T}}^{\text{det}}$",
                        ),
                        AxisConfig("y", label=""),
                    ],
                    text=TextConfig(x=0.97, y=0.97, text=text),
                    legend=LegendConfig(location="upper left", font_size=14),
                ),
                figure=Figure(edge_padding=dict(bottom=0.12)),
            ),
            output_dir=output_dir,
            min_hybrid_kt=5,
        )


def _plot_residual_mean_and_width(
    hists: Mapping[str, bh.Histogram],
    grooming_method: str,
    hybrid_jet_pt_bin: helpers.RangeSelector,
    plot_config_mean: PlotConfig,
    plot_config_width: PlotConfig,
    output_dir: Path,
) -> None:
    logger.debug(
        f"Plotting jet pt residual mean and width for {grooming_method} with hybrid jet pt: {hybrid_jet_pt_bin}"
    )

    fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
    fig_width, ax_width = plt.subplots(figsize=(8, 6))

    for jet_types, label, color in [
        ("hybrid_det", "Fluctuations", "red"),
        ("hybrid_true", "Combined", "black"),
        ("det_true", "Detector", "blue"),
    ]:
        bh_hist = hists[f"{grooming_method}_{jet_types}_jet_pt_residual_mean_hybrid_{str(hybrid_jet_pt_bin)}"]
        # Select in hybrid jet pt during conversion.
        # NOTE: We need to use bh to do the sum and projection because it's a profile hist, which requires extra care.
        h = binned_data.BinnedData.from_existing_data(bh_hist)

        # Plot.
        # The values are scaled by the bin centers as a proxy for the true jet pt. Since it's a steeply falling spectra,
        # the bin centers are a bit too large, but it's close enough.
        # Mean is just the values
        ax_mean.errorbar(
            h.axes[0].bin_centers,
            h.values / h.axes[0].bin_centers,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=label,
            alpha=0.8,
            color=color,
        )
        # Width is the errors.
        ax_width.errorbar(
            h.axes[0].bin_centers,
            h.errors / h.axes[0].bin_centers,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=label,
            alpha=0.8,
            color=color,
        )

    # Labeling and presentation
    plot_config_mean.apply(fig=fig_mean, ax=ax_mean)
    plot_config_width.apply(fig=fig_width, ax=ax_width)

    # Store and cleanup
    filename_mean = f"{plot_config_mean.name}_hybrid_{str(hybrid_jet_pt_bin)}_iterative_splittings_{grooming_method}"
    fig_mean.savefig(output_dir / f"{filename_mean}_mean.pdf")
    plt.close(fig_mean)
    filename_width = f"{plot_config_width.name}_hybrid_{str(hybrid_jet_pt_bin)}_iterative_splittings_{grooming_method}"
    fig_width.savefig(output_dir / f"{filename_width}_width.pdf")
    plt.close(fig_width)


def _plot_jet_pt_residual_distribution(
    hists: Mapping[str, bh.Histogram],
    grooming_method: str,
    true_jet_pt_bin: helpers.RangeSelector,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    """ Plot the full jet pt residual for a pt true selection.

    Note:
        The pt true selections was applied when filling. This just plots the values.
    """
    logger.debug(f"Plotting jet pt residual distribution for {grooming_method} with true jet pt {true_jet_pt_bin}")

    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colors are to match the jet substructure semi-central AN
    for jet_types, label, color in [
        ("hybrid_det", "Flucutations", "red"),
        ("hybrid_true", "Combined", "black"),
        ("det_true", "Detector", "blue"),
    ]:
        bh_hist = hists[f"{grooming_method}_{jet_types}_jet_pt_residual_distribution"]
        # NOTE: We need to use bh to do the sum and projection because it's a profile hist, which requires extra care.
        selection = slice(bh.loc(true_jet_pt_bin.min + 0.0001), bh.loc(true_jet_pt_bin.max), bh.sum)
        h = binned_data.BinnedData.from_existing_data(bh_hist[selection, :])

        # Normalize
        h /= np.sum(h.values)

        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=label,
            color=color,
        )

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)

    # Store and cleanup
    filename = f"{plot_config.name}_true_{str(true_jet_pt_bin)}_iterative_splittings_{grooming_method}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_residuals(hists: Mapping[str, bh.Histogram], grooming_methods: Sequence[str], output_dir: Path) -> None:
    hybrid_jet_pt_bins = [helpers.RangeSelector(40, 120), helpers.RangeSelector(20, 200)]
    true_jet_pt_bin = helpers.RangeSelector(40, 60)
    for grooming_method in grooming_methods:
        for hybrid_jet_pt_bin in hybrid_jet_pt_bins:
            text = "Iterative splittings"
            text += "\n" + f"${hybrid_jet_pt_bin.display_str(label='hybrid')}$"
            text += "\n" + " ".join(grooming_method.split("_")).capitalize()
            _plot_residual_mean_and_width(
                hists=hists,
                grooming_method=grooming_method,
                hybrid_jet_pt_bin=hybrid_jet_pt_bin,
                plot_config_mean=PlotConfig(
                    name="jet_pt_residual_mean",
                    panels=Panel(
                        axes=[
                            AxisConfig("x", label=r"$p_{\text{T}}^{\text{part}}\:(\text{GeV}/c)$"),
                            AxisConfig(
                                "y",
                                label=r"$(p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{part}}) / p_{\text{T}}^{\text{part}}$",
                                range=(-1, 2),
                            ),
                        ],
                        text=TextConfig(x=0.97, y=0.97, text=text),
                        legend=LegendConfig(location="upper center", font_size=14),
                    ),
                    figure=Figure(edge_padding=dict(left=0.13, bottom=0.12)),
                ),
                plot_config_width=PlotConfig(
                    name="jet_pt_residual_width",
                    panels=Panel(
                        axes=[
                            AxisConfig("x", label=r"$p_{\text{T}}^{\text{part}}\:(\text{GeV}/c)$"),
                            AxisConfig(
                                "y",
                                label=r"$\sigma(p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{part}}) / p_{\text{T}}^{\text{part}}$",
                                range=(0, 0.5),
                            ),
                        ],
                        text=TextConfig(x=0.97, y=0.97, text=text),
                        legend=LegendConfig(location="upper center", font_size=14),
                    ),
                    figure=Figure(edge_padding=dict(left=0.10, bottom=0.12)),
                ),
                output_dir=output_dir,
            )

        text = "Iterative splittings"
        text += "\n" + f"${true_jet_pt_bin.display_str(label='part')}$"
        text += "\n" + " ".join(grooming_method.split("_")).capitalize()
        _plot_jet_pt_residual_distribution(
            hists=hists,
            grooming_method=grooming_method,
            true_jet_pt_bin=true_jet_pt_bin,
            plot_config=PlotConfig(
                name="jet_pt_residual_distribution",
                panels=Panel(
                    axes=[
                        AxisConfig(
                            "x",
                            label=r"$(p_{\text{T}}^{\text{hybrid}} - p_{\text{T}}^{\text{det}}) / p_{\text{T}}^{\text{det}}$",
                        ),
                        AxisConfig("y", label="", range=(-0.02, 0.32)),
                    ],
                    text=TextConfig(x=0.97, y=0.97, text=text),
                    legend=LegendConfig(location="upper left", font_size=14),
                ),
                figure=Figure(edge_padding=dict(left=0.10, bottom=0.12)),
            ),
            output_dir=output_dir,
        )


def _plot_response_by_matching_type(
    hists: Mapping[str, bh.Histogram],
    label: str,
    grooming_method: str,
    response_type: skim_analysis_objects.ResponseType,
    matching_types: Sequence[str],
    hybrid_jet_pt_bin: helpers.RangeSelector,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    for matching_type in matching_types:
        logger.debug(
            f"Plotting {label} {response_type} response for {grooming_method}, {matching_type}, hybrid: {hybrid_jet_pt_bin}"
        )

        matches_label = " ".join(matching_type.split("_")).capitalize()
        bh_input_hist = hists[f"{grooming_method}_{response_type}_{label}_response_matching_type_{matching_type}"]
        h_input = binned_data.BinnedData.from_existing_data(bh_input_hist)

        # Select the variables (for the example of kt)
        # Axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt
        # NOTE: We already applied the 40 < hybrid jet pt < 120 cut, so it doesn't need an additional selection.
        h = binned_data.BinnedData(
            axes=[h_input.axes[1], h_input.axes[3]],
            values=np.sum(h_input.values, axis=(0, 2)),
            variances=np.sum(h_input.variances, axis=(0, 2)),
        )

        # Normalize the response.
        normalization_values = h.values.sum(axis=0, keepdims=True)
        h.values = np.divide(
            h.values, normalization_values, out=np.zeros_like(h.values), where=normalization_values != 0
        )

        # Finish setup
        fig, ax = plt.subplots(figsize=(8, 6))

        # Determine the normalization range
        z_axis_range = {
            # "vmin": h_proj.values[h_proj.values > 0].min(),
            "vmin": 1e-4,
            "vmax": h.values.max(),
        }

        # Plot
        mesh = ax.pcolormesh(
            h.axes[0].bin_edges.T, h.axes[1].bin_edges.T, h.values.T, norm=matplotlib.colors.LogNorm(**z_axis_range),
        )
        fig.colorbar(mesh, pad=0.02)

        # Labeling and presentation
        # Help out mypy...
        assert plot_config.panels[0].text is not None
        plot_config.panels[0].text.text += "\n" + matches_label + " matches"
        plot_config.apply(fig=fig, ax=ax)

        # Store and cleanup
        filename = f"{plot_config.name}_iterative_splittings_{grooming_method}_matching_type_{matching_type}"
        fig.savefig(output_dir / f"{filename}.pdf")
        plt.close(fig)


def plot_response_by_matching_type(
    hists: Mapping[str, bh.Histogram],
    grooming_methods: Sequence[str],
    response_types: Sequence[skim_analysis_objects.ResponseType],
    matching_types: Sequence[str],
    output_dir: Path,
) -> None:
    hybrid_jet_pt_bin = helpers.RangeSelector(min=40, max=120)
    for grooming_method in grooming_methods:
        for response_type in response_types:
            # Improve the display of labels (such as "det_level" -> "det"
            measured_like_label = response_type.measured_like.replace("_level", "")
            generator_like_label = response_type.generator_like.replace("_level", "")
            text = "Iterative splittings"
            text += "\n" + f"${hybrid_jet_pt_bin.display_str(label='hybrid')}$"
            text += "\n" + " ".join(grooming_method.split("_")).capitalize()
            _plot_response_by_matching_type(
                hists=hists,
                label="kt",
                grooming_method=grooming_method,
                response_type=response_type,
                matching_types=matching_types,
                hybrid_jet_pt_bin=hybrid_jet_pt_bin,
                plot_config=PlotConfig(
                    name=f"response_kt_{response_type}",
                    panels=Panel(
                        axes=[
                            AxisConfig(
                                "x", label=fr"$k_{{\text{{T}}}}^{{\text{{{measured_like_label}}}}}\:(\text{{GeV}}/c)$"
                            ),
                            AxisConfig(
                                "y", label=fr"$k_{{\text{{T}}}}^{{\text{{{generator_like_label}}}}}\:(\text{{GeV}}/c)$"
                            ),
                        ],
                        text=TextConfig(x=0.03, y=0.97, text=text),
                        legend=LegendConfig(location="upper left", font_size=14),
                    ),
                    figure=Figure(edge_padding=dict(left=0.10, bottom=0.12)),
                ),
                output_dir=output_dir,
            )
            _plot_response_by_matching_type(
                hists=hists,
                label="delta_R",
                grooming_method=grooming_method,
                response_type=response_type,
                matching_types=matching_types,
                hybrid_jet_pt_bin=hybrid_jet_pt_bin,
                plot_config=PlotConfig(
                    name=f"response_delta_R_{response_type}",
                    panels=Panel(
                        axes=[
                            AxisConfig("x", label=fr"$R^{{\text{{ {measured_like_label} }}}}$"),
                            AxisConfig("y", label=fr"$R^{{\text{{ {generator_like_label} }}}}$"),
                        ],
                        text=TextConfig(x=0.03, y=0.97, text=text),
                        legend=LegendConfig(location="upper left", font_size=14),
                    ),
                    figure=Figure(edge_padding=dict(left=0.10, bottom=0.12)),
                ),
                output_dir=output_dir,
            )


def _plot_kt_comparison(
    hists: Mapping[str, bh.Histogram],
    # data_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    data_hist: bh.Histogram,
    grooming_method: str,
    hybrid_jet_pt_bin: helpers.RangeSelector,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    logger.info(f"Plotting kt comparison for {grooming_method} with hybrid {hybrid_jet_pt_bin}")
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)

    # Data
    rebin_factor = 2
    # bh_data = getattr(data_hists, grooming_method).kt.to_boost_histogram()
    # bh_data = data_hist
    # bh_data = bh_data[:: bh.rebin(rebin_factor)]
    # NOTE: We already applied the 40 < hybrid jet pt < 120 cut, so it doesn't need an additional selection.
    bh_data = data_hist
    bh_data = bh_data[:: bh.sum, :]
    h_data = binned_data.BinnedData.from_existing_data(bh_data)
    h_data /= rebin_factor

    # Embedded
    h_embed_response = binned_data.BinnedData.from_existing_data(
        hists[f"{grooming_method}_hybrid_true_kt_response_matching_type_all"]
    )
    # Select the variables (for the example of kt)
    # Axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt
    # NOTE: We already applied the 40 < hybrid jet pt < 120 cut, so it doesn't need an additional selection.
    h_hybrid = binned_data.BinnedData(
        axes=[h_embed_response.axes[1]],
        values=np.sum(h_embed_response.values, axis=(0, 2, 3)),
        variances=np.sum(h_embed_response.variances, axis=(0, 2, 3)),
    )
    h_det = binned_data.BinnedData(
        axes=[h_embed_response.axes[3]],
        values=np.sum(h_embed_response.values, axis=(0, 1, 2)),
        variances=np.sum(h_embed_response.variances, axis=(0, 1, 2)),
    )
    # Pure response spectra
    h_embed_response_pure = binned_data.BinnedData.from_existing_data(
        hists[f"{grooming_method}_hybrid_true_kt_response_matching_type_pure"]
    )
    # Select the variables (for the example of kt)
    # Axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt
    # NOTE: We already applied the 40 < hybrid jet pt < 120 cut, so it doesn't need an additional selection.
    h_hybrid_pure = binned_data.BinnedData(
        axes=[h_embed_response_pure.axes[1]],
        values=np.sum(h_embed_response_pure.values, axis=(0, 2, 3)),
        variances=np.sum(h_embed_response_pure.variances, axis=(0, 2, 3)),
    )
    h_det_pure = binned_data.BinnedData(
        axes=[h_embed_response_pure.axes[3]],
        values=np.sum(h_embed_response_pure.values, axis=(0, 1, 2)),
        variances=np.sum(h_embed_response_pure.variances, axis=(0, 1, 2)),
    )

    # Normalize by n_jets
    # TODO: Update the data approach once we have the skim!
    # h_data /= getattr(data_hists, grooming_method).n_jets
    h_data /= np.sum(h_data.values)
    h_hybrid /= np.sum(h_hybrid.values)
    h_det /= np.sum(h_det.values)
    h_hybrid_pure /= np.sum(h_hybrid_pure.values)
    h_det_pure /= np.sum(h_det_pure.values)

    for h, label in [
        (h_data, "Pb--Pb"),
        (h_hybrid, "Hybrid"),
        (h_det, "Det. level"),
        (h_hybrid_pure, "Hybrid pure matches"),
        (h_det_pure, "Det. level pure matches"),
    ]:
        p = ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=label,
        )

        if label != "Pb--Pb":
            # Temp exclude normalization bin of h
            # TODO: Fix once data is skimmed.
            # h_temp = binned_data.BinnedData(
            #    axes=[h.axes[0].bin_edges[1:]], values=h.values[1:], variances=h.variances[1:],
            # )
            # import IPython; IPython.embed()
            h_ratio = h_data / h
            # If we get 0, we don't want to show that point.
            h_ratio.values[h_ratio.values == 0] = np.nan

            # Plot the ratio
            ax_ratio.errorbar(
                h_ratio.axes[0].bin_centers,
                h_ratio.values,
                yerr=h_ratio.errors,
                xerr=h_ratio.axes[0].bin_widths / 2,
                marker=".",
                linestyle="",
                color=p[0].get_color(),
            )

    # Reference value
    ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])

    # Store and cleanup
    filename = f"{plot_config.name}_hybrid_{hybrid_jet_pt_bin}_iterative_splittings_{grooming_method}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_compare_kt(
    hists: Mapping[str, bh.Histogram],
    data_hists: Mapping[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]],
    grooming_methods: Sequence[str],
    output_dir: Path,
) -> None:
    hybrid_jet_pt_bin = helpers.RangeSelector(min=40, max=120)
    identifier = analysis_objects.Identifier(iterative_splittings=True, jet_pt_bin=hybrid_jet_pt_bin)
    data_hists_for_comparion = data_hists[identifier]

    for grooming_method in grooming_methods:
        if grooming_method in [
            "leading_kt_z_cut_02",
            "leading_kt_z_cut_04",
            "soft_drop_z_cut_02",
            "soft_drop_z_cut_04",
        ]:
            logger.debug(f"Skipping grooming method {grooming_method} because we don't have the data comparison yet.")
            continue

        text = "Iterative splittings"
        text += "\n" + f"${hybrid_jet_pt_bin.display_str(label='hybrid')}$"
        text += "\n" + " ".join(grooming_method.split("_")).capitalize()
        _plot_kt_comparison(
            hists=hists,
            # TODO: This won't work quite right! It needs a rebin + not to be projected in the comparison function.
            data_hist=getattr(data_hists_for_comparion, grooming_method).kt,
            grooming_method=grooming_method,
            hybrid_jet_pt_bin=hybrid_jet_pt_bin,
            plot_config=PlotConfig(
                name="kt_spectra",
                panels=[
                    # Main panel
                    Panel(
                        axes=[
                            AxisConfig(
                                "y",
                                label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                                log=True,
                            )
                        ],
                        text=TextConfig(x=0.97, y=0.97, text=text),
                        legend=LegendConfig(location="lower left", font_size=14),
                    ),
                    # Ratio
                    Panel(
                        axes=[
                            AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0, 25)),
                            AxisConfig("y", label="Pb--Pb/ref.", range=(0, 5)),
                        ],
                    ),
                ],
                figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
            ),
            output_dir=output_dir,
        )


def plot_compare_kt_skim(
    data_hists: Mapping[str, bh.Histogram],
    embed_hists: Mapping[str, bh.Histogram],
    grooming_methods: Sequence[str],
    output_dir: Path,
) -> None:
    hybrid_jet_pt_bin = helpers.RangeSelector(min=40, max=120)
    prefix = "data"

    for grooming_method in grooming_methods:
        text = "Iterative splittings"
        text += "\n" + f"${hybrid_jet_pt_bin.display_str(label='hybrid')}$"
        text += "\n" + " ".join(grooming_method.split("_")).capitalize()
        _plot_kt_comparison(
            hists=embed_hists,
            data_hist=data_hists[f"{grooming_method}_{prefix}_kt"],
            grooming_method=grooming_method,
            hybrid_jet_pt_bin=hybrid_jet_pt_bin,
            plot_config=PlotConfig(
                name="kt_spectra",
                panels=[
                    # Main panel
                    Panel(
                        axes=[
                            AxisConfig(
                                "y",
                                label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                                log=True,
                            )
                        ],
                        text=TextConfig(x=0.97, y=0.97, text=text),
                        legend=LegendConfig(location="lower left", font_size=14),
                    ),
                    # Ratio
                    Panel(
                        axes=[
                            AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0, 25)),
                            AxisConfig("y", label="Pb--Pb/ref.", range=(0, 5)),
                        ],
                    ),
                ],
                figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
            ),
            output_dir=output_dir,
        )


def _plot_kt_vs_jet_pt_raw_with_labels(
    hists: Mapping[str, bh.Histogram],
    grooming_method: str,
    prefix: str,
    jet_pt_bin: helpers.RangeSelector,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    logger.debug(f"Plotting kt vs jet pt for {grooming_method}.")

    fig, ax = plt.subplots(figsize=(8, 6))

    # We want to plot the 2D hist, so no need for any projections.
    # However, first we need to rebin
    bh_hist = hists[f"{grooming_method}_{prefix}_kt"]
    h = binned_data.BinnedData.from_existing_data(
        bh_hist[bh.loc(40) : bh.loc(120) : bh.rebin(4), 1 :: bh.rebin(2)]  # noqa: E203
    )

    # Plot
    # Normally, we transpose the data. However, we want the kt on the x axis and the pt on the y axis.
    # So we leave it as is. Further, we just want the values in text, not the heatmap. So we fill everything
    # with zeros, and then use a heatmap with white at 0 so it doesn't show up. We'll label the values below.
    ax.pcolormesh(
        h.axes[1].bin_edges.T,
        h.axes[0].bin_edges.T,
        np.zeros_like(h.values),
        cmap="bwr",
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=1),
    )

    # Plot values labels. These will be the only things that show up.
    for i, kt_bin_center in enumerate(h.axes[1].bin_centers):
        for j, pt_bin_center in enumerate(h.axes[0].bin_centers):
            ax.text(
                kt_bin_center,
                pt_bin_center,
                f"{h.values[j, i]:g}",
                ha="center",
                va="center",
                color="black",
                rotation=45,
            )

    # Labeling and presentation
    # Add ticks outwards because otherwise they're covered by the white on the plot.
    ax.tick_params(axis="both", which="both", direction="out")
    plot_config.apply(fig=fig, ax=ax)

    # Store and cleanup
    filename = f"{plot_config.name}_{jet_pt_bin}_iterative_splittings_{grooming_method}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_kt_vs_jet_pt(hists: Mapping[str, bh.Histogram], grooming_methods: Sequence[str], output_dir: Path,) -> None:
    jet_pt_bin = helpers.RangeSelector(min=40, max=120)
    prefix = "data"

    for grooming_method in grooming_methods:
        text = "Iterative splittings"
        text += ", " + " ".join(grooming_method.split("_")).capitalize()
        _plot_kt_vs_jet_pt_raw_with_labels(
            hists=hists,
            grooming_method=grooming_method,
            prefix=prefix,
            jet_pt_bin=jet_pt_bin,
            plot_config=PlotConfig(
                name="kt_vs_jet_pt_raw",
                panels=Panel(
                    axes=[
                        AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0, 25)),
                        AxisConfig("y", label=r"$p_{\text{T}}\:(\text{GeV}/c)$", range=(40, 120)),
                    ],
                    text=TextConfig(x=0.98, y=0.02, text=text),
                ),
                figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
            ),
            output_dir=output_dir,
        )


def _plot_distance_comparison(
    hists: Mapping[str, bh.Histogram],
    grooming_method: str,
    matching_types: Sequence[str],
    hybrid_jet_pt_bin: helpers.RangeSelector,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    logger.debug(f"Plotting hybrid-det distance comparison for {grooming_method}.")

    fig, ax = plt.subplots(figsize=(8, 6))

    # We want to plot the 2D hist, so no need for any projections.
    # However, first we need to rebin
    # for name, label in [
    #    (f"{grooming_method}_hybrid_det_distance", "all"),
    #    (f"{grooming_method}_hybrid_det_distance_pure", "pure"),
    #    (
    #        f"{grooming_method}_hybrid_det_distance_corner",
    #        r"$k_{\text{T}}^{\text{true}} > 10, k_{\text{T}}^{\text{hybrid}} < 10$",
    #    ),
    # ]:
    for matching_type in matching_types:
        matches_label = " ".join(matching_type.split("_")).capitalize()
        h = binned_data.BinnedData.from_existing_data(
            hists[f"{grooming_method}_hybrid_det_level_distance_matching_type_{matching_type}"]
        )

        if matching_type not in ["all", "pure", "swap", "leading_correct_subleading_untagged"]:
            # Let's rebin otherwise to reduce error bar size for some other the other methods.
            h = binned_data.BinnedData.from_existing_data(h.to_boost_histogram()[:: bh.rebin(2)])

        # Normalize
        h /= np.sum(h.values)

        # Plot
        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=matches_label,
        )

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)

    # Store and cleanup
    filename = f"{plot_config.name}_hybrid_det_level_{hybrid_jet_pt_bin}_iterative_splittings_{grooming_method}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def _plot_leading_matched_subleading_unmatched_short_distance_response(
    hists: Mapping[str, bh.Histogram],
    grooming_method: str,
    hybrid_jet_pt_bin: helpers.RangeSelector,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    logger.debug(f"Plotting hybrid-det leading correct, subleading unmatched short distance for {grooming_method}.")

    h_input = binned_data.BinnedData.from_existing_data(
        hists[
            f"{grooming_method}_hybrid_det_level_kt_response_matching_type_leading_correct_subleading_untagged_distance_less_than_005"
        ]
    )
    # Select the variables (for the example of kt)
    # Axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt
    # NOTE: We already applied the 40 < hybrid jet pt < 120 cut, so it doesn't need an additional selection.
    h = binned_data.BinnedData(
        axes=[h_input.axes[1], h_input.axes[3]],
        values=np.sum(h_input.values, axis=(0, 2)),
        variances=np.sum(h_input.variances, axis=(0, 2)),
    )

    # Normalize the response.
    normalization_values = h.values.sum(axis=0, keepdims=True)
    h.values = np.divide(h.values, normalization_values, out=np.zeros_like(h.values), where=normalization_values != 0)

    # Finish setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        # "vmin": h.values[h.values > 0].min(),
        "vmin": 1e-4,
        "vmax": h.values.max(),
    }

    # Plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T, h.axes[1].bin_edges.T, h.values.T, norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)

    # Store and cleanup
    filename = f"{plot_config.name}_hybrid_det_level_{hybrid_jet_pt_bin}_iterative_splittings_{grooming_method}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_distance_comparison(
    hists: Mapping[str, bh.Histogram], grooming_methods: Sequence[str], matching_types: Sequence[str], output_dir: Path
) -> None:
    hybrid_jet_pt_bin = helpers.RangeSelector(min=40, max=120)

    for grooming_method in grooming_methods:
        text = "Iterative splittings"
        text += "\n" + f"${hybrid_jet_pt_bin.display_str(label='hybrid')}$"
        text += "\n" + " ".join(grooming_method.split("_")).capitalize()
        _plot_distance_comparison(
            hists=hists,
            grooming_method=grooming_method,
            matching_types=matching_types,
            hybrid_jet_pt_bin=hybrid_jet_pt_bin,
            plot_config=PlotConfig(
                name="distance_comparison",
                panels=Panel(
                    axes=[
                        AxisConfig("x", label=r"$\Delta R_{\text{hybrid-det}}$", range=(0, 0.4)),
                        AxisConfig("y", label=r"Prob."),
                    ],
                    text=TextConfig(x=0.97, y=0.97, text=text),
                    legend=LegendConfig(location="center right", font_size=12),
                ),
                figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
            ),
            output_dir=output_dir,
        )

        text = "Iterative splittings"
        text += "\n" + f"${hybrid_jet_pt_bin.display_str(label='hybrid')}$"
        text += "\n" + " ".join(grooming_method.split("_")).capitalize()
        text += "\n" + "Leading matched, subleading unmatched"
        text += "\n" + r"$\Delta R < 0.05$"
        _plot_leading_matched_subleading_unmatched_short_distance_response(
            hists=hists,
            grooming_method=grooming_method,
            hybrid_jet_pt_bin=hybrid_jet_pt_bin,
            plot_config=PlotConfig(
                name="leading_matched_subleading_unmatched_short_distance",
                panels=Panel(
                    axes=[
                        AxisConfig("x", label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                        AxisConfig("y", label=r"$k_{\text{T}}^{\text{det}}\:(\text{GeV}/c)$"),
                    ],
                    text=TextConfig(x=0.97, y=0.97, text=text),
                    legend=LegendConfig(location="center right", font_size=12),
                ),
                figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
            ),
            output_dir=output_dir,
        )


def _project_and_prepare_grooming_variable_hist(
    bh_hist: bh.Histogram, jet_pt_bin: helpers.RangeSelector, set_zero_to_nan: bool
) -> binned_data.BinnedData:
    # Need to project to just the attr of interest.
    h = binned_data.BinnedData.from_existing_data(
        bh_hist[bh.loc(jet_pt_bin.min) : bh.loc(jet_pt_bin.max) : bh.sum, :]  # noqa: E203
    )

    # Normalize
    # Normalize by the sum of the values to get the n_jets values.
    # Then, we still need to normalize by the bin widths.
    h /= np.sum(h.values)
    h /= h.axes[0].bin_widths

    # Set 0s to NaN (for example, in z_g where have a good portion of the range cut off).
    if set_zero_to_nan:
        mask = h.values == 0
        h.errors[mask] = np.nan
        h.values[mask] = np.nan

    return h


def _plot_compare_grooming_methods_for_attribute(
    hists: Mapping[str, bh.Histogram],
    grooming_methods: Sequence[str],
    attr_name: str,
    prefix: str,
    jet_pt_bin: helpers.RangeSelector,
    set_zero_to_nan: bool,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    logger.info(f"Plotting grooming method comparison for {attr_name}")

    fig, ax = plt.subplots(figsize=(8, 6))

    grooming_styling = define_grooming_styles()

    for grooming_method in grooming_methods:
        # Setup
        style = grooming_styling[grooming_method]

        # Axes: jet_pt, attr_name
        bh_hist = hists[f"{grooming_method}_{prefix}_{attr_name}"]
        # Need to project to just the attr of interest.
        h = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(jet_pt_bin.min) : bh.loc(jet_pt_bin.max) : bh.sum, :]  # noqa: E203
        )

        # Normalize
        # Normalize by the sum of the values to get the n_jets values.
        # Then, we still need to normalize by the bin widths.
        h /= np.sum(h.values)
        h /= h.axes[0].bin_widths

        # Set 0s to NaN (for example, in z_g where have a good portion of the range cut off).
        if set_zero_to_nan:
            h.errors[h.values == 0] = np.nan
            h.values[h.values == 0] = np.nan

        # Plot options
        kwargs = {
            "markerfacecolor": "white" if style.fillstyle == "none" else style.color,
            "alpha": 1 if style.fillstyle == "none" else 0.8,
        }
        if style.fillstyle != "none":
            kwargs["markeredgewidth"] = 0

        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            color=style.color,
            marker=style.marker,
            fillstyle=style.fillstyle,
            linestyle="",
            label=style.label,
            zorder=style.zorder,
            **kwargs,
        )

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)

    # Store and cleanup
    # It's expected that the attr_name is already included in the `plot_config.name`.
    # Sanity check to make sure we don't get that wrong!
    # if attr_name not in plot_config.name:
    #    raise ValueError(
    #        f"PlotConfig name must contain the attr name! attr_name: {attr_name}, name: {plot_config.name}"
    #    )

    filename = f"{plot_config.name}_{jet_pt_bin}_iterative_splittings"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def compare_grooming_methods_for_substructure_prod(
    hists: Mapping[str, bh.Histogram], grooming_methods: Sequence[str], output_dir: Path
) -> None:
    """

    """
    jet_pt_bin = helpers.RangeSelector(min=40, max=120)

    # TODO: Comprehensive ALICE labeling.
    text = "Iterative splittings"
    text += "\n" + f"${jet_pt_bin.display_str(label='')}$"
    # text += "\n" + " ".join(grooming_method.split("_")).capitalize()
    _plot_compare_grooming_methods_for_attribute(
        hists=hists,
        grooming_methods=grooming_methods,
        attr_name="kt",
        prefix="data",
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=PlotConfig(
            name="kt_grooming_methods",
            panels=Panel(
                axes=[
                    AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                    AxisConfig(
                        "y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True
                    ),
                ],
                text=TextConfig(x=0.97, y=0.97, text=text),
                legend=LegendConfig(location="lower left", font_size=14),
            ),
            figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=output_dir,
    )

    _plot_compare_grooming_methods_for_attribute(
        hists=hists,
        grooming_methods=grooming_methods,
        attr_name="delta_R",
        prefix="data",
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=PlotConfig(
            name="delta_R_grooming_methods",
            panels=Panel(
                axes=[
                    AxisConfig("x", label=r"$\Delta R$"),
                    AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\Delta R$", log=False),
                ],
                text=TextConfig(x=0.97, y=0.97, text=text),
                legend=LegendConfig(location="center right", font_size=14),
            ),
            figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=output_dir,
    )

    _plot_compare_grooming_methods_for_attribute(
        hists=hists,
        grooming_methods=grooming_methods,
        attr_name="z",
        prefix="data",
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=True,
        plot_config=PlotConfig(
            name="z_grooming_methods",
            panels=Panel(
                axes=[
                    AxisConfig("x", label=r"$z$"),
                    AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}z$", log=False),
                ],
                text=TextConfig(x=0.97, y=0.97, text=text),
                legend=LegendConfig(location="upper right", font_size=14),
            ),
            figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=output_dir,
    )

    _plot_compare_grooming_methods_for_attribute(
        hists=hists,
        grooming_methods=grooming_methods,
        attr_name="n",
        prefix="data",
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=PlotConfig(
            name="number_to_split_grooming_methods",
            panels=Panel(
                axes=[
                    AxisConfig("x", label=r"$n$"),
                    AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n$", log=False),
                ],
                text=TextConfig(x=0.97, y=0.97, text=text),
                legend=LegendConfig(location="center right", font_size=14),
            ),
            figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=output_dir,
    )

    # High kt comparison
    _plot_compare_grooming_methods_for_attribute(
        hists=hists,
        grooming_methods=grooming_methods,
        attr_name="kt_high_kt",
        prefix="data",
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=PlotConfig(
            name="kt_high_kt_grooming_methods",
            panels=Panel(
                axes=[
                    AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                    AxisConfig(
                        "y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True
                    ),
                ],
                text=TextConfig(x=0.97, y=0.97, text=text),
                legend=LegendConfig(location="lower left", font_size=14),
            ),
            figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=output_dir,
    )

    _plot_compare_grooming_methods_for_attribute(
        hists=hists,
        grooming_methods=grooming_methods,
        attr_name="delta_R_high_kt",
        prefix="data",
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=PlotConfig(
            name="delta_R_high_kt_grooming_methods",
            panels=Panel(
                axes=[
                    AxisConfig("x", label=r"$\Delta R$"),
                    AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\Delta R$", log=False),
                ],
                text=TextConfig(x=0.97, y=0.97, text=text),
                legend=LegendConfig(location="center right", font_size=14),
            ),
            figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=output_dir,
    )

    _plot_compare_grooming_methods_for_attribute(
        hists=hists,
        grooming_methods=grooming_methods,
        attr_name="z_high_kt",
        prefix="data",
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=True,
        plot_config=PlotConfig(
            name="z_high_kt_grooming_methods",
            panels=Panel(
                axes=[
                    AxisConfig("x", label=r"$z$"),
                    AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}z$", log=False),
                ],
                text=TextConfig(x=0.97, y=0.97, text=text),
                legend=LegendConfig(location="upper center", font_size=14),
            ),
            figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=output_dir,
    )

    _plot_compare_grooming_methods_for_attribute(
        hists=hists,
        grooming_methods=grooming_methods,
        attr_name="n_high_kt",
        prefix="data",
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=PlotConfig(
            name="number_to_split_high_kt_grooming_methods",
            panels=Panel(
                axes=[
                    AxisConfig("x", label=r"$n$"),
                    AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n$", log=False),
                ],
                text=TextConfig(x=0.97, y=0.97, text=text),
                legend=LegendConfig(location="upper right", font_size=14),
            ),
            figure=Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=output_dir,
    )


def _plot_compare_grooming_methods_for_attribute_data_embed(
    hists: Sequence[PlotHists],
    attr_name: str,
    grooming_methods: Sequence[str],
    jet_pt_bin: helpers.RangeSelector,
    set_zero_to_nan: bool,
    plot_config: PlotConfig,
    output_dir: Path,
) -> None:
    # Setup
    display_labels_vs = " vs. ".join([obj.display_label for obj in hists])
    logger.info(
        f"Plotting grooming method comparison for {attr_name}, {display_labels_vs}, grooming_methods: {grooming_methods}"
    )
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,)
    grooming_styling = define_grooming_styles()

    for grooming_method in grooming_methods:
        main_hist = None
        for hists_obj in hists:
            # Setup and project
            # Axes: jet_pt, attr_name
            h = _project_and_prepare_grooming_variable_hist(
                bh_hist=hists_obj.hists[f"{grooming_method}_{hists_obj.prefix}_{attr_name}"],
                jet_pt_bin=jet_pt_bin,
                set_zero_to_nan=set_zero_to_nan,
            )

            # Setup
            # First, we determine the style
            style = grooming_styling[grooming_method]
            if main_hist is not None:
                style = grooming_styling[f"{grooming_method}_compare"]
            # And then the label
            label = f"{hists_obj.display_label}, {style.label}"

            # Additional plot styling
            kwargs = {
                "markerfacecolor": "white" if style.fillstyle == "none" else style.color,
                "alpha": 1 if style.fillstyle == "none" else 0.8,
            }
            if style.fillstyle != "none":
                kwargs["markeredgewidth"] = 0

            ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                yerr=h.errors,
                xerr=h.axes[0].bin_widths / 2,
                color=style.color,
                marker=style.marker,
                fillstyle=style.fillstyle,
                linestyle="",
                label=label,
                zorder=style.zorder,
                **kwargs,
            )

            # We've plotted the main obj, so now we store that hist, and we will treat the rest as comparisons.
            # For those comparisons, we want to create ratios.
            if main_hist is None:
                main_hist = h
            else:
                ratio = main_hist / h
                ax_ratio.errorbar(
                    ratio.axes[0].bin_centers,
                    ratio.values,
                    yerr=h.errors,
                    xerr=h.axes[0].bin_widths / 2,
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

    # Store and cleanup
    # It's expected that the attr_name is already included in the `plot_config.name`.
    # Sanity check to make sure we don't get that wrong!
    # if attr_name not in plot_config.name:
    #    raise ValueError(
    #        f"PlotConfig name must contain the attr name! attr_name: {attr_name}, name: {plot_config.name}"
    #    )

    grooming_methods_filename_label = ""
    if len(grooming_methods) == 1:
        grooming_methods_filename_label = f"_{grooming_methods[0]}"
    identifiers = "_".join([obj.identifier for obj in hists])
    filename = f"{plot_config.name}_{jet_pt_bin}{grooming_methods_filename_label}_{identifiers}_iterative_splittings"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def compare_grooming_methods_for_substructure_data_embed_prod(
    data_hists: Mapping[str, bh.Histogram],
    embed_hists: Mapping[str, bh.Histogram],
    grooming_methods: Sequence[str],
    output_dir: Path,
) -> None:
    """

    """
    jet_pt_bin = helpers.RangeSelector(min=40, max=120)
    text = "Iterative splittings"
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    text_high_kt = text + "\n" + r"$k_{\text{T}} > 10\:\text{GeV}/c$"

    hists = [
        PlotHists(hists=data_hists, prefix="data", identifier="PbPb", display_label="Pb--Pb",),
        PlotHists(hists=embed_hists, prefix="hybrid", identifier="hybrid", display_label="Hybrid",),
    ]
    for grooming_method in grooming_methods:
        _plot_compare_grooming_methods_for_attribute_data_embed(
            hists=hists,
            attr_name="kt",
            grooming_methods=[grooming_method],
            jet_pt_bin=jet_pt_bin,
            set_zero_to_nan=False,
            plot_config=PlotConfig(
                name="kt_grooming_methods",
                panels=[
                    # Main axis.
                    Panel(
                        axes=AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                        ),
                        text=TextConfig(x=0.96, y=0.96, text=text),
                        legend=LegendConfig(location="lower left"),
                    ),
                    # Ratio.
                    Panel(
                        axes=[
                            AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                            AxisConfig("y", label="Pb--Pb/Hybrid", range=(-0.2, 10)),
                        ]
                    ),
                ],
                figure=Figure(edge_padding={"left": 0.12}),
            ),
            output_dir=output_dir,
        )
        _plot_compare_grooming_methods_for_attribute_data_embed(
            hists=hists,
            attr_name="delta_R",
            grooming_methods=[grooming_method],
            jet_pt_bin=jet_pt_bin,
            set_zero_to_nan=True,
            plot_config=PlotConfig(
                name="delta_R_grooming_methods",
                panels=[
                    # Main axis.
                    # NOTE: This intentionally cuts off the normalization bin
                    Panel(
                        axes=[
                            AxisConfig("x", range=(0, 0.41)),
                            AxisConfig(
                                "y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\Delta R$", range=(-0.4, 19.1)
                            ),
                        ],
                        text=TextConfig(x=0.04, y=0.96, text=text),
                        legend=LegendConfig(location="upper left", anchor=(0.02, 0.79)),
                    ),
                    # Ratio.
                    Panel(axes=[AxisConfig("x", label=r"$\Delta R$"), AxisConfig("y", label="Pb--Pb/Hybrid")]),
                ],
            ),
            output_dir=output_dir,
        )

        _plot_compare_grooming_methods_for_attribute_data_embed(
            hists=hists,
            attr_name="z",
            grooming_methods=[grooming_method],
            jet_pt_bin=jet_pt_bin,
            set_zero_to_nan=True,
            plot_config=PlotConfig(
                name="z_grooming_methods",
                panels=[
                    # Main axis.
                    Panel(
                        axes=[AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}z$", range=(-0.1, 9.9))],
                        text=TextConfig(x=0.04, y=0.96, text=text),
                        legend=LegendConfig(location="upper left", anchor=(0.02, 0.79)),
                    ),
                    # Ratio.
                    Panel(
                        axes=[AxisConfig("x", label=r"$z$", range=(0, 0.51)), AxisConfig("y", label="Pb--Pb/Hybrid")]
                    ),
                ],
            ),
            output_dir=output_dir,
        )

        _plot_compare_grooming_methods_for_attribute_data_embed(
            hists=hists,
            attr_name="n",
            grooming_methods=[grooming_method],
            jet_pt_bin=jet_pt_bin,
            set_zero_to_nan=False,
            plot_config=PlotConfig(
                name="number_to_split_grooming_methods",
                panels=[
                    # Main axis.
                    Panel(
                        axes=[AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n$")],
                        text=TextConfig(x=0.96, y=0.96, text=text),
                        legend=LegendConfig(location="upper right", anchor=(0.96, 0.79)),
                    ),
                    # Ratio.
                    Panel(axes=[AxisConfig("x", label=r"$n$"), AxisConfig("y", label="Pb--Pb/Hybrid")]),
                ],
            ),
            output_dir=output_dir,
        )

        # High kt comparison
        _plot_compare_grooming_methods_for_attribute_data_embed(
            hists=hists,
            attr_name="kt_high_kt",
            grooming_methods=[grooming_method],
            jet_pt_bin=jet_pt_bin,
            set_zero_to_nan=False,
            plot_config=PlotConfig(
                name="kt_high_kt_grooming_methods",
                panels=[
                    # Main axis.
                    Panel(
                        axes=AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                        ),
                        text=TextConfig(x=0.96, y=0.96, text=text_high_kt),
                        legend=LegendConfig(location="lower left"),
                    ),
                    # Ratio.
                    Panel(
                        axes=[
                            AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                            AxisConfig("y", label="Pb--Pb/Hybrid", range=(-0.2, 9.9)),
                        ]
                    ),
                ],
                figure=Figure(edge_padding={"left": 0.12}),
            ),
            output_dir=output_dir,
        )

        _plot_compare_grooming_methods_for_attribute_data_embed(
            hists=hists,
            attr_name="delta_R_high_kt",
            grooming_methods=[grooming_method],
            jet_pt_bin=jet_pt_bin,
            set_zero_to_nan=True,
            plot_config=PlotConfig(
                name="delta_R_high_kt_grooming_methods",
                panels=[
                    # Main axis.
                    Panel(
                        axes=[
                            AxisConfig(
                                "y",
                                label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\Delta R$",
                                range=(1e-3, 25),
                                log=True,
                            ),
                        ],
                        text=TextConfig(x=0.04, y=0.96, text=text_high_kt),
                        legend=LegendConfig(location="upper left", anchor=(0.02, 0.73)),
                    ),
                    # Ratio.
                    Panel(
                        axes=[
                            AxisConfig("x", label=r"$\Delta R$", range=(0, 0.41)),
                            AxisConfig("y", label="Pb--Pb/Hybrid"),
                        ]
                    ),
                ],
                figure=Figure(edge_padding={"left": 0.12}),
            ),
            output_dir=output_dir,
        )

        _plot_compare_grooming_methods_for_attribute_data_embed(
            hists=hists,
            attr_name="z_high_kt",
            grooming_methods=[grooming_method],
            jet_pt_bin=jet_pt_bin,
            set_zero_to_nan=True,
            plot_config=PlotConfig(
                name="z_high_kt_grooming_methods",
                panels=[
                    # Main axis.
                    Panel(
                        axes=[AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}z$", range=(-1, 21))],
                        text=TextConfig(x=0.04, y=0.96, text=text_high_kt),
                        legend=LegendConfig(location="upper left", anchor=(0.02, 0.73)),
                    ),
                    # Ratio.
                    Panel(
                        axes=[AxisConfig("x", label=r"$z$", range=(0, 0.51)), AxisConfig("y", label="Pb--Pb/Hybrid")]
                    ),
                ],
            ),
            output_dir=output_dir,
        )

        _plot_compare_grooming_methods_for_attribute_data_embed(
            hists=hists,
            attr_name="n_high_kt",
            grooming_methods=[grooming_method],
            jet_pt_bin=jet_pt_bin,
            set_zero_to_nan=False,
            plot_config=PlotConfig(
                name="number_to_split_high_kt_grooming_methods",
                panels=[
                    # Main axis.
                    Panel(
                        axes=[AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n$")],
                        text=TextConfig(x=0.96, y=0.96, text=text_high_kt),
                        legend=LegendConfig(location="upper right", anchor=(0.96, 0.73)),
                    ),
                    # Ratio.
                    Panel(axes=[AxisConfig("x", label=r"$n$"), AxisConfig("y", label="Pb--Pb/Hybrid")]),
                ],
            ),
            output_dir=output_dir,
        )
