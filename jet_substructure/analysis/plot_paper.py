""" Paper plots

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import collections.abc
import copy
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pachyderm.plot
import pachyderm.plot as pb
import seaborn as sns
from pachyderm import binned_data

from jet_substructure.analysis import (
    full_results_helpers,
    model_calculations,
    plot_style,
    plot_unfolding,
    unfolding_analysis,
)
from jet_substructure.base import helpers

logger = logging.getLogger(__name__)

pb.configure()

_event_activity_short_label_map = {
    "pp": "pp",
    "central": r"0-10\%",
    "semi_central": r"30-50\%",
}
_event_activity_full_label_map = {
    "pp": "pp",
    "central": r"0-10\% $\text{Pb--Pb}$",
    "semi_central": r"30-50\% $\text{Pb--Pb}$",
}


def retrieve_model_styles(event_activity: str, model_name: str) -> dict[str, Any]:
    # Setup
    # NOTE: The danger of sometimes calling it `collision_system` when it's really event activity / collision system
    #       is that labeling can get confusing. But this is a good enough work around.
    _event_activity_to_model_styles_key_map = {
        "pp": "pp",
        "PbPb": "PbPb",
        "semi_central": "PbPb",
        "central": "PbPb",
    }

    model_styles = plot_style.define_paper_model_styles()
    return dict(model_styles[f"{_event_activity_to_model_styles_key_map[event_activity]}_{model_name}"])


def _plot_pp_grooming_comparison_with_models_2022(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    models: Mapping[str, Mapping[str, binned_data.BinnedData]],
    set_zero_to_nan: bool,
    kt_range: Mapping[str, helpers.KtRange],
    kt_ranges_for_models: Mapping[str, helpers.KtRange],
    models_to_normalize: Sequence[str],
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    grooming_styling = plot_style.define_grooming_styles()

    name_of_grooming_method_to_draw_models = ""
    for grooming_method in grooming_methods:
        res = [grooming_method in model_predictions for model_predictions in models.values()]  # noqa: F841
        all_models_contain_grooming_method = all(
            grooming_method in model_predictions for model_predictions in models.values()
        )
        if all_models_contain_grooming_method:
            name_of_grooming_method_to_draw_models = grooming_method
            break
    logger.info(f"name of name_of_grooming_method_to_draw_models: {name_of_grooming_method_to_draw_models}")

    with sns.color_palette("colorblind"):
    #with sns.color_palette("Accent"):
        # fig, ax = plt.subplots(figsize=(9, 10))
        # Size is specified to make it convenient to compare against Hard Probes plots.
        fig, (ax, ax_ratio_data, *ax_grooming_methods) = plt.subplots(
            1 + 1 + len(grooming_methods),
            1,
            figsize=(9, 14),
            gridspec_kw={"height_ratios": [4, 1] + [1] * len(grooming_methods)},
            sharex=True,
        )

        # Use selected grooming method as a reference, but only in the range where the others are measured.
        ratio_reference_hist_unselected = hists[reference_grooming_method].data

        for i_grooming_method, grooming_method in enumerate(grooming_methods):
            # plotting_last_method = grooming_method == grooming_methods[-1]

            # Axes: jet_pt, attr_name
            h_input = hists[grooming_method].data

            # Select range to display.
            h = full_results_helpers.select_hist_range(h_input, kt_range[grooming_method])

            # Set 0s to NaN
            if set_zero_to_nan:
                h.errors[h.values == 0] = np.nan
                h.values[h.values == 0] = np.nan

            # Main data points
            p = ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                yerr=h.errors,
                #xerr=h.axes[0].bin_widths / 2,
                xerr=np.zeros_like(h.axes[0].bin_widths),
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
                label=grooming_styling[grooming_method].label,
            )

            # Systematic uncertainty
            pachyderm.plot.error_boxes(
                ax=ax,
                x_data=h.axes[0].bin_centers,
                y_data=h.values,
                x_errors=h.axes[0].bin_widths / 2,
                y_errors=np.stack(
                    [
                        h.metadata["y_systematic"]["quadrature"].low,
                        h.metadata["y_systematic"]["quadrature"].high,
                    ]
                ),
                # y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
                # color=style.color,
                color=p[0].get_color(),
                linewidth=0,
            )

            # Data ratio
            # Skip drawing the reference grooming method in the data ratio because it's not meaningful.
            if grooming_method != reference_grooming_method:
                # Ensure the ratio is defined over the same range.
                kt_range_for_comparison = full_results_helpers.determine_overlapping_range(
                    current_range=kt_range[grooming_method],
                    reference=kt_range[reference_grooming_method]
                )
                logger.info(f"kt_range_for_comparison: {kt_range_for_comparison}")
                ratio_reference_hist = full_results_helpers.select_hist_range(
                    ratio_reference_hist_unselected,
                    kt_range_for_comparison,
                )
                h_for_ratio = full_results_helpers.select_hist_range(
                    h_input,
                    kt_range_for_comparison,
                )
                ratio = h_for_ratio / ratio_reference_hist
                # Ratio + statistical error bars
                ax_ratio_data.errorbar(
                    ratio.axes[0].bin_centers,
                    ratio.values,
                    yerr=ratio.errors,
                    #xerr=ratio.axes[0].bin_widths / 2,
                    xerr=np.zeros_like(ratio.axes[0].bin_widths),
                    color=p[0].get_color(),
                    marker="o",
                    markersize=11,
                    linestyle="",
                    linewidth=3,
                )
                # Systematic errors.
                y_relative_error_low = full_results_helpers.relative_error(
                    full_results_helpers.ErrorInput(value=h_for_ratio.values, error=h_for_ratio.metadata["y_systematic"]["quadrature"].low),
                    full_results_helpers.ErrorInput(
                        value=ratio_reference_hist.values,
                        error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].low,
                    ),
                )
                y_relative_error_high = full_results_helpers.relative_error(
                    full_results_helpers.ErrorInput(value=h_for_ratio.values, error=h_for_ratio.metadata["y_systematic"]["quadrature"].high),
                    full_results_helpers.ErrorInput(
                        value=ratio_reference_hist.values,
                        error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].high,
                    ),
                )
                # Store the systematic.
                ratio.metadata["y_systematic"]["quadrature"] = full_results_helpers.AsymmetricErrors(
                    low=y_relative_error_low * ratio.values,
                    high=y_relative_error_high * ratio.values,
                )
                y_systematic = ratio.metadata["y_systematic"]["quadrature"]
                pachyderm.plot.error_boxes(
                    ax=ax_ratio_data,
                    x_data=ratio.axes[0].bin_centers,
                    y_data=ratio.values,
                    x_errors=ratio.axes[0].bin_widths / 2,
                    y_errors=np.stack([y_systematic.low, y_systematic.high]),
                    color=p[0].get_color(),
                    linewidth=0,
                )

            # Next, the model comparison
            ax_ratio_models = ax_grooming_methods[i_grooming_method]

            # Add a line at 1 for reference.
            ax_ratio_models.axhline(y=1, color="black", linestyle="dashed", zorder=1)

            # First, plot the data systematics at 1. We only want to do this once.
            # To do so, we need the relative systematic errors.
            y_relative_error_low = full_results_helpers.relative_error(
                full_results_helpers.ErrorInput(
                    value=h.values,
                    error=h.metadata["y_systematic"]["quadrature"].low,
                ),
            )
            y_relative_error_high = full_results_helpers.relative_error(
                full_results_helpers.ErrorInput(
                    value=h.values,
                    error=h.metadata["y_systematic"]["quadrature"].high,
                ),
            )
            pachyderm.plot.error_boxes(
                ax=ax_ratio_models,
                x_data=h.axes[0].bin_centers,
                y_data=np.ones_like(h.values),
                x_errors=h.axes[0].bin_widths / 2,
                # NOTE: This will implicitly be relative to 1.
                y_errors=np.stack(
                    [y_relative_error_low, y_relative_error_high]
                ),
                color="grey",
                linewidth=0,
            )

            #_temp_i = 0
            colors = ["Blues_r", "Oranges_r", "Greens_r", "Reds_r"]  # noqa: F841
            #colors_for_models = sns.color_palette(colors[i_grooming_method], n_colors = 6)
            t = mpl.colors.LinearSegmentedColormap.from_list(
                f"{grooming_method}_col",
                N=6,
                # From white to the plotted color
                colors=[
                    (1, 1, 1),
                    #adjust_lightness(p[0].get_color(), 0.25)
                    p[0].get_color(),
                ]
            )
            colors_for_models = [t.reversed()(i/6.) for i in range(6)]
            for i_model, (model_name, model_with_all_grooming_methods) in enumerate(models.items()):
                model = model_with_all_grooming_methods.get(grooming_method, None)
                if not model:
                    logger.debug(
                        f"Skipping model {model_name}, grooming method: {grooming_method} because predictions aren't available"
                    )
                    continue

                # Then, plot the model
                model_style = grooming_styling[f"{grooming_method}_compare"]  # noqa: F841
                # Get the model hist
                model = binned_data.BinnedData.from_existing_data(model)

                # Then normalize as appropriate
                # NOTE: Careful, pythia is already normalized, but jetscape wasn't. So we only want to normalize in some cases
                if model_name in models_to_normalize:
                    model /= np.sum(model.values)
                    model /= model.axes[0].bin_widths

                # Determine the overlapping range, since not all of them are the same...
                kt_range_for_model_comparison = full_results_helpers.determine_overlapping_range(
                    current_range=kt_range[grooming_method],
                    reference=kt_ranges_for_models[reference_grooming_method]
                )
                logger.info(
                    f"kt_range_for_model_comparison: {kt_range_for_model_comparison}, {grooming_method}, {model_name}"
                )
                logger.info(f"kt_range_for_model: {kt_ranges_for_models[reference_grooming_method]}")

                # And select the same range.
                model_kt_range_selected = full_results_helpers.select_hist_range(model, kt_range_for_model_comparison)  # noqa: F841

                # And plot
                # Make sure we copy the settings so we can modify them
                # temp_kwargs = dict(plot_unfolding._models_styles[model_name])
                # temp_kwargs["label"] = (
                #     temp_kwargs["label"] if grooming_method == name_of_grooming_method_to_draw_models else None
                # )
                # For now, skip the models on the main plot so that it doesn't get too busy
                # ax.errorbar(
                #     model_kt_range_selected.axes[0].bin_centers + (0.1 * _temp_i),
                #     model_kt_range_selected.values,
                #     # yerr=model_kt_range_selected.errors,
                #     # xerr=model.axes[0].bin_widths / 2,
                #     #color=grooming_styling[grooming_method].color,
                #     color=p[0].get_color(),
                #     # marker=style.marker,
                #     # fillstyle=grooming_styling[grooming_method].fillstyle,
                #     # linestyle="",
                #     # label=_models_styles[model_name]["label"] if plotting_last_method else None,
                #     zorder=model_style.zorder,
                #     alpha=0.7,
                #     **temp_kwargs,
                # )

                # _temp_i += 1

                model_for_ratio = full_results_helpers.select_hist_range(
                    model,
                    kt_range_for_model_comparison,
                )
                h_for_model_ratio = full_results_helpers.select_hist_range(
                    h_input,
                    kt_range_for_model_comparison,
                )

                # Ratio
                ratio = model_for_ratio / h_for_model_ratio

                # Ratio + statistical error bars
                temp_kwargs: dict[str, Any] = dict(plot_unfolding._models_styles[model_name])
                temp_kwargs["label"] = (
                    # For all in one panel
                    temp_kwargs["label"] if grooming_method == name_of_grooming_method_to_draw_models else None
                    # NOTE: Could generalize if this looks okay...
                    #temp_kwargs["label"] if (model_name in ["pythia", "sherpa_ahadic", "sherpa_lund"] and grooming_method == "dynamical_core") or (model_name in ["analytical", "jetscape"] and grooming_method == "dynamical_kt") else None
                    # For one label per axis
                    #temp_kwargs["label"] if (model_name == list(models)[i_grooming_method]) else None
                )
                #temp_kwargs["color"] = colors_for_models[4 - i_model]
                temp_kwargs["color"] = colors_for_models[i_model]
                ax_ratio_models.errorbar(
                    ratio.axes[0].bin_centers,
                    ratio.values,
                    yerr=ratio.errors,
                    #xerr=ratio.axes[0].bin_widths / 2,
                    #color=p[0].get_color(),
                    #marker="o",
                    markersize=11,
                    #linestyle="",
                    #linewidth=3,
                    **temp_kwargs,
                )

                # For theory curves with systematic uncertainties, such as the analytical calculations,
                # we need to propagate the systematics. It's a bit awkward since the systematic bar is already
                # plotted, but I don't see an obviously better way forward
                if "y_systematic" in model_for_ratio.metadata:
                    y_relative_error_low = full_results_helpers.relative_error(
                        full_results_helpers.ErrorInput(value=h_for_model_ratio.values, error=h_for_model_ratio.metadata["y_systematic"]["quadrature"].low),
                        full_results_helpers.ErrorInput(
                            value=model_for_ratio.values,
                            error=model_for_ratio.metadata["y_systematic"]["quadrature"].low,
                        ),
                    )
                    y_relative_error_high = full_results_helpers.relative_error(
                        full_results_helpers.ErrorInput(value=h_for_model_ratio.values, error=h_for_model_ratio.metadata["y_systematic"]["quadrature"].high),
                        full_results_helpers.ErrorInput(
                            value=model_for_ratio.values,
                            error=model_for_ratio.metadata["y_systematic"]["quadrature"].high,
                        ),
                    )
                    model_systematic = full_results_helpers.AsymmetricErrors(
                        low=y_relative_error_low * ratio.values,
                        high=y_relative_error_high * ratio.values,
                    )

                    pachyderm.plot.error_boxes(
                        ax=ax_ratio_models,
                        x_data=ratio.axes[0].bin_centers,
                        y_data=ratio.values,
                        x_errors=ratio.axes[0].bin_widths / 2,
                        y_errors=np.stack(
                            [model_systematic.low, model_systematic.high]
                        ),
                        linewidth=0,
                        color=temp_kwargs["color"],
                    )

    # Add legend for model. We're doing some tricks here, so we need to do it by hand.
    ax_ratio_models = ax_grooming_methods[grooming_methods.index(name_of_grooming_method_to_draw_models)]
    models_legend = ax_ratio_models.legend(frameon=False, loc="lower left", fontsize=22)
    handles, labels = ax_ratio_models.get_legend_handles_labels()
    logger.info(f"models_legend handles: {ax_ratio_models.get_legend_handles_labels()}")
    #ax_ratio_models.legend().set_visible(False)
    # Remove from the existing axis.
    models_legend.remove()
    logger.info(f"models_legend: {models_legend}")
    # Make the legend on the main axis.
    models_legend = ax.legend(handles=handles, labels=labels, frameon=False, loc="lower left", fontsize=22)

    # reference value for data and model ratios
    ax_ratio_data.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio_data, *ax_grooming_methods])
    # A few additional tweaks.
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
    # Add the second legend
    ax.add_artist(models_legend)

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_pp_grooming_comparison_with_models_2022(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    models: Mapping[str, Mapping[str, binned_data.BinnedData]],
    collision_system: str,
    collision_system_key: str,
    jet_R_str: str,
    output_dir: Path,
    kt_range: helpers.KtRange | Mapping[str, helpers.KtRange],
    kt_ranges_for_models: Mapping[str, helpers.KtRange],
    models_to_normalize: Sequence[str],
    figure_kt_range: helpers.KtRange | None = None,
) -> None:
    """Plot comparison of grooming methods, along with models, for pp."""

    # Validation
    if isinstance(kt_range, helpers.KtRange):
        kt_range = {grooming_method: kt_range for grooming_method in grooming_methods}
    if figure_kt_range is None:
        figure_kt_range = helpers.KtRange(1.5, 15)

    grooming_styling = plot_style.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    # Grooming method panels
    grooming_method_panels = [
        pb.Panel(
            axes=[
                #pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=tuple(figure_kt_range), font_size=22),  # type: ignore[arg-type]
                pb.AxisConfig(
                    "y",
                    label=r"$\frac{\text{Model}}{\text{DyG}\;a=0.5}$",
                    range=(0.05, 2.95),
                    font_size=22,
                ),
            ],
            #legend=pb.LegendConfig(location="upper center", font_size=20, anchor=(0.5, 0.90)),
        ),
        pb.Panel(
            axes=[
                #pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=tuple(figure_kt_range), font_size=22),  # type: ignore[arg-type]
                pb.AxisConfig(
                    "y",
                    label=r"$\frac{\text{Model}}{\text{DyG}\;a=1}$",
                    range=(0.05, 2.95),
                    font_size=22,
                ),
            ],
            #legend=pb.LegendConfig(location="upper center", font_size=20, anchor=(0.5, 0.90)),
        ),
        pb.Panel(
            axes=[
                #pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=tuple(figure_kt_range), font_size=22),  # type: ignore[arg-type]
                pb.AxisConfig(
                    "y",
                    label=r"$\frac{\text{Model}}{\text{DyG}\;a=2}$",
                    range=(0.05, 1.95),
                    font_size=22,
                ),
            ],
            #legend=pb.LegendConfig(location="upper center", font_size=20, anchor=(0.5, 0.90)),
        ),
        pb.Panel(
            axes=[
                pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=tuple(figure_kt_range), font_size=22),  # type: ignore[arg-type]
                pb.AxisConfig(
                    "y",
                    label=r"$\frac{\text{Model}}{\text{SD}\;z>0.2}$",
                    range=(0.05, 1.95),
                    font_size=22,
                ),
            ],
            #legend=pb.LegendConfig(location="upper center", font_size=20, anchor=(0.5, 0.90)),
        ),
    ]

    text = plot_style.label_to_display_string["ALICE"]["work_in_progress"]
    text += "\n" + plot_style.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + plot_style.label_to_display_string["jets"]["general"]
    text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_pp_grooming_comparison_with_models_2022(
        hists=hists,
        grooming_methods=grooming_methods,
        reference_grooming_method=reference_grooming_method,
        models=models,
        set_zero_to_nan=False,
        kt_range=kt_range,
        kt_ranges_for_models=kt_ranges_for_models,
        models_to_normalize=models_to_normalize,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system}_data_model_comparison_{jet_R_str}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            #range=(5e-3, 1),
                            range=(9e-3, 1),
                            font_size=22,
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="center right", anchor=(0.985, 0.52), font_size=22),
                ),
                # Data ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{Method}}{\text{"
                            + grooming_styling[reference_grooming_method].label
                            + "}}$",
                            #range=(0.3, 1.7),
                            range=(0.3, 1.9),
                            font_size=22,
                        ),
                    ],
                ),
                # Grooming method specific panels
                *grooming_method_panels
            ],
            figure=pb.Figure(edge_padding={"left": 0.13, "bottom": 0.06}),
        ),
        output_dir=output_dir,
    )


def _determine_uncertainty_lower_upper_for_model(model: binned_data.BinnedData) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Based on whether the model contains a systematic uncertainty or not."""
    if "y_systematic" in model.metadata:
        if not np.allclose(model.errors, np.zeros(len(model.errors))):
            msg = "Model has both statistical and systematic uncertainties. This is not yet supported."
            raise ValueError(msg)
        lower_error = model.metadata["y_systematic"]["quadrature"].low
        upper_error = model.metadata["y_systematic"]["quadrature"].high
    else:
        lower_error = model.errors
        upper_error = model.errors
    return lower_error, upper_error


def _plot_data_model_comparison_for_single_system(  # noqa: C901
    hists: Mapping[str, unfolding_analysis.SingleResult],
    models: Mapping[str, model_calculations.ModelCalculation],
    grooming_methods: Sequence[str],
    collision_system: str,
    set_zero_to_nan: bool,
    all_methods_on_one_figure: bool,
    kt_range: Mapping[str, helpers.KtRange],
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """Plot data/model comparison for a single collision system.

    Args:
        hists: Mapping from grooming method to unfolding result.
        models: Mapping from model name to model calculation.
        grooming_methods: List of grooming methods.
        collision_system: Collision system.
        set_zero_to_nan: Whether to set zero bins to NaN.
        all_methods_on_one_figure: Whether to plot all grooming methods on one figure.
        kt_range: Mapping from grooming method to kt range.
        plot_config: Plot configuration.
        output_dir: Output directory.
    Returns:
        None.
    """
    grooming_styles = plot_style.define_paper_grooming_styles()

    if all_methods_on_one_figure:
        n_panels = int(np.ceil(len(grooming_methods) / 2))
        fig, all_axes = plt.subplots(
            4,
            n_panels,
            figsize=(7.5 * n_panels, 15),
            gridspec_kw={"height_ratios": [3, 1, 3, 1]},
            sharex="col",
            sharey="row",
        )
        ax_pairs = [  # noqa: C416
            (ax, ax_ratio)
            # NOTE: This is tricky because all_axes is 2x2 here, so stepping by 2 goes to
            #       the next row!
            for ax, ax_ratio in zip(all_axes[::2].flatten(), all_axes[1::2].flatten())
        ]
    else:
        fig, all_axes = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        ax, ax_ratio = all_axes
        ax_pairs = [
            (ax, ax_ratio)
            for _ in range(len(grooming_methods))
        ]

    for _plot_counter, (grooming_method, (ax, ax_ratio)) in enumerate(zip(grooming_methods, ax_pairs)):
        plotting_last_method = grooming_method == grooming_methods[-1]

        # First, the data
        h = hists[grooming_method].data

        # Select range to display.
        h = full_results_helpers.select_hist_range(h, kt_range[grooming_method])

        # Set 0s to NaN
        if set_zero_to_nan:
            h.errors[h.values == 0] = np.nan
            h.values[h.values == 0] = np.nan

        # Plot options
        kwargs_plot_errorbar = grooming_styles[grooming_method].kwargs_for_plot_errorbar()

        # Main data points
        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            label=grooming_styles[grooming_method].label_short,
            # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
            # NOTE: The extra 2 * is to ensure we stay on top of the systematic error boxes
            zorder=3 + 2 * len(grooming_methods) - _plot_counter,
            **kwargs_plot_errorbar,
        )

        # Systematic uncertainty
        kwargs_plot_error_boxes = grooming_styles[grooming_method].kwargs_for_plot_error_boxes()
        kwargs_plot_error_boxes["zorder"] = 2 + len(grooming_methods) - _plot_counter
        pachyderm.plot.error_boxes(
            ax=ax,
            x_data=h.axes[0].bin_centers,
            y_data=h.values,
            x_errors=h.axes[0].bin_widths / 2,
            y_errors=np.array(
                [
                    h.metadata["y_systematic"]["quadrature"].low,
                    h.metadata["y_systematic"]["quadrature"].high,
                ]
            ),
            **kwargs_plot_error_boxes,
        )

        # Next, draw the data and uncertainties at one as black and grey boxes
        # Ratio + statistical error bars at one
        kwargs_plot_errorbar = grooming_styles[grooming_method].kwargs_for_plot_errorbar()
        kwargs_plot_errorbar["color"] = "black"
        kwargs_plot_errorbar["markeredgecolor"] = "black"
        kwargs_plot_errorbar["markerfacecolor"] = "white" if kwargs_plot_errorbar["markerfacecolor"] == "white" else "black"
        kwargs_plot_errorbar["zorder"] = 6
        ax_ratio.errorbar(
            h.axes[0].bin_centers,
            np.ones_like(h.axes[0].bin_centers),
            yerr=h.errors / h.values,
            xerr=h.axes[0].bin_widths / 2,
            **kwargs_plot_errorbar,
        )
        kwargs_plot_error_boxes = grooming_styles[grooming_method].kwargs_for_plot_error_boxes()
        kwargs_plot_error_boxes["color"] = "black"
        kwargs_plot_error_boxes["zorder"] = 5.5
        pachyderm.plot.error_boxes(
            ax=ax_ratio,
            x_data=h.axes[0].bin_centers,
            y_data=np.ones_like(h.values),
            x_errors=h.axes[0].bin_widths / 2,
            y_errors=np.array(
                [
                    h.metadata["y_systematic"]["quadrature"].low / h.values,
                    h.metadata["y_systematic"]["quadrature"].high / h.values,
                ]
            ),
            **kwargs_plot_error_boxes,
        )

        for model_name, model_calculation in models.items():
            model = model_calculation.spectra(event_activity=collision_system).get(grooming_method, None)
            if not model:
                logger.debug(
                    f"Skipping model {model_name}, grooming method: {grooming_method} because predictions aren't available"
                )
                continue

            # Then, plot the model
            # Get the model for the reference.
            model = binned_data.BinnedData.from_existing_data(model)
            # And select the same range.
            # NOTE: Because some models may not cover the entire kt range, we need to explicitly allow that here
            model = full_results_helpers.select_hist_range(model, kt_range[grooming_method], allow_range_broader_than_bin_edges=True)

            # Further setup
            # NOTE: If we naively construct the ratio here by just dividing the model by the data,
            #       then the errors stored in the ratio aren't what we want since they convolve the
            #       model uncertainties with the data uncertainties. So want to calculate the ratio
            #       using a hist without the data uncertainties.
            # NOTE: We define this here (ie. early) so we can decide what to rebin (which) we need to
            #       know before we plot the model
            h_without_uncertainties = binned_data.BinnedData(
                axes=[h.axes[0].bin_edges],
                # NOTE: The `np.array` is really important here because we need to make a copy!
                #       Otherwise, we modify the underlying values, and everything gets fucked up in
                #       future loop iterations.
                values=np.array(h.values),
                variances=np.zeros_like(h.values),
            )

            # Check that binning matches up. If it doesn't attempt to rebin
            if h_without_uncertainties.axes[0].bin_edges.shape != model.axes[0].bin_edges.shape or \
                not np.allclose(h_without_uncertainties.axes[0].bin_edges, model.axes[0].bin_edges):
                # Rebin according to the data which we are supposed to be plotting
                # NOTE: We take as a proxy that whichever hist has more bins is the one that needs to be rebinned.
                #       We can't just assume that the model is more finely binned because some (eg. Caucal) is not.
                if h_without_uncertainties.axes[0].bin_edges.shape[0] > model.axes[0].bin_edges.shape[0]:
                    h_without_uncertainties = full_results_helpers.rebin_bin_width_scaled_hist(
                        h_to_rebin=h_without_uncertainties,
                        h_target_axis=model.axes[0],
                        # This is okay since the data is explicitly constructed without systematic systematic uncertainties.
                        okay_for_systematic_not_to_exist=True,
                    )
                else:
                    model = full_results_helpers.rebin_bin_width_scaled_hist(
                        h_to_rebin=model,
                        h_target_axis=h_without_uncertainties.axes[0],
                        # This is okay since the model doesn't usually have a systematic uncertainty.
                        okay_for_systematic_not_to_exist=True,
                    )

            # And plot
            # Make sure we copy the settings so we can modify them
            temp_kwargs = retrieve_model_styles(event_activity=collision_system, model_name=model_name)
            temp_kwargs["label"] = model_calculation.label(collision_system=collision_system) if plotting_last_method and not all_methods_on_one_figure else None
            # Need to pop for fill_between since these aren't valid args
            temp_kwargs.pop("marker")
            temp_kwargs.pop("markerfacecolor", None)
            temp_kwargs.pop("markeredgewidth", None)
            # And switch to the proper color
            temp_kwargs["facecolor"] = temp_kwargs.pop("color")
            lower_error, upper_error = _determine_uncertainty_lower_upper_for_model(model=model)

            ax.fill_between(
                model.axes[0].bin_centers,
                model.values - lower_error,
                model.values + upper_error,
                zorder=5,
                alpha=0.75,
                **temp_kwargs,
            )

            # Ratio
            ratio = model / h_without_uncertainties

            # We need to propagate the systematic uncertainty manually since the data is constructed not to have
            # uncertainties that we would usually propagate with
            if "y_systematic" in model.metadata:
                y_relative_error_low = full_results_helpers.relative_error(
                    full_results_helpers.ErrorInput(value=model.values, error=model.metadata["y_systematic"]["quadrature"].low),
                )
                y_relative_error_high = full_results_helpers.relative_error(
                    full_results_helpers.ErrorInput(value=model.values, error=model.metadata["y_systematic"]["quadrature"].high),
                )
                ratio_systematic = full_results_helpers.AsymmetricErrors(
                    low=y_relative_error_low * ratio.values,
                    high=y_relative_error_high * ratio.values,
                )
                ratio.metadata["y_systematic"]["quadrature"] = ratio_systematic

            # Finally, plot the band in the ratio
            lower_error, upper_error = _determine_uncertainty_lower_upper_for_model(model=ratio)
            ax_ratio.fill_between(
                ratio.axes[0].bin_centers,
                ratio.values - lower_error,
                ratio.values + upper_error,
                zorder=5,
                alpha=0.75,
                **temp_kwargs,
            )

        if all_methods_on_one_figure:
            # Reference value for ratio
            ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=0.9)

    if not all_methods_on_one_figure:
        # Reference value for ratio
        ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=0.9)

    if all_methods_on_one_figure:
        # We want to split the legend into two separate entries in the bottom right main panel
        # To do so, we need to heavily edit the legend. We need to do this manually since it's quite complicated.
        panel_config = plot_config.panels[n_panels * 3 - 1]
        legend_config = panel_config.legend
        assert legend_config is not None
        # First, we define the config for the legend. This way, we'll always have the same settings except for the location
        legend_models = copy.deepcopy(legend_config)
        legend_models.location = "upper right"
        legend_models.anchor = (0.98, 0.98)

        # Create handles and labels by hand, using all models
        legend_elements = []
        for model_name, model_calculation in models.items():
            model_kwargs = retrieve_model_styles(event_activity=collision_system, model_name=model_name)
            legend_elements.append(
                mpl.patches.Patch(
                    facecolor=model_kwargs["color"],
                    label=model_calculation.label(collision_system=collision_system)
                )
            )

        # Next, before we create the new legend, we need the existing data point handles
        ax_legend = all_axes[::2].flatten()[-1]
        handles, labels = ax_legend.get_legend_handles_labels()

        # Now that we have the handles, we can apply
        # NOTE: As a convention, we decide to use legend_config for the data, and we create the new legend for the models.
        legend_data_obj = legend_config.apply(
            ax=ax_legend,
            legend_handles=handles,
            legend_labels=labels,
        )
        legend_models_obj = legend_models.apply(
            ax=ax_legend,
            legend_handles=legend_elements,
        )
        # Now that we've gotten the legends all figured out, we need to make sure that the standard formatting doesn't interfere.
        # We do this by removing the legend config
        plot_config.panels[n_panels * 3 - 1].legend = None
        # And then add the legend objects back to the axis (dumb, but apparently required)
        ax_legend.add_artist(legend_data_obj)
        ax_legend.add_artist(legend_models_obj)

        # Labeling and presentation
        plot_config.apply(fig=fig, axes=list(all_axes.flatten()))

        # A few additional tweaks.
        for _ax in all_axes[::2].flatten():
            _ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
    else:
        # Labeling and presentation
        plot_config.apply(fig=fig, axes=[ax, ax_ratio])
        # A few additional tweaks.
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))

    filename = f"{plot_config.name}"
    if len(list(grooming_methods)) < 5:
        filename += f"_{'_'.join(grooming_methods)}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_grooming_methods_comparison_with_model_for_single_system(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    models: Mapping[str, model_calculations.ModelCalculation],
    grooming_methods: Sequence[str],
    collision_system: str,
    collision_system_key: str,
    output_dir: Path,
    kt_range: helpers.KtRange | Mapping[str, helpers.KtRange],
    figure_kt_range: helpers.KtRange | None = None,
    jet_R_str: str = "R04",
    alice_status: str = "work_in_progress",
    text_font_size: int = 31,
) -> None:
    """Plot comparison of grooming methods for a single system."""
    # Validation
    if figure_kt_range is None:
        figure_kt_range = helpers.KtRange(1.5, 15)
    if isinstance(kt_range, helpers.KtRange):
        kt_range = {grooming_method: kt_range for grooming_method in grooming_methods}

    # Setup
    event_activity = ""
    if collision_system != "pp":
        event_activity = f"{_event_activity_short_label_map[collision_system]} "
    collision_system_filename_label = collision_system
    if collision_system != "pp":
        collision_system_filename_label = f"PbPb_{collision_system_filename_label}"
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = plot_style.label_to_display_string["ALICE"][alice_status]
    # Since the final text is short, we can merge onto one line
    if alice_status != "final":
        text += "\n"
    else:
        text += " "
    text += event_activity + plot_style.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + plot_style.label_to_display_string["jets"]["general"]
    text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_data_model_comparison_for_single_system(
        hists=hists,
        models=models,
        grooming_methods=grooming_methods,
        collision_system=collision_system,
        set_zero_to_nan=False,
        all_methods_on_one_figure=False,
        kt_range=kt_range,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system_filename_label}_model_comparison_{jet_R_str}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            range=(4e-3, 1),
                            font_size=text_font_size,
                        ),
                    ],
                    text=pb.TextConfig(x=0.98, y=0.98, text=text, font_size=text_font_size),
                    legend=pb.LegendConfig(location="lower left", font_size=round(text_font_size*0.8), anchor=(0.015, 0.025), marker_label_spacing=0.075),
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$k_{\text{T,g}}\:(\text{GeV}/c)$",
                            range=tuple(figure_kt_range),  # type: ignore[arg-type]
                            font_size=text_font_size,
                        ),
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{Model}}{\text{Data}}$",
                            range=(0.1, 1.9) if "soft_drop_z_cut_04" in grooming_methods else (0.45, 1.55),
                            font_size=text_font_size,
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.15, "bottom": 0.095, "top": 0.975}),
        ),
        output_dir=output_dir,
    )


def plot_grooming_methods_comparison_with_model_for_single_system_one_figure(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    models: Mapping[str, model_calculations.ModelCalculation],
    grooming_methods: Sequence[str],
    collision_system: str,
    collision_system_key: str,
    output_dir: Path,
    kt_range: helpers.KtRange | Mapping[str, helpers.KtRange],
    main_panel_y_axis_range: tuple[float | None, float | None],
    ratio_y_axis_range: tuple[float | None, float | None],
    figure_kt_range: helpers.KtRange | None = None,
    jet_R_str: str = "R04",
    alice_status: str = "work_in_progress",
    text_font_size: int = 31,
) -> None:
    """Plot comparison of grooming methods for a single system in one composite figure."""
    # Validation
    if figure_kt_range is None:
        figure_kt_range = helpers.KtRange(1.5, 15)
    if isinstance(kt_range, helpers.KtRange):
        kt_range = {grooming_method: kt_range for grooming_method in grooming_methods}

    # Setup
    event_activity = ""
    if collision_system != "pp":
        event_activity = f"{_event_activity_short_label_map[collision_system]} "
    collision_system_filename_label = collision_system
    if collision_system != "pp":
        collision_system_filename_label = f"PbPb_{collision_system_filename_label}"
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = plot_style.label_to_display_string["ALICE"][alice_status]
    # Since the final text is short, we can merge onto one line
    if alice_status != "final":
        text += "\n"
    else:
        text += " "
    text += event_activity + plot_style.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + plot_style.label_to_display_string["jets"]["general"]
    text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"

    # Setup panels
    # We need to handle this carefully, so we do it slowly, and step-by-step
    # NOTE: The deepcopy calls are critical - otherwise, we may accidentally modify one config
    #       when we modify another.
    n_horizontal_panels = int(np.ceil(len(grooming_methods) / 2))
    panels = []
    # Start with main panels
    main_panel_standard = pb.Panel(
        axes=[
            pb.AxisConfig(
                "x",
                range=tuple(figure_kt_range),  # type: ignore[arg-type]
                font_size=text_font_size,
            ),
            pb.AxisConfig(
                "y",
                log=True,
                range=main_panel_y_axis_range,
                font_size=text_font_size,
            ),
        ],
        legend=pb.LegendConfig(location="lower left", font_size=round(text_font_size*0.8), anchor=(0.015, 0.025), marker_label_spacing=0.075),
    )
    main_panel_first = copy.deepcopy(main_panel_standard)
    main_panel_first.axes[1].label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$"
    # Full ALICE label in panel
    main_panel_full_label = copy.deepcopy(main_panel_standard)
    main_panel_full_label.text = pb.TextConfig(x=0.98, y=0.98, text=text, font_size=round(text_font_size * 0.9)),
    panels.append(main_panel_first)
    # NOTE: We can't simply copy the list with * 2 since that would do a simple copy of the object
    #       (defeating the purpose of the deepcopy).
    panels.extend([copy.deepcopy(main_panel_standard) for _ in range(n_horizontal_panels - 2)])
    panels.append(main_panel_full_label)
    # Next, onto ratio panels
    ratio_panel_mid_standard = pb.Panel(
        axes=[
            pb.AxisConfig(
                "x",
                range=tuple(figure_kt_range),  # type: ignore[arg-type]
                font_size=text_font_size,
            ),
            pb.AxisConfig(
                "y",
                range=ratio_y_axis_range,
                font_size=text_font_size,
            ),
        ],
    )
    # Mid left needs the label
    ratio_panel_mid_left = copy.deepcopy(ratio_panel_mid_standard)
    ratio_panel_mid_left.axes[1].label = r"$\frac{\text{Model}}{\text{Data}}$"
    panels.append(ratio_panel_mid_left)
    # Fill out the last ones as standard ratios
    panels.extend([copy.deepcopy(ratio_panel_mid_standard) for _ in range(n_horizontal_panels - 1)])
    # Next row of main panels
    panels.append(copy.deepcopy(main_panel_first))
    # Fill out the last ones as standard
    panels.extend([copy.deepcopy(main_panel_standard) for _ in range(n_horizontal_panels - 1)])
    # Finish with the rest of the ratios
    ratio_panel_bottom_standard = copy.deepcopy(ratio_panel_mid_standard)
    ratio_panel_bottom_standard.axes[0].label = r"$k_{\text{T,g}}\:(\text{GeV}/c)$"
    ratio_panel_bottom_left = copy.deepcopy(ratio_panel_bottom_standard)
    ratio_panel_bottom_left.axes[1].label = r"$\frac{\text{Model}}{\text{Data}}$"
    panels.append(ratio_panel_bottom_left)
    # Fill out the last ones as standard
    panels.extend([copy.deepcopy(ratio_panel_bottom_standard)] * (n_horizontal_panels - 1))

    _plot_data_model_comparison_for_single_system(
        hists=hists,
        models=models,
        grooming_methods=grooming_methods,
        collision_system=collision_system,
        set_zero_to_nan=False,
        all_methods_on_one_figure=True,
        kt_range=kt_range,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system_filename_label}_model_comparison_{jet_R_str}_one_figure",
            panels=panels,
            figure=pb.Figure(edge_padding={"left": 0.05, "bottom": 0.08, "top": 0.98, "right": 0.98}),
        ),
        output_dir=output_dir,
    )


def _plot_single_system_comparison(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    set_zero_to_nan: bool,
    kt_range: Mapping[str, helpers.KtRange | helpers.RgRange],
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """Plot comparison of grooming methods for a single system.

    Args:
        hists: Mapping of grooming method to SingleResult.
        grooming_methods: List of grooming methods to plot.
        reference_grooming_method: Grooming method to use as reference.
        set_zero_to_nan: Whether to set zero bins to NaN.
        kt_range: Mapping of grooming method to kt range.
        plot_config: Plot configuration.
        output_dir: Output directory.
    """
    grooming_styles = plot_style.define_paper_grooming_styles()

    # fig, ax = plt.subplots(figsize=(9, 10))
    # Size is specified to make it convenient to compare against Hard Probes plots.
    fig, all_axes = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax, ax_ratio = all_axes
    ax_pairs = [
        (ax, ax_ratio)
        for _ in range(len(grooming_methods))
    ]

    # Use selected grooming method as a reference, but only in the range where the others are measured.
    ratio_reference_hist_unselected = hists[reference_grooming_method].data

    for _plot_counter, (grooming_method, (ax, ax_ratio)) in enumerate(zip(grooming_methods, ax_pairs)):
        # Axes: jet_pt, attr_name
        h_input = hists[grooming_method].data

        # Select range to display.
        h = full_results_helpers.select_hist_range(h_input, kt_range[grooming_method])

        # Set 0s to NaN
        if set_zero_to_nan:
            h.errors[h.values == 0] = np.nan
            h.values[h.values == 0] = np.nan

        # Plot options
        kwargs_plot_errorbar = grooming_styles[grooming_method].kwargs_for_plot_errorbar()

        # Main data points
        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            label=grooming_styles[grooming_method].label_short,
            # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
            # NOTE: The extra 2 * is to ensure we stay on top of the systematic error boxes
            zorder=3 + 2 * len(grooming_methods) - _plot_counter,
            **kwargs_plot_errorbar,
        )

        # Systematic uncertainty
        kwargs_plot_error_boxes = grooming_styles[grooming_method].kwargs_for_plot_error_boxes()
        kwargs_plot_error_boxes["zorder"] = 2 + len(grooming_methods) - _plot_counter
        pachyderm.plot.error_boxes(
            ax=ax,
            x_data=h.axes[0].bin_centers,
            y_data=h.values,
            x_errors=h.axes[0].bin_widths / 2,
            y_errors=np.array(
                [
                    h.metadata["y_systematic"]["quadrature"].low,
                    h.metadata["y_systematic"]["quadrature"].high,
                ]
            ),
            **kwargs_plot_error_boxes,
        )

        # Ratio
        # Skip the reference method because it's not meaningful for the ratio.
        if grooming_method == reference_grooming_method:
            continue

        # Ensure the ratio is defined over the same range.
        kt_range_for_comparison = full_results_helpers.determine_overlapping_range(
            current_range=kt_range[grooming_method],
            reference=kt_range[reference_grooming_method]
        )
        logger.info(f"kt_range_for_comparison: {kt_range_for_comparison}")
        ratio_reference_hist = full_results_helpers.select_hist_range(
            ratio_reference_hist_unselected,
            kt_range_for_comparison,
        )
        h = full_results_helpers.select_hist_range(
            h_input,
            kt_range_for_comparison,
        )
        # Check that binning matches up. If it doesn't attempt to rebin
        if h.axes[0].bin_edges.shape != ratio_reference_hist.axes[0].bin_edges.shape or \
            not np.allclose(h.axes[0].bin_edges, ratio_reference_hist.axes[0].bin_edges):
            # Rebin according to the data which we are supposed to be plotting
            ratio_reference_hist = full_results_helpers.rebin_bin_width_scaled_hist(
                h_to_rebin=ratio_reference_hist.copy(),
                h_target_axis=h.axes[0],
            )

        ratio = h / ratio_reference_hist
        # Ratio + statistical error bars
        ax_ratio.errorbar(
            ratio.axes[0].bin_centers,
            ratio.values,
            yerr=ratio.errors,
            xerr=ratio.axes[0].bin_widths / 2,
            # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
            # NOTE: The extra 2 * is to ensure we stay on top of the systematic error boxes
            zorder=3 + 2 * len(grooming_methods) - _plot_counter,
            **kwargs_plot_errorbar,
        )
        # Systematic errors.
        y_relative_error_low = full_results_helpers.relative_error(
            full_results_helpers.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].low),
            full_results_helpers.ErrorInput(
                value=ratio_reference_hist.values,
                error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].low,
            ),
        )
        y_relative_error_high = full_results_helpers.relative_error(
            full_results_helpers.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].high),
            full_results_helpers.ErrorInput(
                value=ratio_reference_hist.values,
                error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].high,
            ),
        )

        # Store the systematic.
        ratio.metadata["y_systematic"]["quadrature"] = full_results_helpers.AsymmetricErrors(
            low=y_relative_error_low * ratio.values,
            high=y_relative_error_high * ratio.values,
        )
        y_systematic = ratio.metadata["y_systematic"]["quadrature"]
        pachyderm.plot.error_boxes(
            ax=ax_ratio,
            x_data=ratio.axes[0].bin_centers,
            y_data=ratio.values,
            x_errors=ratio.axes[0].bin_widths / 2,
            y_errors=np.array([y_systematic.low, y_systematic.high]),
            **kwargs_plot_error_boxes,
        )

    # Reference value for ratio
    ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=0.9)

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])
    # A few additional tweaks.
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_comparisons_of_grooming_methods_for_single_system(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    collision_system: str,
    collision_system_key: str,
    output_dir: Path,
    kt_range: helpers.KtRange | Mapping[str, helpers.KtRange],
    figure_kt_range: helpers.KtRange | None = None,
    ratio_y_range: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    jet_R_str: str = "R04",
    alice_status: str = "work_in_progress",
    text_font_size: int = 31,
    label: str = "",
) -> None:
    """Plot comparison of grooming methods for a single system."""
    # Validation
    if figure_kt_range is None:
        figure_kt_range = helpers.KtRange(1.5, 15)
    if isinstance(kt_range, helpers.KtRange):
        kt_range = {grooming_method: kt_range for grooming_method in [*grooming_methods, reference_grooming_method]}
    if ratio_y_range is None:
        ratio_y_range = (0.45, 1.55) if "soft_drop_z_cut_04" not in grooming_methods else (0.1, 1.9)
    if isinstance(ratio_y_range, tuple):
        ratio_y_range_map = {grooming_method: ratio_y_range for grooming_method in [*grooming_methods, reference_grooming_method]}
        ratio_y_range = (
            min([r[0] for r in ratio_y_range_map.values()]),
            max([r[1] for r in ratio_y_range_map.values()]),
        )
    assert isinstance(ratio_y_range, tuple), f"Invalid y range for ratio. Provided: {ratio_y_range}"
    if label:
        label = f"_{label}"

    # Add event activity to label if needed
    event_activity = ""
    if collision_system != "pp":
        event_activity = f"{_event_activity_short_label_map[collision_system]} "
    collision_system_filename_label = collision_system
    if collision_system != "pp":
        collision_system_filename_label = f"PbPb_{collision_system_filename_label}"

    grooming_styling = plot_style.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = plot_style.label_to_display_string["ALICE"][alice_status]
    # We skip this in PbPb because the name becomes too long!
    if alice_status != "final" or collision_system != "pp":
        text += "\n"
    else:
        text += " "
    text += event_activity + plot_style.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + plot_style.label_to_display_string["jets"]["general"]
    text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_single_system_comparison(
        hists=hists,
        grooming_methods=grooming_methods,
        reference_grooming_method=reference_grooming_method,
        set_zero_to_nan=False,
        kt_range=kt_range,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system_filename_label}_grooming_method_comparison_{jet_R_str}{label}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            range=(4e-3, 1),
                            font_size=text_font_size,
                        ),
                    ],
                    text=pb.TextConfig(x=0.98, y=0.98, text=text, font_size=text_font_size),
                    legend=pb.LegendConfig(location="lower left", font_size=round(text_font_size*0.8), anchor=(0.015, 0.025), marker_label_spacing=0.075),
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T,g}}\:(\text{GeV}/c)$", range=tuple(figure_kt_range), font_size=text_font_size),  # type: ignore[arg-type]
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{Method}}{\text{"
                            + grooming_styling[reference_grooming_method].label_short
                            + "}}$",
                            range=ratio_y_range,
                            font_size=text_font_size,
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.15, "bottom": 0.095, "top": 0.975}),
        ),
        output_dir=output_dir,
    )


def _plot_pp_PbPb_comparison_single_panel(
    ax: mpl.axes.Axes,
    axes_ratio: mpl.axes.Axes | list[mpl.axes.Axes],
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_method: str,
    set_zero_to_nan: bool,
    all_methods_on_one_figure: bool,
    event_activity_to_kt_range: Mapping[str, helpers.KtRange],
    models_ratio: Mapping[str, Mapping[str, model_calculations.ModelCalculation]] | None = None,
) -> None:
    # Validation
    if not isinstance(axes_ratio, collections.abc.Iterable):
        axes_ratio = [axes_ratio]

    # Setup
    grooming_styles = plot_style.define_paper_grooming_styles()
    _event_activity_to_color = plot_style.define_paper_event_activity_comparison_styles()

    # Use pp as reference, but only in the range where the others are measured.
    ratio_reference_hist_unselected = hists["pp"].data
    # Determine whether we should plot the ratio in black points and grey uncertainties
    # We only want to do that if we have models for comparison and:
    # 1. Have just pp and one other collision system available.
    #   OR
    # 2. Have multiple other collision systems, but additional ratio axes.
    plot_ratio_black_and_white = (models_ratio and (len(hists) == 2 or len(axes_ratio) > 1))

    axis_ratio_counter = 0
    for _plot_counter, (collision_system, hist) in enumerate(hists.items()):
        # Axes: jet_pt, attr_name
        h = hist.data

        # Select range to display.
        h = full_results_helpers.select_hist_range(h, event_activity_to_kt_range[collision_system])

        # Set 0s to NaN
        if set_zero_to_nan:
            h.errors[h.values == 0] = np.nan
            h.values[h.values == 0] = np.nan


        # Main data points
        kwargs_plot_errorbar = grooming_styles[grooming_method].kwargs_for_plot_errorbar()
        kwargs_plot_errorbar["color"] = _event_activity_to_color[collision_system]
        kwargs_plot_errorbar["markeredgecolor"] = kwargs_plot_errorbar["color"]
        kwargs_plot_errorbar["markerfacecolor"] = "white" if kwargs_plot_errorbar["markerfacecolor"] == "white" else kwargs_plot_errorbar["color"]
        p = ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            label=_event_activity_full_label_map[collision_system],
            # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
            zorder=3 + _plot_counter,
            **kwargs_plot_errorbar,
        )

        # Systematic uncertainty
        kwargs_plot_error_boxes = grooming_styles[grooming_method].kwargs_for_plot_error_boxes()
        kwargs_plot_error_boxes["color"] = p[0].get_color()
        kwargs_plot_error_boxes["zorder"] = 2
        pachyderm.plot.error_boxes(
            ax=ax,
            x_data=h.axes[0].bin_centers,
            y_data=h.values,
            x_errors=h.axes[0].bin_widths / 2,
            y_errors=np.array(
                [
                    h.metadata["y_systematic"]["quadrature"].low,
                    h.metadata["y_systematic"]["quadrature"].high,
                ]
            ),
            **kwargs_plot_error_boxes,
        )

        # Ratio
        # Skip pp because it's not meaningful.
        if collision_system == "pp":
            continue

        # Ensure the ratio is defined over the same range.
        ratio_reference_hist = full_results_helpers.select_hist_range(
            ratio_reference_hist_unselected, event_activity_to_kt_range[collision_system]
        )
        ratio = h / ratio_reference_hist
        # Ratio + statistical error bars
        kwargs_plot_errorbar = grooming_styles[grooming_method].kwargs_for_plot_errorbar()
        kwargs_plot_errorbar["color"] = "black" if plot_ratio_black_and_white else p[0].get_color()
        kwargs_plot_errorbar["markeredgecolor"] = kwargs_plot_errorbar["color"]
        kwargs_plot_errorbar["markerfacecolor"] = "white" if kwargs_plot_errorbar["markerfacecolor"] == "white" else kwargs_plot_errorbar["color"]
        axes_ratio[axis_ratio_counter].errorbar(
            ratio.axes[0].bin_centers,
            ratio.values,
            yerr=ratio.errors,
            xerr=ratio.axes[0].bin_widths / 2,
            # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
            zorder=3 + _plot_counter,
            **kwargs_plot_errorbar,
        )
        # Systematic errors.
        y_relative_error_low = full_results_helpers.relative_error(
            full_results_helpers.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].low),
            full_results_helpers.ErrorInput(
                value=ratio_reference_hist.values,
                error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].low,
            ),
        )
        y_relative_error_high = full_results_helpers.relative_error(
            full_results_helpers.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].high),
            full_results_helpers.ErrorInput(
                value=ratio_reference_hist.values,
                error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].high,
            ),
        )
        # Store the systematic.
        ratio.metadata["y_systematic"]["quadrature"] = full_results_helpers.AsymmetricErrors(
            low=y_relative_error_low * ratio.values,
            high=y_relative_error_high * ratio.values,
        )
        y_systematic = ratio.metadata["y_systematic"]["quadrature"]
        kwargs_plot_error_boxes = grooming_styles[grooming_method].kwargs_for_plot_error_boxes()
        kwargs_plot_error_boxes["color"] = "grey" if plot_ratio_black_and_white else p[0].get_color()
        kwargs_plot_error_boxes["zorder"] = 2
        pachyderm.plot.error_boxes(
            ax=axes_ratio[axis_ratio_counter],
            x_data=ratio.axes[0].bin_centers,
            y_data=ratio.values,
            x_errors=ratio.axes[0].bin_widths / 2,
            y_errors=np.array([y_systematic.low, y_systematic.high]),
            **kwargs_plot_error_boxes,
        )

        # Plot model comparison if available
        for model_name, model_calculation in models_ratio.items():
            model = model_calculation.ratio(event_activity=collision_system).get(grooming_method, None)
            if not model:
                logger.debug(f"{model_calculation.ratio(event_activity=collision_system)}")
                logger.debug(
                    f"Skipping model {model_name}, grooming method: {grooming_method}, {collision_system} because predictions aren't available"
                )
                continue

            # Select the relevant kt range
            model = full_results_helpers.select_hist_range(
                model, event_activity_to_kt_range[collision_system]
            )

            # Fill between
            # NOTE: This is assuming we'll only plot PbPb model colors here, but I think that's a reasonable assumption,
            #       since that's the only models that could compare to the PbPb/pp ratio
            temp_kwargs = retrieve_model_styles(event_activity="PbPb", model_name=model_name)
            temp_kwargs["facecolor"] = temp_kwargs.pop("color")
            # In the case of a single figure, we'll create the handles later
            if not all_methods_on_one_figure:
                temp_kwargs["label"] = model_calculation.label(collision_system="PbPb")
            temp_kwargs.pop("marker")
            axes_ratio[axis_ratio_counter].fill_between(
                model.axes[0].bin_centers,
                model.values - model.errors,
                model.values + model.errors,
                alpha=0.7,
                **temp_kwargs,
            )

        # Advance to the next ratio axis
        # NOTE: Can only advance if is more than one axis. Otherwise, it will go out of bounds.
        if len(axes_ratio) > 1:
            axis_ratio_counter += 1

    # Reference value for ratio
    for ax_ratio in axes_ratio:
        ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=0.9)


def _plot_pp_PbPb_comparison(  # noqa: C901
    hists: Mapping[str, Mapping[str, unfolding_analysis.SingleResult]],
    grooming_methods: list[str],
    set_zero_to_nan: bool,
    all_methods_on_one_figure: bool,
    event_activity_to_kt_range: Mapping[str, Mapping[str, helpers.KtRange]],
    plot_config: pb.PlotConfig,
    output_dir: Path,
    models_ratio: Mapping[str, Mapping[str, model_calculations.ModelCalculation]] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Plot PbPb with systematics compared to pp with systematics for a set of grooming methods."""
    # Validations
    if models_ratio is None:
        models_ratio = {}

    logger.info(f"Plotting pp-PbPb comparison for {grooming_methods}")

    # Setup
    n_ratio_panels = len(hists) - 1 if models_ratio else 1
    height_ratios = [3] + [1] * n_ratio_panels
    if all_methods_on_one_figure:
        n_horizontal_panels = int(np.ceil(len(grooming_methods) / 2))
        fig, all_axes = plt.subplots(
            2 * (1 + n_ratio_panels),
            n_horizontal_panels,
            figsize=(7.5 * n_horizontal_panels, 15),
            gridspec_kw={"height_ratios": height_ratios * 2},
            sharex="col",
            sharey="row",
        )
        ax_pairs = [
            (ax, axes_ratio)
            # NOTE: This is tricky because all_axes is 2x2 here, so stepping by 2 goes to
            #       the next row!
            for ax, *axes_ratio in zip(all_axes[:: 1 + n_ratio_panels].flatten(), all_axes[1:: 1 + n_ratio_panels].flatten())
        ]
    else:
        fig, all_axes = plt.subplots(
            1 + n_ratio_panels,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": height_ratios},
            sharex=True,
        )
        ax, *axes_ratio = all_axes
        # Repeat the pairs
        ax_pairs = [
            (ax, axes_ratio)
            for _ in range(len(grooming_methods))
        ]

    # Loop over grooming methods to plot
    for grooming_method, (ax, axes_ratio) in zip(grooming_methods, ax_pairs):
        _plot_pp_PbPb_comparison_single_panel(
            ax=ax, axes_ratio=axes_ratio,
            hists={
                k: v[grooming_method]
                for k, v in hists.items()
            },
            grooming_method=grooming_method,
            set_zero_to_nan=set_zero_to_nan,
            all_methods_on_one_figure=all_methods_on_one_figure,
            event_activity_to_kt_range={
                k: v[grooming_method]
                for k, v in event_activity_to_kt_range.items()
            },
            models_ratio=models_ratio,
        )

    if all_methods_on_one_figure:
        # We want to manually add the models legend so we can put it where we want (namely, lower left main panel)
        if models_ratio:
            # To do so, we need to create a new legend by hand. We need to do this manually since it's quite complicated.
            # First, we grab an existing legend to ensure that we plot in the same style
            panel_config = plot_config.panels[0]
            legend_config = panel_config.legend
            assert legend_config is not None
            # Next, we define the config for the legend. This way, we'll always have the same settings except for the location
            legend_models = copy.deepcopy(legend_config)
            legend_models.location = "upper right"
            legend_models.anchor= (0.98, 0.98)
            # Make smaller to try to fit...
            legend_models.font_size = round(legend_models.font_size * 0.8)
            legend_models.marker_label_spacing = 0.
            legend_models.label_spacing = 0.1

            # Create handles and labels by hand, using all models
            legend_elements = []
            for model_name, model_calculation in models_ratio.items():
                # NOTE: This is assuming we'll only plot PbPb model colors here, but I think that's a reasonable assumption,
                #       since that's the only models that could compare to the PbPb/pp ratio
                model_kwargs = retrieve_model_styles(event_activity="PbPb", model_name=model_name)
                legend_elements.append(
                    mpl.patches.Patch(
                        facecolor=model_kwargs["color"],
                        label=model_calculation.label(collision_system="PbPb")
                    )
                )

            # Put on the second panel, next to the data labels
            ax_legend = all_axes[::2].flatten()[1]

            # Now that we have the handles, we can apply
            legend_models.apply(
                ax=ax_legend,
                legend_handles=legend_elements,
            )

        # Labeling and presentation
        plot_config.apply(fig=fig, axes=list(all_axes.flatten()))

        # A few additional tweaks.
        for _ax in all_axes[::2].flatten():
            _ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
    else:
        ax_ratio_legend_config = None
        if models_ratio:
            # Grab the axes which has the legend config.
            ratio_axes_with_legend_config = [ax for ax, panel_config in zip(axes_ratio, plot_config.panels[1:]) if panel_config.legend is not None]
            if len(ratio_axes_with_legend_config) != 1:
                msg = f"Expected exactly one ratio axis to have a legend, got {len(ratio_axes_with_legend_config)}"
                raise ValueError(msg)
            ax_ratio_legend = ratio_axes_with_legend_config[0]

            # Labeling and presentation
            ax_ratio_handles, ax_ratio_labels = ax_ratio_legend.get_legend_handles_labels()
            #logger.info(f"{len(ax_ratio_handles)=}")
            #logger.info(f"{len(ax_ratio_labels)=}, {ax_ratio_labels=}")
            if models_ratio and len(ax_ratio_handles) % 2 == 1:
                #logger.info("Handling manually")
                # Pop out legend handler so that it skips due the plot config and
                # and we can handle it manually
                ax_ratio_legend_config = plot_config.panels[1].legend
                plot_config.panels[1].legend = None
                insert_position = round((len(ax_ratio_handles) + 1)/2)
                ax_ratio_handles.insert(insert_position, ax_ratio_legend.plot([], [], color=(0, 0, 0, 0), label=" ")[0])
                ax_ratio_labels.insert(insert_position, "")
                #ax_ratio_handles.insert(insert_position, ax_ratio_handles[0])
                #ax_ratio_labels.insert(insert_position, ax_ratio_labels[0])

        plot_config.apply(fig=fig, axes=[ax, *axes_ratio])
        # A few additional tweaks.
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
        # Need a manual hack here since the range has gotten so big with the models
        if models_ratio and grooming_method == "soft_drop_z_cut_04":
            ax_ratio_legend.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
        if ax_ratio_legend_config:
            ax_ratio_legend_config.apply(
                ax=ax_ratio_legend,
                legend_handles=ax_ratio_handles,
                legend_labels=ax_ratio_labels,
            )

    filename = f"{plot_config.name}"
    if len(grooming_methods) == 1:
        filename += f"_{grooming_methods[0]}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_pp_PbPb_comparison(
    hists: Mapping[str, Mapping[str, unfolding_analysis.SingleResult]],
    grooming_methods: list[str],
    output_dir: Path,
    event_activity_to_kt_range: Mapping[str, helpers.KtRange | Mapping[str, helpers.KtRange]],
    kt_display_range: tuple[float, float] = (1.5, 15),
    jet_R_str: str = "R04",
    alice_status: str = "work_in_progress",
    text_font_size: int = 31,
    models_ratio: Mapping[str, Mapping[str, binned_data.BinnedData]] | None = None,
    additional_label: str = "",
) -> None:
    """Compare pp and PbPb results with ratio."""
    # Validation
    for ev, kt_range in event_activity_to_kt_range.items():
        if isinstance(kt_range, helpers.KtRange):
            event_activity_to_kt_range[ev] = {grooming_method: kt_range for grooming_method in grooming_methods}

    # Setup
    jet_pt_bin = next(iter(next(iter(hists.values())).values())).ranges[0]
    grooming_styles = plot_style.define_paper_grooming_styles()

    for grooming_method in grooming_methods:
        style = grooming_styles[grooming_method]

        text = plot_style.label_to_display_string["ALICE"][alice_status]
        # Since the final text is short, we can merge onto one line
        if alice_status != "final":
            text += "\n"
        else:
            text += " "
        text += plot_style.label_to_display_string["collision_system"]["pp_PbPb_5TeV"]
        text += "\n" + plot_style.label_to_display_string["jets"]["general"]
        text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
        text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"

        name = "unfolded_kt_pp_PbPb"
        if additional_label:
            name += f"_{additional_label}"
        if models_ratio:
            name += "_models"
        name += f"_comparison_{jet_R_str}"

        _ratio_range = (0.3, 1.7)
        if "central" in hists and models_ratio:
            _ratio_range = (0.1, 1.9)
        if any("z_cut_04" in m for m in grooming_methods):
            _ratio_range = (-0.2, 2.2) if models_ratio else (0.1, 1.9)

        model_legend_config = None
        if models_ratio:
            model_legend_config = pb.LegendConfig(location="lower left", font_size=22, anchor=(0.01, 0.02), ncol=2, marker_label_spacing=0.05, label_spacing=0.1, handle_height=1.3, column_spacing=0.30)

        _plot_pp_PbPb_comparison(
            hists=hists,
            models_ratio=models_ratio,
            grooming_methods=[grooming_method],
            set_zero_to_nan=False,
            all_methods_on_one_figure=False,
            event_activity_to_kt_range=event_activity_to_kt_range,
            plot_config=pb.PlotConfig(
                name=name,
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$",
                                log=True,
                                #range=(7e-3, 1),
                                range=(4e-3, 1),
                                font_size=text_font_size,
                            ),
                        ],
                        text=[
                            pb.TextConfig(x=0.98, y=0.98, text=text, font_size=text_font_size),
                            # Add the grooming label in a separate location in the bottom left
                            # Otherwise, it will overlap with the data
                            pb.TextConfig(x=0.02, y=0.02, text=style.label, font_size=text_font_size),
                        ],
                        legend=pb.LegendConfig(location="lower left", font_size=text_font_size, anchor=(0.0, 0.10), marker_label_spacing=-0.2),
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T,g}}\:(\text{GeV}/c)$", range=kt_display_range, font_size=text_font_size),
                            pb.AxisConfig("y", label=r"$\frac{\text{Pb--Pb}}{\text{pp}}$",
                                        range=_ratio_range,
                                        # Make the label a bit bigger since it's stack on top
                                        font_size=text_font_size * 1.05
                                        ),
                        ],
                        legend=model_legend_config,
                    ),
                ],
                figure=pb.Figure(edge_padding={"left": 0.1525, "bottom": 0.095, "top": 0.975}),
            ),
            output_dir=output_dir,
        )


def plot_pp_PbPb_comparison_single_figure(
    hists: Mapping[str, Mapping[str, unfolding_analysis.SingleResult]],
    grooming_methods: list[str],
    output_dir: Path,
    event_activity_to_kt_range: Mapping[str, helpers.KtRange | Mapping[str, helpers.KtRange]],
    kt_display_range: tuple[float, float] = (1.5, 15),
    jet_R_str: str = "R04",
    alice_status: str = "work_in_progress",
    text_font_size: int = 31,
    models_ratio: Mapping[str, Mapping[str, binned_data.BinnedData]] | None = None,
    additional_label: str = "",
) -> None:
    """Compare pp and PbPb results with ratio."""
    # Validation
    for ev, kt_range in event_activity_to_kt_range.items():
        if isinstance(kt_range, helpers.KtRange):
            event_activity_to_kt_range[ev] = {grooming_method: kt_range for grooming_method in grooming_methods}
    # Setup
    jet_pt_bin = next(iter(next(iter(hists.values())).values())).ranges[0]

    name = "unfolded_kt_pp_PbPb"
    if additional_label:
        name += f"_{additional_label}"
    if models_ratio:
        name += "_models"
    name += f"_comparison_{jet_R_str}"
    name += "_one_figure"

    _ratio_range = (0.3, 1.7)
    if "central" in hists and models_ratio:
        _ratio_range = (0.1, 1.9)

    text = plot_style.label_to_display_string["ALICE"][alice_status]
    # Since the final text is short, we can merge onto one line
    if alice_status != "final":
        text += "\n"
    else:
        text += " "
    text += plot_style.label_to_display_string["collision_system"]["pp_PbPb_5TeV"]
    text += "\n" + plot_style.label_to_display_string["jets"]["general"]
    text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"

    # Setup panels
    # We need to handle this carefully, so we do it slowly, and step-by-step
    # NOTE: The deepcopy calls are critical - otherwise, we may accidentally modify one config
    #       when we modify another.
    n_horizontal_panels = int(np.ceil(len(grooming_methods) / 2))
    grooming_styles = plot_style.define_paper_grooming_styles()
    panels: list[pb.Panel] = []
    main_panel_standard = pb.Panel(
        axes=[
            pb.AxisConfig(
                "y",
                log=True,
                #range=(7e-3, 1),
                range=(4e-3, 1),
                font_size=text_font_size,
            ),
        ],
        text=[
            # Add the grooming label in a separate location in the bottom left
            # Otherwise, it will overlap with the data
            pb.TextConfig(x=0.02, y=0.02, text="", font_size=text_font_size),
        ],
    )
    main_panel_left = copy.deepcopy(main_panel_standard)
    main_panel_left.axes[0].label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$"
    # Also add the legend in upper right of the upper left panel
    main_panel_left.legend = pb.LegendConfig(location="upper right", font_size=text_font_size, anchor=(0.98, 0.98), marker_label_spacing=-0.2)
    panels.append(main_panel_left)
    # NOTE: We can't simply copy the list with * 2 since that would do a simple copy of the object
    #       (defeating the purpose of the deepcopy).
    panels.extend([copy.deepcopy(main_panel_standard) for _ in range(n_horizontal_panels - 2) ])
    # Full ALICE label in right panel
    main_panel_upper_right = copy.deepcopy(main_panel_standard)
    main_panel_upper_right.text.append(pb.TextConfig(x=0.98, y=0.98, text=text, font_size=round(text_font_size * 0.9)))
    panels.append(main_panel_upper_right)
    # Assign grooming method labels
    for p, grooming_method in zip(panels, grooming_methods):
        p.text[0].text = grooming_styles[grooming_method].label

    # Next, onto ratio panels
    ratio_panel_mid_standard = pb.Panel(
        axes=[
            pb.AxisConfig(
                "x",
                #label=r"$k_{\text{T,g}}\:(\text{GeV}/c)$",
                range=kt_display_range,
                font_size=text_font_size,
            ),
            pb.AxisConfig(
                "y",
                range=_ratio_range,
                # Make the label a bit bigger since it's stack in a fraction
                font_size=text_font_size * 1.05
            ),
        ],
    )
    # Mid left needs the label
    ratio_panel_mid_left = copy.deepcopy(ratio_panel_mid_standard)
    ratio_panel_mid_left.axes[1].label = r"$\frac{\text{Pb--Pb}}{\text{pp}}$"
    panels.append(ratio_panel_mid_left)
    # Fill out the last ones as standard ratios
    panels.extend([copy.deepcopy(ratio_panel_mid_standard) for _ in range(n_horizontal_panels - 1)])

    # Now onto the next set of main panels
    panels.append(copy.deepcopy(main_panel_left))
    # Make sure we don't put the legend there twice
    panels[-1].legend = None
    # Fill out the last ones as standard
    panels.extend([copy.deepcopy(main_panel_standard) for _ in range(n_horizontal_panels - 1)])
    # Assign grooming method labels
    for p, grooming_method in zip(panels[-1 * n_horizontal_panels:], grooming_methods[-1 * n_horizontal_panels:]):
        p.text[0].text = grooming_styles[grooming_method].label

    # Finish with the rest of the ratios
    ratio_panel_bottom_standard = copy.deepcopy(ratio_panel_mid_standard)
    ratio_panel_bottom_standard.axes[0].label = r"$k_{\text{T,g}}\:(\text{GeV}/c)$"
    ratio_panel_bottom_left = copy.deepcopy(ratio_panel_bottom_standard)
    ratio_panel_bottom_left.axes[1].label = r"$\frac{\text{Pb--Pb}}{\text{pp}}$"
    panels.append(ratio_panel_bottom_left)
    # Fill out the last ones as standard
    panels.extend([copy.deepcopy(ratio_panel_bottom_standard)] * (n_horizontal_panels - 1))

    _plot_pp_PbPb_comparison(
        hists=hists,
        models_ratio=models_ratio,
        grooming_methods=grooming_methods,
        set_zero_to_nan=False,
        all_methods_on_one_figure=True,
        event_activity_to_kt_range=event_activity_to_kt_range,
        plot_config=pb.PlotConfig(
            name=name,
            panels=panels,
            figure=pb.Figure(edge_padding={"left": 0.055, "bottom": 0.07, "top": 0.98, "right": 0.98}),
        ),
        output_dir=output_dir,
    )


def plot_pp_PbPb_comparison_with_multiple_model_ratios(
    hists: Mapping[str, Mapping[str, unfolding_analysis.SingleResult]],
    grooming_methods: list[str],
    output_dir: Path,
    event_activity_to_kt_range: Mapping[str, helpers.KtRange | Mapping[str, helpers.KtRange]],
    kt_display_range: tuple[float, float] = (1.5, 15),
    jet_R_str: str = "R02",
    alice_status: str = "work_in_progress",
    text_font_size: int = 31,
    models_ratio: Mapping[str, Mapping[str, binned_data.BinnedData]] | None = None,
    additional_label: str = "",
) -> None:
    """Compare pp and PbPb results with ratio."""
    # Validation
    for ev, kt_range in event_activity_to_kt_range.items():
        if isinstance(kt_range, helpers.KtRange):
            event_activity_to_kt_range[ev] = {grooming_method: kt_range for grooming_method in grooming_methods}

    # Setup
    jet_pt_bin = next(iter(next(iter(hists.values())).values())).ranges[0]
    grooming_styles = plot_style.define_paper_grooming_styles()
    non_pp_collision_system = [k for k in hists if k != "pp"]
    assert len(non_pp_collision_system) == 2, "Need to pass both semi-central and central!"

    for grooming_method in grooming_methods:
        style = grooming_styles[grooming_method]

        text = plot_style.label_to_display_string["ALICE"][alice_status]
        # Since the final text is short, we can merge onto one line
        if alice_status != "final":
            text += "\n"
        else:
            text += " "
        text += plot_style.label_to_display_string["collision_system"]["pp_PbPb_5TeV"]
        text += "\n" + plot_style.label_to_display_string["jets"]["general"]
        text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
        text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"

        name = "unfolded_kt_pp_PbPb"
        if additional_label:
            name += f"_{additional_label}"
        if models_ratio:
            name += "_models"
        name += f"_comparison_{jet_R_str}"

        _ratio_range = (0.3, 1.7)
        if "central" in hists and models_ratio:
            _ratio_range = (0.1, 1.9)
        if any("z_cut_04" in m for m in grooming_methods):
            _ratio_range = (-0.2, 2.2) if models_ratio else (0.1, 1.9)

        model_legend_config = None
        if models_ratio:
            model_legend_config = pb.LegendConfig(location="lower left", font_size=22, anchor=(0.01, 0.02), ncol=2, marker_label_spacing=0.05, label_spacing=0.1, handle_height=1.3, column_spacing=0.30)

        _plot_pp_PbPb_comparison(
            hists=hists,
            models_ratio=models_ratio,
            grooming_methods=[grooming_method],
            set_zero_to_nan=False,
            all_methods_on_one_figure=False,
            event_activity_to_kt_range=event_activity_to_kt_range,
            plot_config=pb.PlotConfig(
                name=name,
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$",
                                log=True,
                                #range=(7e-3, 1),
                                range=(4e-3, 1),
                                font_size=text_font_size,
                            ),
                        ],
                        text=[
                            pb.TextConfig(x=0.98, y=0.98, text=text, font_size=text_font_size),
                            # Add the grooming label in a separate location in the bottom left
                            # Otherwise, it will overlap with the data
                            pb.TextConfig(x=0.02, y=0.02, text=style.label, font_size=text_font_size),
                        ],
                        legend=pb.LegendConfig(location="lower left", font_size=text_font_size, anchor=(0.0, 0.10), marker_label_spacing=-0.2),
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=fr"$\frac{{\text{{{_event_activity_short_label_map[non_pp_collision_system[0]]}}}}}{{\text{{pp}}}}$",
                                        range=_ratio_range,
                                        # Make the label a bit bigger since it's stack on top
                                        font_size=text_font_size * 1.05
                                        ),
                        ],
                        legend=model_legend_config,
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T,g}}\:(\text{GeV}/c)$", range=kt_display_range, font_size=text_font_size),
                            pb.AxisConfig("y", label=fr"$\frac{{\text{{{_event_activity_short_label_map[non_pp_collision_system[1]]}}}}}{{\text{{pp}}}}$",
                                        range=_ratio_range,
                                        # Make the label a bit bigger since it's stack on top
                                        font_size=text_font_size * 1.05
                                        ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding={"left": 0.1525, "bottom": 0.095, "top": 0.975}),
            ),
            output_dir=output_dir,
        )


def _plot_pp_PbPb_only_ratios(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_method: str,
    set_zero_to_nan: bool,
    all_methods_on_one_figure: bool,
    event_activity_to_kt_range: Mapping[str, helpers.KtRange],
    fit_parameters: Mapping[str, Mapping[str, Mapping[str, float]]],
    fit_QA_plot: bool,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    models_ratio: Mapping[str, Mapping[str, model_calculations.ModelCalculation]],
    model_labels_on_axes: list[str],
) -> None:
    """Plot model/data ratios for all provided collision systems."""
    # Setup
    grooming_styles = plot_style.define_paper_grooming_styles()

    fig, axes = plt.subplots(
        len(hists),
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [1] * len(hists)},
        sharex=True,
    )

    # TODO: Fill these in or pass them...

    for _plot_counter, ((collision_system, hist), ax) in enumerate(zip(hists.items(), axes)):
        # Axes: jet_pt, attr_name
        h: binned_data.BinnedData = hist[grooming_method].data

        # Select range to display.
        h = full_results_helpers.select_hist_range(h, event_activity_to_kt_range[collision_system][grooming_method])

        # Set 0s to NaN
        if set_zero_to_nan:
            h.errors[h.values == 0] = np.nan
            h.values[h.values == 0] = np.nan

        if fit_parameters:
            h_for_fit = h
            # This adds back in points since the phase space will be too restricted otherwise...
            # TODO: Try this out for a smaller value, like 2.0
            #if h.axes[0].bin_edges[0] >= 3.0:
            if h.axes[0].bin_edges[0] >= 2.0:
                h_for_fit = full_results_helpers.select_hist_range(
                    hist[grooming_method].data,
                    #helpers.KtRange(2.0, event_activity_to_kt_range[collision_system][grooming_method].max)
                    helpers.KtRange(1.5, event_activity_to_kt_range[collision_system][grooming_method].max)
                )
            from jet_substructure.analysis import fit_paper
            disable_y_scale = False
            initial_arguments = {
                "amplitude": -1,
                "shift": -1,
                "intercept": 1,
                "power_law": 3.,
                "power_law_amp": 1.,
            }
            fit_result = fit_paper.fit_spectra(
                x0=fit_parameters[collision_system][grooming_method]["x0"],
                tanh_transition_scale=fit_parameters[collision_system][grooming_method]["tanh_transition_scale"],
                h=h_for_fit,
                initial_arguments=initial_arguments,
                disable_y_scale=disable_y_scale,
            )
            reference_values = fit_result(h.axes[0].bin_centers)

            # Save parameters
            spectra_fit_parameters_filename = output_dir / "spectra_fit" / f"{collision_system}_{grooming_method}.yaml"
            spectra_fit_parameters_filename.parent.mkdir(parents=True, exist_ok=True)
            fit_paper.write_fit_result(
                fit_result=fit_result,
                x0=fit_parameters[collision_system][grooming_method]["x0"],
                tanh_transition_scale=fit_parameters[collision_system][grooming_method]["tanh_transition_scale"],
                output_path=spectra_fit_parameters_filename,
            )

            # Fit QA
            if fit_QA_plot:
                fig_QA, (ax_QA, ax_ratio_QA) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
                # Data
                ax_QA.errorbar(
                    h_for_fit.axes[0].bin_centers,
                    h_for_fit.values,
                    yerr=h_for_fit.errors,
                    xerr=h_for_fit.axes[0].bin_widths / 2,
                    color="black",
                    marker=grooming_styles[grooming_method].marker,
                    markersize=11,
                    linestyle="",
                    linewidth=3,
                    zorder=6,
                    label="data (for fit)",
                )
                # Fit
                fit_paper.fit_and_plot(
                    x0=fit_parameters[collision_system][grooming_method]["x0"],
                    tanh_transition_scale=fit_parameters[collision_system][grooming_method]["tanh_transition_scale"],
                    h=h_for_fit,
                    disable_y_scale=disable_y_scale,
                    initial_arguments=initial_arguments,
                    x_for_plotting=np.linspace(h_for_fit.axes[0].bin_centers[0], h_for_fit.axes[0].bin_centers[-1], num=100, endpoint=True),
                    plot_label=f"{collision_system}, {grooming_method}",
                    plot_components=True,
                    ax=ax_QA,
                    ax_ratio=ax_ratio_QA,
                    fit_result=fit_result,
                )
                ax_ratio_QA.set_xlabel(r"$k_{\text{T,g}}\:(\text{GeV}/c)$")
                ax_ratio_QA.set_ylabel("Fit/data")
                ax_QA.set_yscale("log")
                #ax_QA.set_ylim([1e-3, 1])
                ax_QA.legend()
                ax_ratio_QA.legend(loc="upper left")
                ax_ratio_QA.set_ylim([0.75, 1.5])
                fig_QA.tight_layout()
                fig_QA.savefig(output_dir / "spectra_fit" / f"spectra_fit_QA_{collision_system}_{grooming_method}.pdf")

                plt.close(fig_QA)
        else:
            # If not fitting, just use the measured data
            reference_values = h.values
        #logger.info(f"{reference_values=}")

        # Next, draw the data and uncertainties at one as black and grey boxes
        # Ratio + statistical error bars at one
        hist_values = h.values / reference_values if fit_parameters else np.ones_like(h.values)
        ax.errorbar(
            h.axes[0].bin_centers,
            hist_values,
            yerr=h.errors / h.values * hist_values,
            xerr=h.axes[0].bin_widths / 2,
            color="black",
            marker=grooming_styles[grooming_method].marker,
            markersize=11,
            linestyle="",
            linewidth=3,
            zorder=6,
            #label=_event_activity_short_label_map[collision_system],
        )
        pachyderm.plot.error_boxes(
            ax=ax,
            x_data=h.axes[0].bin_centers,
            y_data=hist_values,
            x_errors=h.axes[0].bin_widths / 2,
            y_errors=np.array(
                [
                    h.metadata["y_systematic"]["quadrature"].low / h.values * hist_values,
                    h.metadata["y_systematic"]["quadrature"].high / h.values * hist_values,
                ]
            ),
            color="black",
            linewidth=0,
            alpha=0.3,
            zorder=5.5,
        )

        # Plot model comparisons
        for model_name, model_calculation in models_ratio.items():
            model = model_calculation.spectra(event_activity=collision_system).get(grooming_method, None)
            if not model:
                logger.debug(f"{model_calculation.ratio(event_activity=collision_system)}")
                logger.debug(
                    f"Skipping model {model_name}, grooming method: {grooming_method}, {collision_system} because predictions aren't available"
                )
                continue

            # Select the relevant kt range
            model = full_results_helpers.select_hist_range(
                model, event_activity_to_kt_range[collision_system][grooming_method]
            )

            # Further setup
            # NOTE: If we naively construct the ratio here by just dividing the model by the data,
            #       then the errors stored in the ratio aren't what we want since they convolve the
            #       model uncertainties with the data uncertainties. So want to calculate the ratio
            #       using a hist without the data uncertainties.
            # NOTE: We define this here (ie. early) so we can decide what to rebin (which) we need to
            #       know before we plot the model
            h_without_uncertainties = binned_data.BinnedData(
                axes=[h.axes[0].bin_edges],
                # NOTE: The `np.array` is really important here because we need to make a copy!
                #       Otherwise, we modify the underlying values, and everything gets fucked up in
                #       future loop iterations.
                values=np.array(reference_values, copy=True) if fit_parameters else np.array(h.values, copy=True),
                variances=np.zeros_like(h.values),
            )

            # Check that binning matches up. If it doesn't attempt to rebin
            if h_without_uncertainties.axes[0].bin_edges.shape != model.axes[0].bin_edges.shape or \
                not np.allclose(h_without_uncertainties.axes[0].bin_edges, model.axes[0].bin_edges):
                # Rebin according to the data which we are supposed to be plotting
                # NOTE: We take as a proxy that whichever hist has more bins is the one that needs to be rebinned.
                #       We can't just assume that the model is more finely binned because some (eg. Caucal) is not.
                if h_without_uncertainties.axes[0].bin_edges.shape[0] > model.axes[0].bin_edges.shape[0]:
                    h_without_uncertainties = full_results_helpers.rebin_bin_width_scaled_hist(
                        h_to_rebin=h_without_uncertainties,
                        h_target_axis=model.axes[0],
                        # This is okay since the data is explicitly constructed without systematic systematic uncertainties.
                        okay_for_systematic_not_to_exist=True,
                    )
                else:
                    model = full_results_helpers.rebin_bin_width_scaled_hist(
                        h_to_rebin=model,
                        h_target_axis=h_without_uncertainties.axes[0],
                        # This is okay since the model doesn't usually have a systematic uncertainty.
                        okay_for_systematic_not_to_exist=True,
                    )

            # Ratio
            ratio = model / h_without_uncertainties

            # We need to propagate the systematic uncertainty manually since the data is constructed not to have
            # uncertainties that we would usually propagate with
            if "y_systematic" in model.metadata:
                y_relative_error_low = full_results_helpers.relative_error(
                    full_results_helpers.ErrorInput(value=model.values, error=model.metadata["y_systematic"]["quadrature"].low),
                )
                y_relative_error_high = full_results_helpers.relative_error(
                    full_results_helpers.ErrorInput(value=model.values, error=model.metadata["y_systematic"]["quadrature"].high),
                )
                ratio_systematic = full_results_helpers.AsymmetricErrors(
                    low=y_relative_error_low * ratio.values,
                    high=y_relative_error_high * ratio.values,
                )
                ratio.metadata["y_systematic"]["quadrature"] = ratio_systematic

            # Finally, plot the band in the ratio
            # NOTE: This is assuming we'll only plot PbPb model colors here, but I think that's a reasonable assumption,
            #       since that's the only models that could compare to the PbPb/pp ratio
            temp_kwargs = retrieve_model_styles(event_activity="PbPb", model_name=model_name)
            # Will fill label in legend manually, so no need to do it here...
            temp_kwargs["label"] = model_calculation.label(collision_system=collision_system) if not all_methods_on_one_figure else None
            # Need to pop for fill_between since these aren't valid args
            temp_kwargs.pop("marker")
            temp_kwargs.pop("markerfacecolor", None)
            temp_kwargs.pop("markeredgewidth", None)
            # And switch to the proper color
            temp_kwargs["facecolor"] = temp_kwargs.pop("color")
            lower_error, upper_error = _determine_uncertainty_lower_upper_for_model(model=ratio)
            ax.fill_between(
                ratio.axes[0].bin_centers,
                ratio.values - lower_error,
                ratio.values + upper_error,
                zorder=5,
                alpha=0.75,
                **temp_kwargs,
            )

    # Reference value for ratio
    for _plot_counter, (ax_ratio, panel_config) in enumerate(zip(axes, plot_config.panels)):
        ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=0.9)

        # Update the data legends to show both the marker and box
        # NOTE: As of 2023 Jun 8, it wasn't worth the effort. The box was too small for the marker, etc.
        #       In general, these things are often quite tough with mpl... :-(

        if not model_labels_on_axes[_plot_counter]:
            continue

        # Setup
        legend_config = panel_config.legend
        assert legend_config is not None

        # Begin with the data legend...
        #handles, labels = ax_ratio.get_legend_handles_labels()

        #import matplotlib.lines as mlines
        #handles = [
        #    (
        #        copy.deepcopy(handle),
        #        mpl.patches.Patch(
        #            facecolor=p_boxes_data.get_facecolor()[0],
        #            alpha=p_boxes_data.get_alpha(),
        #        ),
        #    )
        #    for handle in handles
        #]
        #logger.info(f"{handles=}")
        #legend_object = legend_config.apply(
        #    ax=ax_ratio,
        #    legend_handles=handles,
        #    legend_labels=labels
        #)

        ######
        # Create handles and labels by hand, using all models
        ######
        legend_config = copy.deepcopy(legend_config)
        # NOTE: We only have to change these settings **if** we're passing a data legend.
        #       As of 2023 June 8, we're not doing that, so we have these lines commented out.
        #legend_config.location = "lower left"
        #legend_config.ncol = 2
        #legend_config.anchor = (0.02, 0.02)

        model_legend_elements = []
        for model_name in model_labels_on_axes[_plot_counter]:
            model_calculation = models_ratio[model_name]

            # NOTE: This is assuming we'll only plot PbPb model colors here, but I think that's a reasonable assumption,
            #       since that's the only models that could compare to the PbPb/pp ratio
            model_kwargs = retrieve_model_styles(event_activity="PbPb", model_name=model_name)
            model_legend_elements.append(
                mpl.patches.Patch(
                    facecolor=model_kwargs["color"],
                    label=model_calculation.label(collision_system=collision_system)
                )
            )
        model_legend_object = legend_config.apply(
            ax=ax_ratio,
            legend_handles=model_legend_elements,
        )
        ax_ratio.add_artist(model_legend_object)

        # Commented - see above.
        ## Add the legend
        #ax_ratio.add_artist(legend_object)
        # And since we've manually added the legend and we don't want plot_config to interfere, we set it to None.
        panel_config.legend = None

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=axes)
    # A few additional tweaks.
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}_{grooming_method}.pdf")
    plt.close(fig)


def plot_pp_PbPb_only_model_data_ratios(
    hists: Mapping[str, Mapping[str, unfolding_analysis.SingleResult]],
    grooming_methods: list[str],
    output_dir: Path,
    event_activity_to_kt_range: Mapping[str, helpers.KtRange | Mapping[str, helpers.KtRange]],
    models_ratio: Mapping[str, Mapping[str, binned_data.BinnedData]],
    model_labels_on_axes: list[list[str]],
    kt_display_range: tuple[float, float] = (1.5, 15),
    jet_R_str: str = "R04",
    alice_status: str = "work_in_progress",
    text_font_size: int = 31,
    additional_label: str = "",
    logy: bool = False,
    fit_parameters: Mapping[str, Mapping[str, float | Mapping[str, float]]] = {},
    fit_QA_plot: bool = False,
) -> None:
    """Compare pp and PbPb results with ratio."""
    # Validation
    for ev, kt_range in event_activity_to_kt_range.items():
        if isinstance(kt_range, helpers.KtRange):
            event_activity_to_kt_range[ev] = {grooming_method: kt_range for grooming_method in grooming_methods}
    # NOTE: Need the deep copy because we modify the dict in place
    fit_parameters = copy.deepcopy(fit_parameters)
    for ev, parameters in fit_parameters.items():
        # Proxy for whether just the values are provided.
        if "x0" in parameters:
            fit_parameters[ev] = {grooming_method: parameters for grooming_method in grooming_methods}
    logger.info(f"{fit_parameters=}")

    # NOTE: This ordering is important to get the panels right!
    #       We want pp first, and then the order for the rest is determined by the order in which the hists are passed
    assert next(iter(list(hists.keys()))) == "pp"

    # Setup
    jet_pt_bin = next(iter(next(iter(hists.values())).values())).ranges[0]
    grooming_styles = plot_style.define_paper_grooming_styles()
    event_activity_order = iter(list(hists))
    model_label_order = iter(model_labels_on_axes)

    for grooming_method in grooming_methods:
        logger.info(f"Plotting all ratios for {grooming_method}")
        style = grooming_styles[grooming_method]

        name = "unfolded_kt_pp_PbPb"
        if fit_parameters:
            name += "_spectra_fit"
        if additional_label:
            name += f"_{additional_label}"
        name += f"_model_data_ratios_{jet_R_str}"

        _ratio_range = (0.3, 1.7)
        if "central" in hists and models_ratio:
            _ratio_range = (0.35, 1.6)
        if any("z_cut_04" in m for m in grooming_methods):
            _ratio_range = (-0.2, 2.2) if models_ratio else (0.1, 1.9)
        if logy:
            _ratio_range = (0.45, 1.8)
            # If I move the collision system, I could try to make something like the below work
            #_ratio_range = (0.55, 1.65)

        # Define panels
        panels = []
        standard_y_axis = pb.AxisConfig(
            "y",
            label=r"$\frac{\text{Model}}{\text{Data}}$" if not fit_parameters else r"$\frac{\text{Spectra}}{\text{Param.}}$",
            range=_ratio_range,
            # Make the label a bit bigger since it's stack on top
            font_size=text_font_size * 1.05,
            log=logy,
        )
        #standard_data_legend = pb.LegendConfig(location="lower right", font_size=text_font_size, anchor=(0.98, 0.02), marker_label_spacing=-0.2)
        standard_model_legend = pb.LegendConfig(
            location="lower left",
            font_size=round(text_font_size * 0.8),
            anchor=(0.02, 0.02),
            #ncol=2,
            marker_label_spacing=0.05,
            label_spacing=0.1,
            handle_height=1.3,
            column_spacing=0.30,
        )
        # pp - top panel
        # ALICE pp, PbPb 5.02 TeV
        text = plot_style.label_to_display_string["ALICE"][alice_status]
        # Since the final text is short, we can merge onto one line
        if alice_status != "final":
            text += "\n"
        else:
            text += " "
        text += plot_style.label_to_display_string["collision_system"]["pp_PbPb_5TeV"]
        panels.append(
            pb.Panel(
                axes=[
                    copy.deepcopy(standard_y_axis)
                ],
                text=[
                    pb.TextConfig(x=0.98, y=0.98, text=text, font_size=text_font_size),
                    # Add the grooming label in a separate location in the bottom right
                    #pb.TextConfig(x=0.98, y=0.02, text=style.label, font_size=text_font_size),
                    # And the collision system
                    pb.TextConfig(x=0.97, y=0.04, text=_event_activity_full_label_map[next(event_activity_order)], font_size=text_font_size),
                ],
                #legend=copy.deepcopy(standard_data_legend),
                # Only provide if there are model entries for this panel.
                legend=copy.deepcopy(standard_model_legend) if next(model_label_order) else None,
            )
        )
        # Middle panel
        text = plot_style.label_to_display_string["jets"]["general"]
        text += " " + plot_style.label_to_display_string["jets"][jet_R_str]
        #text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
        text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
        panels.append(
            pb.Panel(
                axes=[
                    copy.deepcopy(standard_y_axis)
                ],
                text=[
                    pb.TextConfig(x=0.98, y=0.98, text=text, font_size=text_font_size),
                    # Add the grooming label in a separate location in the bottom right
                    #pb.TextConfig(x=0.98, y=0.02, text=style.label, font_size=text_font_size),
                    # And the collision system
                    pb.TextConfig(x=0.98, y=0.02, text=_event_activity_full_label_map[next(event_activity_order)], font_size=text_font_size),
                ],
                #legend=copy.deepcopy(standard_data_legend),
                # Only provide if there are model entries for this panel.
                legend=copy.deepcopy(standard_model_legend) if next(model_label_order) else None,
            )
        )
        # Bottom panel
        panels.append(
            pb.Panel(
                axes=[
                    copy.deepcopy(standard_y_axis),
                    pb.AxisConfig("x", label=r"$k_{\text{T,g}}\:(\text{GeV}/c)$", range=kt_display_range, font_size=text_font_size),
                ],
                text=[
                    # Add the grooming label in a separate location in the upper right
                    pb.TextConfig(x=0.95, y=0.97, text=style.label, font_size=text_font_size),
                    # And the collision system
                    pb.TextConfig(x=0.98, y=0.02, text=_event_activity_full_label_map[next(event_activity_order)], font_size=text_font_size),
                ],
                #legend=copy.deepcopy(standard_data_legend),
                # Only provide if there are model entries for this panel.
                legend=copy.deepcopy(standard_model_legend) if next(model_label_order) else None,
            )
        )

        _plot_pp_PbPb_only_ratios(
            hists=hists,
            models_ratio=models_ratio,
            model_labels_on_axes=model_labels_on_axes,
            grooming_method=grooming_method,
            set_zero_to_nan=False,
            all_methods_on_one_figure=False,
            event_activity_to_kt_range=event_activity_to_kt_range,
            fit_parameters=fit_parameters,
            fit_QA_plot=fit_QA_plot,
            plot_config=pb.PlotConfig(
                name=name,
                panels=panels,
                figure=pb.Figure(edge_padding={"left": 0.125, "bottom": 0.095, "top": 0.975}),
            ),
            output_dir=output_dir,
        )
