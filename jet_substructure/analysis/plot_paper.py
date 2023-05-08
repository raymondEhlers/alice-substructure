""" Paper plots

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import cycler
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
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"  # noqa: ISC003
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


def _determine_uncertainty_limits_for_model(model: binned_data.BinnedData) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    models: Mapping[str, tuple[model_calculations.ModelCalculation, Mapping[str, binned_data.BinnedData]]],
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
        models: Mapping from model name to model calculation and binned data.
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
    model_styles = plot_style.define_paper_model_styles()

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
        ax_pairs = [
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

        # Main data points
        p = ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            marker=grooming_styles[grooming_method].marker,
            markersize=11,
            linestyle="",
            linewidth=3,
            label=grooming_styles[grooming_method].label_short,
            color=grooming_styles[grooming_method].color,
            # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
            zorder=3 + _plot_counter + grooming_styles[grooming_method].zorder,
        )

        # Systematic uncertainty
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
            color=p[0].get_color(),
            linewidth=0,
            alpha=0.3,
        )

        # Next, draw the data and uncertainties at one as black and grey boxes
        # Ratio + statistical error bars at one
        ax_ratio.errorbar(
            h.axes[0].bin_centers,
            np.ones_like(h.axes[0].bin_centers),
            yerr=h.errors / h.values,
            xerr=h.axes[0].bin_widths / 2,
            color="black",
            marker=grooming_styles[grooming_method].marker,
            markersize=11,
            linestyle="",
            linewidth=3,
        )
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
            color="black",
            linewidth=0,
            alpha=0.3,
        )

        for model_name, (model_calculation, model_with_all_grooming_methods) in models.items():
            model = model_with_all_grooming_methods.get(grooming_method, None)
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
            temp_kwargs = dict(model_styles[f"{collision_system}_{model_name}"])
            temp_kwargs["label"] = model_calculation.label(collision_system=collision_system) if plotting_last_method and not all_methods_on_one_figure else None
            # Need to pop for fill_between since these aren't valid args
            temp_kwargs.pop("marker")
            temp_kwargs.pop("markerfacecolor", None)
            temp_kwargs.pop("markeredgewidth", None)
            # And switch to the proper color
            temp_kwargs["facecolor"] = temp_kwargs.pop("color")
            lower_error, upper_error = _determine_uncertainty_limits_for_model(model=model)

            ax.fill_between(
                model.axes[0].bin_centers,
                model.values - lower_error,
                model.values + upper_error,
                zorder=5,
                alpha=0.8,
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
            lower_error, upper_error = _determine_uncertainty_limits_for_model(model=ratio)
            ax_ratio.fill_between(
                ratio.axes[0].bin_centers,
                ratio.values - lower_error,
                ratio.values + upper_error,
                zorder=5,
                alpha=0.8,
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
        legend_models.anchor= (0.98, 0.98)

        # Create handles and labels by hand, using all models
        legend_elements = []
        for model_name, (model_calculation, _) in models.items():
            model_kwargs = dict(model_styles[f"{collision_system}_{model_name}"])
            legend_elements.append(
                mpl.patches.Patch(
                    facecolor=model_kwargs["color"],
                    label=model_calculation.label(collision_system=collision_system)
                )
            )

        # Next, to create the new legend, we need the existing handles
        # NOTE: We won't have handles from every model because I don't have all of their predictions right now.
        #       However, this should only be a temporary issue. Once fixed, the plots will fix themselves. So for now (May 2023),
        #       I just should look for a quick hack as a temporary fix.
        ax_legend = all_axes[::2].flatten()[-1]
        handles, labels = ax_legend.get_legend_handles_labels()

        # Now that we have the handles, we can apply
        # By convention, we plot the models before the data, so we just need to separate out the data (in the last position)
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


def plot_grooming_model_comparisons_for_single_system(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    models: Mapping[str, tuple[model_calculations.ModelCalculation, Mapping[str, binned_data.BinnedData]]],
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

    # grooming_styling = pb.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = plot_style.label_to_display_string["ALICE"][alice_status]
    # Since the final text is short, we can merge onto one line
    if alice_status != "final":
        text += "\n"
    else:
        text += " "
    text += plot_style.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + plot_style.label_to_display_string["jets"]["general"]
    text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"  # noqa: ISC003
    _plot_data_model_comparison_for_single_system(
        hists=hists,
        models=models,
        grooming_methods=grooming_methods,
        collision_system=collision_system,
        set_zero_to_nan=False,
        all_methods_on_one_figure=False,
        kt_range=kt_range,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system}_model_comparison_{jet_R_str}",
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


def plot_grooming_model_comparisons_for_single_system_one_figure(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    models: Mapping[str, tuple[model_calculations.ModelCalculation, Mapping[str, binned_data.BinnedData]]],
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
    """Plot comparison of grooming methods for a single system."""

    # Validation
    if figure_kt_range is None:
        figure_kt_range = helpers.KtRange(1.5, 15)
    if isinstance(kt_range, helpers.KtRange):
        kt_range = {grooming_method: kt_range for grooming_method in grooming_methods}

    # grooming_styling = pb.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = plot_style.label_to_display_string["ALICE"][alice_status]
    # Since the final text is short, we can merge onto one line
    if alice_status != "final":
        text += "\n"
    else:
        text += " "
    text += plot_style.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + plot_style.label_to_display_string["jets"]["general"]
    text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"  # noqa: ISC003

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
            name=f"unfolded_kt_{collision_system}_model_comparison_{jet_R_str}_one_figure",
            panels=panels,
            figure=pb.Figure(edge_padding={"left": 0.05, "bottom": 0.08, "top": 0.99, "right": 0.99}),
        ),
        output_dir=output_dir,
    )


def _plot_single_system_comparison(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    set_zero_to_nan: bool,
    all_methods_on_one_figure: bool,
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
        all_methods_on_one_figure: Whether to plot all methods on one figure.
        kt_range: Mapping of grooming method to kt range.
        plot_config: Plot configuration.
        output_dir: Output directory.
    """
    grooming_styling = plot_style.define_grooming_styles()

    # Blue, Purple, Green, Red
    _palette = ["#5a97d3",
                "#9671c3",
                "#69a75f",
                "#cc5366",
                "#c758a9"]
    # Pink, Green, Purple, Orange
    _palette_2 = ["#c85a9b",
                 "#78a352",
                 "#787bcf",
                 "#ca6d42"]
    # Orange, Purple, Green, Pink, Teal
    _palette_3 = [#"#c57b3d",
                  "#946fc7",
                  "#7aa444",
                  "#ca5477",
                  "#4cab98"]
    # Green, purple, orange, teal, red/pink
    _palette_4 = ["#72a553",
                  "#a265c2",
                  "#c57c3d",
                  "#6097ce",
                  "#ca5572"]
    # Pastel
    _palette_5 = ["#c7d49f",
                  "#d3b3e3",
                  "#93dacb",
                  "#ebb0a4",
                  "#82c7eb"]
    #
    _palette_6 = [#"#ba543d",
                  "#7e459e",
                  "#85aa55",
                  "#7385d9",
                  "#b84c7d",
                  "#4cab98"]
    _palette_6_mod = {
        "purple": "#7e459e",
        "green": "#85aa55",
        "blue": "#7385d9",
        "magenta": "#b84c7d",
        "teal": "#4cab98",
        "orange": "#FF8301",
    }
    _extended_colors = {
        "alt_purple": "#c09cd3",
        # Generated
        #"alt_green": "#3f591d",
        "alt_green": "#517225",
        # Already existing green
        #"alt_green": "#55a270",
        "alt_blue": "#4bafd0",
    }

    _colors_for_assignments = []
    for _method in grooming_methods:
        _method_to_color = {
            "dynamical_core": _palette_6_mod["purple"],
            "dynamical_kt": _palette_6_mod["green"],
            "dynamical_time": _palette_6_mod["blue"],
            "soft_drop_z_cut_02": _palette_6_mod["magenta"],
            "dynamical_core_z_cut_02": _extended_colors["alt_purple"],
            "dynamical_kt_z_cut_02": _extended_colors["alt_green"],
            "dynamical_time_z_cut_02": _extended_colors["alt_blue"],
            "soft_drop_z_cut_04": _palette_6_mod["orange"],
        }
        _colors_for_assignments.append(_method_to_color[_method])
        # if "dynamical_core" in _method:
        #     _color_for_method = _palette_6_mod["purple"]
        # elif "dynamical_kt" in _method:
        #     _color_for_method = _palette_6_mod["green"]
        # elif "dynamical_time" in _method:
        #     _color_for_method = _palette_6_mod["blue"]
        # elif _method == "soft_drop_z_cut_02":
        #     _color_for_method = _palette_6_mod["magenta"]
        # elif _method == "soft_drop_z_cut_04":
        #     _color_for_method = _palette_6_mod["orange"]
        # else:
        #     raise ValueError(f"Could not assign color for method {_method}")
        #_colors_for_assignments.append(_color_for_method)

    #_markers = ["D", "s", "o", "P", "o"]
    _markers = ["o", "o", "o", "s", "o"]
    # Need to rotate down one since we plot one less
    _markers_ratio = ["o", "o", "s", "o", "o"]

    with sns.color_palette("Set2"):
        # fig, ax = plt.subplots(figsize=(9, 10))
        # Size is specified to make it convenient to compare against Hard Probes plots.
        if all_methods_on_one_figure:
            n_panels = int(np.ceil(len(grooming_methods) / 2))
            fig, all_axes = plt.subplots(
                4,
                n_panels,
                figsize=(10 * n_panels, 20),
                gridspec_kw={"height_ratios": [3, 1, 3, 1]},
                sharex="col",
                sharey="row",
            )
            ax_pairs = [
                (ax, ax_ratio)
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

        #ax.set_prop_cycle(cycler.cycler(color=_palette_6_mod.values()) + cycler.cycler(marker=_markers))
        #ax_ratio.set_prop_cycle(cycler.cycler(color=_palette_6_mod.values()) + cycler.cycler(marker=_markers_ratio))
        ax.set_prop_cycle(cycler.cycler(color=_colors_for_assignments) + cycler.cycler(marker=_markers[:len(_colors_for_assignments)]))
        ax_ratio.set_prop_cycle(cycler.cycler(color=_colors_for_assignments) + cycler.cycler(marker=_markers_ratio[:len(_colors_for_assignments)]))

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

            ## Plot options
            #kwargs = {
            #    "markerfacecolor": "white" if style.fillstyle == "none" else style.color,
            #    "alpha": 1 if style.fillstyle == "none" else 0.8,
            #}
            #if style.fillstyle != "none":
            #    kwargs["markeredgewidth"] = 0

            # Main data points
            p = ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                yerr=h.errors,
                xerr=h.axes[0].bin_widths / 2,
                #marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
                label=grooming_styling[grooming_method].label_short,
                # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
                zorder=3 + _plot_counter,
            )

            # Systematic uncertainty
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
                # y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
                # color=style.color,
                color=p[0].get_color(),
                linewidth=0,
                alpha=0.3,
                zorder=2,
            )

            # Ratio
            # Skip pp because it's not meaningful.
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
                color=p[0].get_color(),
                #marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
                # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
                zorder=3 + _plot_counter,
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
                color=p[0].get_color(),
                linewidth=0,
                alpha=0.3,
                zorder=2,
            )

    # Reference value for ratio
    if all_methods_on_one_figure:
        ...
    else:
        ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=0.9)

        # Labeling and presentation
        plot_config.apply(fig=fig, axes=[ax, ax_ratio])
        # A few additional tweaks.
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
        # ax_ratio.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_grooming_comparisons_for_single_system(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    collision_system: str,
    collision_system_key: str,
    output_dir: Path,
    kt_range: helpers.KtRange | Mapping[str, helpers.KtRange],
    figure_kt_range: helpers.KtRange | None = None,
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
    if label:
        label = f"_{label}"

    # Add event activity to label if needed
    event_activity = ""
    _event_activity_label_map = {
        "pp": "pp",
        "central": r"0-10\%",
        "semi_central": r"30-50\%",
    }
    if collision_system != "pp":
        event_activity = f"{_event_activity_label_map[collision_system]} "

    grooming_styling = plot_style.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = plot_style.label_to_display_string["ALICE"][alice_status]
    text += "\n" + event_activity + plot_style.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + plot_style.label_to_display_string["jets"]["general"]
    text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"  # noqa: ISC003
    _plot_single_system_comparison(
        hists=hists,
        grooming_methods=grooming_methods,
        reference_grooming_method=reference_grooming_method,
        set_zero_to_nan=False,
        all_methods_on_one_figure=False,
        kt_range=kt_range,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system}_comparison_{jet_R_str}{label}",
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
                            range=(0.45, 1.55) if "soft_drop_z_cut_04" not in grooming_methods else (0.1, 1.9),
                            font_size=text_font_size,
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.15, "bottom": 0.095, "top": 0.975}),
        ),
        output_dir=output_dir,
    )


def plot_grooming_comparisons_for_single_system_one_figure(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    collision_system: str,
    collision_system_key: str,
    output_dir: Path,
    kt_range: helpers.KtRange | Mapping[str, helpers.KtRange],
    figure_kt_range: helpers.KtRange | None = None,
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
    if label:
        label = f"_{label}"

    # Add event activity to label if needed
    event_activity = ""
    _event_activity_label_map = {
        "pp": "pp",
        "central": r"0-10\%",
        "semi_central": r"30-50\%",
    }
    if collision_system != "pp":
        event_activity = f"{_event_activity_label_map[collision_system]} "

    grooming_styling = plot_style.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = plot_style.label_to_display_string["ALICE"][alice_status]
    text += "\n" + event_activity + plot_style.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + plot_style.label_to_display_string["jets"]["general"]
    text += "\n" + plot_style.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"  # noqa: ISC003
    _plot_single_system_comparison(
        hists=hists,
        grooming_methods=grooming_methods,
        reference_grooming_method=reference_grooming_method,
        set_zero_to_nan=False,
        all_methods_on_one_figure=True,
        kt_range=kt_range,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system}_comparison_{jet_R_str}{label}_one_figure",
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
                            range=(0.45, 1.55) if "soft_drop_z_cut_04" not in grooming_methods else (0.1, 1.9),
                            font_size=text_font_size,
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.15, "bottom": 0.095, "top": 0.975}),
        ),
        output_dir=output_dir,
    )

