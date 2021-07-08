""" Paper plots

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Mapping, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import seaborn as sns
from pachyderm import binned_data

from jet_substructure.analysis import plot_base as pb
from jet_substructure.analysis import plot_unfolding, unfolding_base
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)

pachyderm.plot.configure()


def _plot_pp_grooming_comparison_with_models(  # noqa: C901
    hists: Mapping[str, plot_unfolding.SingleResult],
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
    grooming_styling = pb.define_grooming_styles()

    name_of_grooming_method_to_draw_models = ""
    for grooming_method in grooming_methods:
        res = [grooming_method in model_predictions for model_predictions in models.values()]
        print(f"grooming_method: {grooming_method}, res: {res}")
        all_models_contain_grooming_method = all(
            [grooming_method in model_predictions for model_predictions in models.values()]
        )
        if all_models_contain_grooming_method:
            print("Setting...")
            name_of_grooming_method_to_draw_models = grooming_method
    logger.info(f"name of name_of_grooming_method_to_draw_models: {name_of_grooming_method_to_draw_models}")

    with sns.color_palette("Set2"):
        # fig, ax = plt.subplots(figsize=(9, 10))
        # Size is specified to make it convenient to compare against Hard Probes plots.
        fig, (ax, ax_ratio_data, ax_ratio_models) = plt.subplots(
            3,
            1,
            figsize=(9, 12),
            gridspec_kw={"height_ratios": [4, 1, 1]},
            sharex=True,
        )

        # Use selected grooming method as a reference, but only in the range where the others are measured.
        ratio_reference_hist_unselected = hists[reference_grooming_method].data

        for grooming_method in grooming_methods:
            # plotting_last_method = grooming_method == grooming_methods[-1]

            # Axes: jet_pt, attr_name
            h_input = hists[grooming_method].data

            # Select range to display.
            h = unfolding_base.select_hist_range(h_input, kt_range[grooming_method])

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
            )

            # Data ratio
            # Skip drawing the reference grooming method in the data ratio because it's not meaningful.
            if grooming_method != reference_grooming_method:
                # Ensure the ratio is defined over the same range.
                # TODO: Refactor when more awake...
                kt_range_for_current_grooming_method = kt_range[grooming_method]
                kt_range_for_reference = kt_range[reference_grooming_method]
                kt_range_min, kt_range_max = tuple(kt_range_for_current_grooming_method)
                if kt_range_min < kt_range_for_reference.min:
                    kt_range_min = kt_range_for_reference.min
                if kt_range_max > kt_range_for_reference.max:
                    kt_range_max = kt_range_for_reference.max
                kt_range_for_comparison = helpers.KtRange(kt_range_min, kt_range_max)
                logger.info(f"kt_range_for_comparison: {kt_range_for_comparison}")
                ratio_reference_hist = unfolding_base.select_hist_range(
                    ratio_reference_hist_unselected,
                    kt_range_for_comparison,
                )
                h = unfolding_base.select_hist_range(
                    h_input,
                    kt_range_for_comparison,
                )
                ratio = h / ratio_reference_hist
                # Ratio + statistical error bars
                ax_ratio_data.errorbar(
                    ratio.axes[0].bin_centers,
                    ratio.values,
                    yerr=ratio.errors,
                    xerr=ratio.axes[0].bin_widths / 2,
                    color=p[0].get_color(),
                    marker="o",
                    markersize=11,
                    linestyle="",
                    linewidth=3,
                )
                # Systematic errors.
                y_relative_error_low = unfolding_base.relative_error(
                    unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].low),
                    unfolding_base.ErrorInput(
                        value=ratio_reference_hist.values,
                        error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].low,
                    ),
                )
                y_relative_error_high = unfolding_base.relative_error(
                    unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].high),
                    unfolding_base.ErrorInput(
                        value=ratio_reference_hist.values,
                        error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].high,
                    ),
                )
                # Store the systematic.
                ratio.metadata["y_systematic"]["quadrature"] = unfolding_base.AsymmetricErrors(
                    low=y_relative_error_low * ratio.values,
                    high=y_relative_error_high * ratio.values,
                )
                y_systematic = ratio.metadata["y_systematic"]["quadrature"]
                pachyderm.plot.error_boxes(
                    ax=ax_ratio_data,
                    x_data=ratio.axes[0].bin_centers,
                    y_data=ratio.values,
                    x_errors=ratio.axes[0].bin_widths / 2,
                    y_errors=np.array([y_systematic.low, y_systematic.high]),
                    color=p[0].get_color(),
                    linewidth=0,
                )

            # Next, the model comparison
            for model_name, model_with_all_grooming_methods in models.items():
                model = model_with_all_grooming_methods.get(grooming_method, None)
                if not model:
                    logger.debug(
                        f"Skipping model {model_name}, grooming method: {grooming_method} because predictions aren't available"
                    )
                    continue

                # Then, plot the model
                model_style = grooming_styling[f"{grooming_method}_compare"]
                # Get the model hist
                model = binned_data.BinnedData.from_existing_data(model)
                # Then normalize as appropriate
                # TODO: Careful, pythia is already normalized, but jetscape wasn't. So we only want to normalize in some cases
                if model_name in models_to_normalize:
                    model /= np.sum(model.values)
                    model /= model.axes[0].bin_widths

                # Determine the overlapping range, since not all of them are the same...
                # TODO: Refactor when more awake...
                kt_range_for_current_grooming_method = kt_range[grooming_method]
                kt_range_for_model = kt_ranges_for_models[model_name]
                kt_range_min, kt_range_max = tuple(kt_range_for_current_grooming_method)
                if kt_range_min < kt_range_for_model.min:
                    kt_range_min = kt_range_for_model.min
                if kt_range_max > kt_range_for_model.max:
                    kt_range_max = kt_range_for_model.max
                kt_range_for_model_comparison = helpers.KtRange(kt_range_min, kt_range_max)
                logger.info(
                    f"kt_range_for_model_comparison: {kt_range_for_model_comparison}, {grooming_method}, {model_name}"
                )
                logger.info(f"kt_range_for_model: {kt_range_for_model}")

                # And select the same range.
                model_kt_range_selected = unfolding_base.select_hist_range(model, kt_range_for_model)

                # NOTE: For analytical, we need to plot a fill_between to account for the systematics...

                # And plot
                # Make sure we copy the settings so we can modify them
                temp_kwargs = dict(plot_unfolding._models_styles[model_name])
                temp_kwargs["label"] = (
                    temp_kwargs["label"] if grooming_method == name_of_grooming_method_to_draw_models else None
                )
                ax.errorbar(
                    model_kt_range_selected.axes[0].bin_centers,
                    model_kt_range_selected.values,
                    # yerr=model.errors,
                    # xerr=model.axes[0].bin_widths / 2,
                    color=grooming_styling[grooming_method].color,
                    # marker=style.marker,
                    # fillstyle=grooming_styling[grooming_method].fillstyle,
                    # linestyle="",
                    # label=_models_styles[model_name]["label"] if plotting_last_method else None,
                    zorder=model_style.zorder,
                    alpha=0.7,
                    **temp_kwargs,
                )

                model_for_ratio = unfolding_base.select_hist_range(
                    model,
                    kt_range_for_model_comparison,
                )
                h_for_model_ratio = unfolding_base.select_hist_range(
                    h_input,
                    kt_range_for_model_comparison,
                )

                # Ratio
                ratio = model_for_ratio / h_for_model_ratio

                # Ratio + statistical error bars
                ax_ratio_models.errorbar(
                    ratio.axes[0].bin_centers,
                    ratio.values,
                    yerr=ratio.errors,
                    xerr=ratio.axes[0].bin_widths / 2,
                    color=p[0].get_color(),
                    marker="o",
                    markersize=11,
                    linestyle="",
                    linewidth=3,
                )
                # Systematic errors.
                y_relative_error_low = unfolding_base.relative_error(
                    unfolding_base.ErrorInput(
                        value=h_for_model_ratio.values,
                        error=h_for_model_ratio.metadata["y_systematic"]["quadrature"].low,
                    ),
                )
                y_relative_error_high = unfolding_base.relative_error(
                    unfolding_base.ErrorInput(
                        value=h_for_model_ratio.values,
                        error=h_for_model_ratio.metadata["y_systematic"]["quadrature"].high,
                    ),
                )
                # From error prop, pythia has no systematic error, so we just convert the relative errors.
                ratio.metadata["y_systematic"] = {}
                ratio.metadata["y_systematic"]["quadrature"] = unfolding_base.AsymmetricErrors(
                    low=y_relative_error_low * ratio.values,
                    high=y_relative_error_high * ratio.values,
                )
                y_systematic = ratio.metadata["y_systematic"]["quadrature"]
                pachyderm.plot.error_boxes(
                    ax=ax_ratio_models,
                    x_data=ratio.axes[0].bin_centers,
                    y_data=ratio.values,
                    x_errors=ratio.axes[0].bin_widths / 2,
                    y_errors=np.array([y_systematic.low, y_systematic.high]),
                    color=p[0].get_color(),
                    linewidth=0,
                )

    # reference value for data and model ratios
    ax_ratio_data.axhline(y=1, color="black", linestyle="dashed", zorder=1)
    ax_ratio_models.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio_data, ax_ratio_models])
    # A few additional tweaks.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_pp_grooming_comparison_with_models(
    hists: Mapping[str, plot_unfolding.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    models: Mapping[str, Mapping[str, binned_data.BinnedData]],
    collision_system: str,
    collision_system_key: str,
    jet_R_str: str,
    output_dir: Path,
    kt_range: Union[helpers.KtRange, Mapping[str, helpers.KtRange]],
    kt_ranges_for_models: Mapping[str, helpers.KtRange],
    models_to_normalize: Sequence[str],
    figure_kt_range: helpers.KtRange = helpers.KtRange(1.5, 15),
) -> None:
    """Plot comparison of grooming methods, along with models, for pp."""

    # Validation
    if isinstance(kt_range, helpers.KtRange):
        kt_range = {grooming_method: kt_range for grooming_method in grooming_methods}

    grooming_styling = pb.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = pb.label_to_display_string["ALICE"]["work_in_progress"]
    text += "\n" + pb.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_pp_grooming_comparison_with_models(
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
                            range=(5e-3, 1),
                            font_size=22,
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="lower left", font_size=22),
                ),
                # Data ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{Method}}{\text{"
                            + grooming_styling[reference_grooming_method].label
                            + "}}$",
                            range=(0.45, 1.55),
                            font_size=22,
                        ),
                    ],
                ),
                # Model ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=tuple(figure_kt_range), font_size=22),  # type: ignore
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{Model}}{\text{Data}}$",
                            range=(0.45, 1.55),
                            font_size=22,
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.08)),
        ),
        output_dir=output_dir,
    )
