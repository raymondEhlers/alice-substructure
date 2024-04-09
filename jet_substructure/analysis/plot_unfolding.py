""" Functionality related to preparing unfolding outputs and plotting.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Callable,
    Mapping,
    Sequence,
)

import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

import pachyderm.plot
from jet_substructure.analysis import full_results_helpers, unfolding_analysis
from jet_substructure.analysis import plot_style as pb
from jet_substructure.base import helpers
from pachyderm import binned_data

logger = logging.getLogger(__name__)

pachyderm.plot.configure()


def hist_stat_tests_KS_chi2(
    ratio: binned_data.BinnedData,
    use_ROOT: bool = True,
) -> tuple[float, float]:
    if use_ROOT:
        # Create ROOT hist (:-()
        import ROOT  # pyright: ignore [reportMissingImports] # noqa: F401

        #data_hist = ROOT.TH1D("data", "data", len(hist.axes[0].bin_edges) - 1, hist.axes[0].bin_edges)

        #for i, (_value, _error) in enumerate(zip(hist.values, hist.errors), start=1):
        #    data_hist.SetBinContent(i, _value)
        #    data_hist.SetBinError(i, _error)
        hist = ratio
        data_hist = hist.to_ROOT()
        # Add systematics in quadrature with stat
        # Generate the comparison
        y_systematic: full_results_helpers.AsymmetricErrors = ratio.metadata["y_systematic"]["quadrature"]
        # Determine which values we need to grab
        # When below 1, take the upper error, and do the opposite above 1
        # NOTE: We basically assume that they're one sigma
        y_systematic_values = np.where(ratio.values <= 1, y_systematic.high, y_systematic.low)
        for i, (_value, _error) in enumerate(zip(hist.values, hist.errors), start=1):
            data_hist.SetBinError(i, np.sqrt(_error ** 2 + y_systematic_values[i-1] ** 2))

        reference_hist = data_hist.Clone("reference")
        # Clear out the data, etc
        reference_hist.Reset()
        # And define at unity
        for i in range(1, reference_hist.GetXaxis().GetNbins() + 1):
            reference_hist.SetBinContent(i, 1)
            reference_hist.SetBinError(i, 0)

        chi2_pvalue: float = data_hist.Chi2Test(reference_hist, "WW")
        ks_test_pvalue: float = data_hist.KolmogorovTest(reference_hist)
        return chi2_pvalue, ks_test_pvalue
    else:  # noqa: RET505
        import scipy.stats

        # Create distribution from histogram and bin edges
        # From: https://stackoverflow.com/a/72224046/12907985
        hist_dist = scipy.stats.rv_histogram(
            (ratio.values, ratio.axes[0].bin_edges)
        )

        # Perform the test
        # NOTE: Validation by passing the same distributions...
        result = scipy.stats.kstest(
            hist_dist,
            scipy.stats.uniform.cdf,
        )

        return result.pvalue  # type: ignore[no-any-return]


def plausible_stat_test(ratio: binned_data.BinnedData, n_samples: int, n_sigma_stat: float = 1.0) -> float:
    """
    Sample the number of counts stat as a Poisson distributions, and then compare to unity with the uncertainties being the systematic
    (consistent if all points overlap with unity within one sigma)

    In practice, I normalize very early on in grabbing my unfolding hists, so turn this off will be a pain in the ass. So we'll
    instead assume Gaussian distributed counts.
    """
    samples = np.zeros((len(ratio.axes[0].bin_centers), n_samples), dtype=np.float32)
    rng = np.random.default_rng()

    # Sample possible values
    for i, (value, error) in enumerate(zip(ratio.values, ratio.errors)):
        #logger.debug(f"{i=}: {value=}, {error=}")
        samples[i, :] = rng.normal(loc=value, scale=error * n_sigma_stat, size=n_samples)

    # Generate the comparison
    y_systematic: full_results_helpers.AsymmetricErrors = ratio.metadata["y_systematic"]["quadrature"]
    # Determine which values we need to grab
    # When below 1, take the upper error, and do the opposite above 1
    # NOTE: Shift up to 1 so we can make the comparison to the sampled values directly
    # NOTE: We basically assume that they're one sigma
    comparison_values_from_systematics = np.where(ratio.values <= 1, 1 - y_systematic.high, 1 + y_systematic.low)
    #logger.debug(f"{comparison_values_from_systematics}")
    comparison_values_from_systematics_promoted = comparison_values_from_systematics[:, np.newaxis]

    results = np.where(
        (ratio.values <= 1)[:, np.newaxis],
        samples > comparison_values_from_systematics_promoted,
        samples < comparison_values_from_systematics_promoted,
    )

    # Access the number of samples where all are within the comparison range
    #logger.debug(f"{np.count_nonzero(results, axis=1)}")
    passed = np.all(results, axis=0)
    fraction_passed = np.count_nonzero(passed) / n_samples
    return fraction_passed  # noqa: RET504


def plot_relative_individual_systematics(
    unfolded: unfolding_analysis.SingleResult,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    """Plot relative individual systematic errors."""
    import mplhep as hep

    # Setup
    logger.debug("Plotting systematic relative errors.")
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for name, systematic in unfolded.data.metadata["y_systematic"].items():
        # Upper values
        extra_args = {}
        if name == "quadrature":
            extra_args = {
                "color": "black",
                "linewidth": 2,
            }
        p = hep.histplot(
            H=np.ones_like(unfolded.data.values) + (systematic.high / unfolded.data.values),
            bins=unfolded.data.axes[0].bin_edges,
            label=name.replace("_", " "),
            alpha=0.8,
            **extra_args,
        )
        # Lower values
        # Need to drop this - otherwise it will conflict with existing arguments.
        if name == "quadrature":
            extra_args.pop("color")
        hep.histplot(
            H=np.ones_like(unfolded.data.values) - (systematic.low / unfolded.data.values),
            bins=unfolded.data.axes[0].bin_edges,
            color=p[0].stairs.get_edgecolor(),
            alpha=0.8,
            **extra_args,
        )

    # For comparison, add the statistical too
    ax.errorbar(
        unfolded.data.axes[0].bin_centers,
        np.ones_like(unfolded.data.axes[0].bin_centers),
        yerr=unfolded.data.errors / unfolded.data.values,
        # color=style.color,
        marker="o",
        linestyle="",
        label="Statistical",
        # alpha=0.8,
    )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    logger.info(f"Writing plot to '{output_dir / figure_name}.pdf'")
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)



#_model_palette = sns.color_palette("husl", n_colors=6)
#_model_palette = sns.color_palette("Accent", n_colors=10)
#_model_palette = sns.color_palette("colorblind", n_colors=10)
#_model_palette = sns.color_palette("dark", n_colors=6)

_model_palette: list[tuple[float, float, float]] = [
    (53, 73, 222),
    (170, 52, 222),
    (223, 82, 87),
    (225, 220, 103),
    (90, 224, 102),
    (57, 225, 215),
    (137, 185, 224),
    (89, 147, 223),
    (223, 34, 219),
    (216, 142, 224),
    (223, 124, 53),
    (108, 223, 41),
]
_model_palette = [
    (color[0] / 254, color[1] / 254, color[2] / 254)
    for color in _model_palette
]
_model_palette = _model_palette[1:] + [_model_palette[0]]

_models_styles = {
    "pythia": {
        "label": "PYTHIA8 Monash 2013",
        "linewidth": 3,
        "linestyle": "-",
        "marker": "s",
        "color": _model_palette[0],
        #color=_model_palette[7],
        "markerfacecolor": "none",
        #markerfacecolor="white",
        "markeredgewidth": 3,
    },
    "analytical": {
        "label": "Caucal et al.",
        "linewidth": 3,
        "linestyle": "-.",
        "marker": "P",
        #color=_model_palette[1],
        #color=_model_palette[5],
        #color=_model_palette[8],
        #color=_model_palette[4],
        "color": _model_palette[3],
    },
    "sherpa_lund": {
        "label": "SHERPA (Lund)",
        # NOTE: This will overlap with jetscape, but we currently (8 July 2021) can't compare them, so it's fine.
        #       To be resolved when the plotting plans are a bit clearer.
        "linewidth": 3,
        "linestyle": "--",
        "marker": "*",
        #color=_model_palette[2],
        "color": _model_palette[1],
        #color=_model_palette[5],
        #color=_model_palette[2],
    },
    "sherpa_ahadic": {
        "label": "SHERPA (AHADIC)",
        "linewidth": 3,
        "linestyle": ":",
        "marker": "X",
        #color=_model_palette[3],
        #color=_model_palette[7],
        #color=_model_palette[6],
        #color=_model_palette[3],
        "color": _model_palette[6],
    },
    "jetscape": {
        "label": "JETSCAPEv3.5 AA22",
        "linewidth": 3,
        "linestyle": "--",
        "marker": "D",
        #color=_model_palette[4],
        #color=_model_palette[8],
        #color=_model_palette[4],
        "color": _model_palette[3],
    },
    "hybrid_moliere": {
        "label": "Hybrid w/ wake + Moliere",
        "linewidth": 3,
        "linestyle": "--",
        "marker": "D",
        #color=_model_palette[4],
        #color=_model_palette[8],
        #color=_model_palette[4],
        "color": _model_palette[5],
    },
    "hybrid_without_moliere": {
        "label": "Hybrid w/ wake",
        "linewidth": 3,
        "linestyle": "--",
        "marker": "D",
        #color=_model_palette[4],
        #color=_model_palette[8],
        #color=_model_palette[4],

        #color=_model_palette[6],
        "color": _model_palette[1],
    },
}

# Based on better colorblind presets
_palette_6_mod = [
    "#7e459e",
    "#85aa55",
    "#7385d9",
    "#b84c7d",
    "#4cab98",
]


def _plot_data_model_comparison_for_single_system(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    models: Mapping[str, Mapping[str, binned_data.BinnedData]],
    grooming_methods: Sequence[str],
    set_zero_to_nan: bool,
    kt_range: Mapping[str, helpers.KtRange],
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    grooming_styling = pb.define_grooming_styles()

    _markers_by_grooming_method = {
        "dynamical_core": "o",
        "dynamical_kt": "o",
        "dynamical_time": "o",
        "soft_drop_z_cut_02": "s",
        "dynamical_core_z_cut_02": "o",
        "dynamical_kt_z_cut_02": "o",
        "dynamical_time_z_cut_02": "o",
        "soft_drop_z_cut_04": "s",
    }

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

    #_colors_for_assignments = []
    #for _method in grooming_methods:
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
    #_colors_for_assignments.append(_method_to_color[_method])

    with sns.color_palette("Set2"):
        # fig, ax = plt.subplots(figsize=(9, 10))
        # Size is specified to make it convenient to compare against Hard Probes plots.
        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        #ax.set_prop_cycle(cycler.cycler(color=_palette_6_mod) + cycler.cycler(marker=_markers))
        #ax_ratio.set_prop_cycle(cycler.cycler(color=_palette_6_mod) + cycler.cycler(marker=_markers_ratio))
        #ax.set_prop_cycle(cycler.cycler(marker=_markers))
        #ax_ratio.set_prop_cycle(cycler.cycler(marker=_markers_ratio))

        for _i, grooming_method in enumerate(grooming_methods):
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
                marker=_markers_by_grooming_method[grooming_method],
                markersize=11,
                linestyle="",
                linewidth=3,
                label=grooming_styling[grooming_method].label_short,
                color=_method_to_color[grooming_method],
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
            )

            for model_name, model_with_all_grooming_methods in models.items():
                model = model_with_all_grooming_methods.get(grooming_method, None)
                if not model:
                    logger.debug(
                        f"Skipping model {model_name}, grooming method: {grooming_method} because predictions aren't available"
                    )
                    continue

                # Then, plot the model
                model_style = grooming_styling[f"{grooming_method}_compare"]
                # Get the model for the reference.
                model = binned_data.BinnedData.from_existing_data(model)
                # NOTE: Careful, pythia is already normalized, but jetscape wasn't. So we need to resolve this...
                #       Probably best to have some kind of "prepare model" function, which we can decide to use or not.
                # Then normalize
                #####model /= np.sum(model.values)
                #####model /= model.axes[0].bin_widths
                # And select the same range.
                model = full_results_helpers.select_hist_range(model, kt_range[grooming_method])

                # And plot
                # Make sure we copy the settings so we can modify them
                temp_kwargs = dict(_models_styles[model_name])
                temp_kwargs["label"] = temp_kwargs["label"] if plotting_last_method else None
                temp_kwargs.pop("color")
                temp_kwargs.pop("marker")
                ax.errorbar(
                    model.axes[0].bin_centers,
                    model.values,
                    # yerr=model.errors,
                    # xerr=model.axes[0].bin_widths / 2,
                    # NOTE: This isn't right if there are multiple models, but it got me through the previews
                    color=p[0].get_color(),
                    #color=grooming_styling[grooming_method].color,
                    # marker=style.marker,
                    # fillstyle=grooming_styling[grooming_method].fillstyle,
                    # linestyle="",
                    # label=_models_styles[model_name]["label"] if plotting_last_method else None,
                    zorder=model_style.zorder,
                    alpha=0.7,
                    **temp_kwargs,
                )

                # Ratio
                # Could move down here if you want to see the entire range
                model = full_results_helpers.select_hist_range(model, kt_range[grooming_method])
                ratio = model / h

                # Ratio + statistical error bars
                ax_ratio.errorbar(
                    ratio.axes[0].bin_centers,
                    ratio.values,
                    yerr=ratio.errors,
                    xerr=ratio.axes[0].bin_widths / 2,
                    color=p[0].get_color(),
                    marker=_markers_by_grooming_method[grooming_method],
                    markersize=11,
                    linestyle="",
                    linewidth=3,
                )
                # Systematic errors.
                y_relative_error_low = full_results_helpers.relative_error(
                    full_results_helpers.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].low),
                )
                y_relative_error_high = full_results_helpers.relative_error(
                    full_results_helpers.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].high),
                )
                # From error prop, pythia has no systematic error, so we just convert the relative errors.
                ratio.metadata["y_systematic"] = {}
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
                )

    # reference value for ratio
    ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=0.9)

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])
    # A few additional tweaks.
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_grooming_model_comparisons_for_single_system(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    models: Mapping[str, Mapping[str, binned_data.BinnedData]],
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

    text = pb.label_to_display_string["ALICE"][alice_status]
    text += "\n" + pb.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_data_model_comparison_for_single_system(
        hists=hists,
        models=models,
        grooming_methods=grooming_methods,
        set_zero_to_nan=False,
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
                        pb.AxisConfig("x", label=r"$k_{\text{T,g}}\:(\text{GeV}/c)$", range=tuple(figure_kt_range), font_size=text_font_size),  # type: ignore[arg-type]
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



def _plot_single_system_comparison(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    set_zero_to_nan: bool,
    kt_range: Mapping[str, helpers.KtRange | helpers.RgRange],
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    grooming_styling = pb.define_grooming_styles()

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
        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        #ax.set_prop_cycle(cycler.cycler(color=_palette_6_mod.values()) + cycler.cycler(marker=_markers))
        #ax_ratio.set_prop_cycle(cycler.cycler(color=_palette_6_mod.values()) + cycler.cycler(marker=_markers_ratio))
        ax.set_prop_cycle(cycler.cycler(color=_colors_for_assignments) + cycler.cycler(marker=_markers[:len(_colors_for_assignments)]))
        ax_ratio.set_prop_cycle(cycler.cycler(color=_colors_for_assignments) + cycler.cycler(marker=_markers_ratio[:len(_colors_for_assignments)]))

        # Use selected grooming method as a reference, but only in the range where the others are measured.
        ratio_reference_hist_unselected = hists[reference_grooming_method].data

        for _plot_counter, grooming_method in enumerate(grooming_methods):
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
                reference=kt_range[reference_grooming_method],
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

    grooming_styling = pb.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = pb.label_to_display_string["ALICE"][alice_status]
    text += "\n" + event_activity + pb.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_single_system_comparison(
        hists=hists,
        grooming_methods=grooming_methods,
        reference_grooming_method=reference_grooming_method,
        set_zero_to_nan=False,
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

def plot_Rg_grooming_comparisons_for_single_system(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    collision_system: str,
    collision_system_key: str,
    output_dir: Path,
    rg_range: helpers.RgRange | Mapping[str, helpers.RgRange],
    figure_rg_range: helpers.RgRange | None = None,
    jet_R_str: str = "R02",
    alice_status: str = "work_in_progress",
    text_font_size: int = 31,
    label: str = "",
) -> None:
    """Plot comparison of grooming methods for a single system."""

    # Validation
    if figure_rg_range is None:
        figure_rg_range = helpers.RgRange(0, 0.2)
    if isinstance(rg_range, helpers.RgRange):
        rg_range = {grooming_method: rg_range for grooming_method in grooming_methods}
    if label:
        label = f"_{label}"

    grooming_styling = pb.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = pb.label_to_display_string["ALICE"][alice_status]
    text += "\n" + pb.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_single_system_comparison(
        hists=hists,
        grooming_methods=grooming_methods,
        reference_grooming_method=reference_grooming_method,
        set_zero_to_nan=False,
        kt_range=rg_range,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system}_comparison_{jet_R_str}{label}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}R_{\text{g}}$",
                            font_size=text_font_size,
                        ),
                    ],
                    text=pb.TextConfig(x=0.98, y=0.98, text=text, font_size=text_font_size),
                    legend=pb.LegendConfig(location="lower left", font_size=round(text_font_size*0.8), anchor=(0.015, 0.025), marker_label_spacing=0.075),
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$R_{\text{g}}$", range=tuple(figure_rg_range), font_size=text_font_size),  # type: ignore[arg-type]
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{Method}}{\text{"
                            + grooming_styling[reference_grooming_method].label_short
                            + "}}$",
                            range=(0.45, 1.55) if "soft_drop_z_cut_04" not in grooming_methods else (0.25, 1.55),
                            font_size=text_font_size,
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.15, "bottom": 0.095, "top": 0.975}),
        ),
        output_dir=output_dir,
    )


def _plot_pp_PbPb_comparison(  # noqa: C901
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_method: str,
    set_zero_to_nan: bool,
    event_activity_to_kt_range: Mapping[str, helpers.KtRange],
    plot_config: pb.PlotConfig,
    output_dir: Path,
    models: Mapping[str, Mapping[str, binned_data.BinnedData]] | None = None,
    plausible_stat_test_parameters: list[float] | None = None,
    calculate_hist_stat_tests_KS_chi2: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Plot PbPb with systematics compared to pp with systematics for a set of grooming methods."""
    # Validations
    if models is None:
        models = {}
    _plausible_stat_test_results: dict[str, dict[float, float]] = {}
    _ks_test_results: dict[str, float] = {}
    _chi2_test_results: dict[str, float] = {}

    logger.info("Plotting grooming method comparison for kt with systematics")

    # Setup
    _event_activity_label_map = {
        "pp": "pp",
        "central": r"0-10\% $\text{Pb--Pb}$",
        "semi_central": r"30-50\% $\text{Pb--Pb}$",
    }

    _p7 = [
        # Suggested by Hannah:
        # Too yellow for my test :-/
        #"#FFA500",
        #"#dd9132",
        #"#5a3c00",
        #"#FF8301",

        #"#9cb94a",
        "#845cba",
        #"#be4977",

        "#FF8301",

        ## Red, green from generation where the first two values are fixed
        #"#ca5c61",
        ## Teal:
        ##"#59c28c",
        ##
        ## Blue:
        #"#b1c2de",
        ##
        #"#7ca153",

        # Option #2
        #"#85aa55",
        # Blue
        #"#7385d9",

        # Option #1
        # I think I like these...
        # A blue
        # This first blue seems too similar
        #"#7277cb",
        "#4bafd0",
        # A green
        "#55a270",

        # Others from the original generation
        # Orange
        #"#c06835",
        # Teal-ish
        "#59c28c",
    ]
    # Use consistent colors by settings it via names
    _p7_named = {
        "pp": "#845cba",
        "semi_central": "#FF8301",
        "central": "#4bafd0",
    }

    with sns.color_palette("Set2"):
        # fig, ax = plt.subplots(figsize=(9, 10))
        # Size is specified to make it convenient to compare against Hard Probes plots.
        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        ax.set_prop_cycle(cycler.cycler(color=_p7))
        ax_ratio.set_prop_cycle(cycler.cycler(color=_p7[2:]))

        # Use pp as reference, but only in the range where the others are measured.
        ratio_reference_hist_unselected = hists["pp"].data

        # Determine whether we should plot the ratio in black points and grey uncertainties
        # We only want to do that if we have models for comparison and have just pp and one
        # other collision system available
        plot_ratio_black_and_white = (models and len(hists) == 2)

        # Collision system is a bit misleading because it's really just a high label, but good enough for a quick look.
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
            p = ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                yerr=h.errors,
                xerr=h.axes[0].bin_widths / 2,
                marker="s" if "soft_drop" in grooming_method else "o",
                markersize=11,
                linestyle="",
                linewidth=3,
                label=_event_activity_label_map[collision_system],
                # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
                zorder=3 + _plot_counter,
                color=_p7_named[collision_system],
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
            if collision_system == "pp":
                continue

            # Ensure the ratio is defined over the same range.
            ratio_reference_hist = full_results_helpers.select_hist_range(
                ratio_reference_hist_unselected, event_activity_to_kt_range[collision_system]
            )
            logger.debug(f"h: {h.axes[0].bin_edges}")
            logger.debug(f"ratio_reference_hist: {ratio_reference_hist.axes[0].bin_edges}")
            ratio = h / ratio_reference_hist
            # Ratio + statistical error bars
            ax_ratio.errorbar(
                ratio.axes[0].bin_centers,
                ratio.values,
                yerr=ratio.errors,
                xerr=ratio.axes[0].bin_widths / 2,
                color="black" if plot_ratio_black_and_white else p[0].get_color(),
                marker="s" if "soft_drop" in grooming_method else "o",
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
            # Sanity check
            # TODO: If this passes once, delete it. I've checked this a lot now...
            test_relative_y_error_low = np.sqrt(
                (h.metadata["y_systematic"]["quadrature"].low / h.values) ** 2
                + (ratio_reference_hist.metadata["y_systematic"]["quadrature"].low / ratio_reference_hist.values) ** 2
            )
            test_relative_y_error_high = np.sqrt(
                (h.metadata["y_systematic"]["quadrature"].high / h.values) ** 2
                + (ratio_reference_hist.metadata["y_systematic"]["quadrature"].high / ratio_reference_hist.values) ** 2
            )
            np.testing.assert_allclose(y_relative_error_low, test_relative_y_error_low)
            np.testing.assert_allclose(y_relative_error_high, test_relative_y_error_high)
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
                color="grey" if plot_ratio_black_and_white else p[0].get_color(),
                linewidth=0,
                alpha=0.3,
                zorder=2,
            )

            if plausible_stat_test_parameters is not None:
                _plausible_stat_test_results[collision_system] = {}
                for _param in plausible_stat_test_parameters:
                    _result = plausible_stat_test(ratio=ratio, n_samples=int(1e6), n_sigma_stat=_param)
                    _plausible_stat_test_results[collision_system][_param] = _result

            if calculate_hist_stat_tests_KS_chi2:
                _chi2_res, _ks_res = hist_stat_tests_KS_chi2(ratio=ratio, use_ROOT=True)
                _chi2_test_results[collision_system] = _chi2_res
                _ks_test_results[collision_system] = _ks_res

        # Plot model comparison if available
        for i_model, (model_name, model_with_all_grooming_methods) in enumerate(models.items()):
            model = model_with_all_grooming_methods.get(grooming_method, None)
            if not model:
                logger.debug(
                    f"Skipping model {model_name}, grooming method: {grooming_method} because predictions aren't available"
                )
                continue

            # Select the relevant kt range
            model = full_results_helpers.select_hist_range(
                model, event_activity_to_kt_range[collision_system]
            )

            # TODO: Cleanup - this is a mess! Just needed to move quickly for HP2023
            # Colors (copied from elsewhere)
            _palette_6_mod = {
                #"purple": "#7e459e",
                "green": "#85aa55",
                #"blue": "#7385d9",
                "magenta": "#b84c7d",
                "teal": "#4cab98",
                #"orange": "#FF8301",
            }
            _extended_colors = {
                "purple": "#7e459e",
                #"alt_purple": "#c09cd3",
                # Generated
                #"alt_green": "#3f591d",
                #"alt_green": "#517225",
                # Already existing green
                #"alt_green": "#55a270",
                #"alt_blue": "#4bafd0",
                # random additions...
                "teal": "#4cab98",
                #"blue": "#7385d9",
                #"alt_green_2": "#55a270",
                #"alt_red": _model_palette[1],
                "alt_red": _model_palette[1],
            }
            # Fill between
            temp_kwargs = dict(_models_styles[model_name])
            temp_kwargs["label"] = temp_kwargs["label"]
            temp_kwargs["facecolor"] = temp_kwargs.pop("color")
            temp_kwargs["facecolor"] = list(reversed(_extended_colors.values()))[i_model]
            temp_kwargs.pop("marker")
            ax_ratio.fill_between(
                model.axes[0].bin_centers,
                model.values - model.errors,
                model.values + model.errors,
                alpha=0.7,
                **temp_kwargs,
            )

    # Test results
    # Since we return the values, it's not so critical to print them most of them.
    # Can always uncomment them again later if needed.
    #if _plausible_stat_test_results:
    #    logger.info(f"Plausible stat test fractions: {_plausible_stat_test_results}")
    #if _chi2_test_results:
    #    logger.info(f"Chi2 p value: {_chi2_test_results}")
    #if _ks_test_results:
    #    logger.info(f"KS p value: {_ks_test_results}")

    # Reference value for ratio
    ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=0.9)

    # Labeling and presentation
    ax_ratio_legend_config = None
    ax_ratio_handles, ax_ratio_labels = ax_ratio.get_legend_handles_labels()
    logger.info(f"{len(ax_ratio_handles)=}")
    logger.info(f"{len(ax_ratio_labels)=}, {ax_ratio_labels=}")
    if models and len(ax_ratio_handles) % 2 == 1:
        logger.info("Handling manually")
        # Pop out legend handler so that it skips due the plot config and
        # and we can handle it manually
        ax_ratio_legend_config = plot_config.panels[1].legend
        plot_config.panels[1].legend = None
        insert_position = round((len(ax_ratio_handles) + 1)/2)
        ax_ratio_handles.insert(insert_position, ax_ratio.plot([], [], color=(0, 0, 0, 0), label=" ")[0])
        ax_ratio_labels.insert(insert_position, "")
        #ax_ratio_handles.insert(insert_position, ax_ratio_handles[0])
        #ax_ratio_labels.insert(insert_position, ax_ratio_labels[0])
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])
    # A few additional tweaks.
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
    # Need a manual hack here since the range has gotten so big with the models
    if models and grooming_method == "soft_drop_z_cut_04":
        ax_ratio.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
    if ax_ratio_legend_config:
        ax_ratio_legend_config.apply(
            ax=ax_ratio,
            legend_handles=ax_ratio_handles,
            legend_labels=ax_ratio_labels,
        )

    filename = f"{plot_config.name}_{grooming_method}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)

    return _ks_test_results, _chi2_test_results


def plot_pp_PbPb_comparison(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_method: str,
    output_dir: Path,
    event_activity_to_kt_range: Mapping[str, helpers.KtRange],
    kt_display_range: tuple[float, float] = (1.5, 15),
    jet_R_str: str = "R04",
    alice_status: str = "work_in_progress",
    text_font_size: int = 31,
    models: Mapping[str, Mapping[str, binned_data.BinnedData]] | None = None,
    plausible_stat_test_parameters: list[float] | None = None,
    calculate_hist_stat_tests_KS_chi2: bool = False,
    additional_label: str = "",
) -> tuple[dict[str, float], dict[str, float]]:
    """Plot PbPb unfolded results with systematics."""
    jet_pt_bin = next(iter(hists.values())).ranges[0]
    grooming_styling = pb.define_grooming_styles()
    style = grooming_styling[grooming_method]

    text = pb.label_to_display_string["ALICE"][alice_status]
    text += "\n" + pb.label_to_display_string["collision_system"]["pp_PbPb_5TeV"]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"

    name = "unfolded_kt_pp_PbPb"
    if additional_label:
        name += f"_{additional_label}"
    if models:
        name += "_models"
    name += f"_comparison_{jet_R_str}"

    # from: https://stackoverflow.com/a/42170161
    # doesn't seem to work...
    from matplotlib.legend_handler import HandlerLine2D
    class SymHandler2(HandlerLine2D):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):  # type: ignore[no-untyped-def] # noqa: ARG002
            xx= 2.3*height
            return super().create_artists(legend, orig_handle, xdescent, xx, width, height, fontsize, trans)

    _ratio_range = (0.3, 1.7)
    if "central" in hists and models:
        _ratio_range = (0.1, 1.9)
    if "z_cut_04" in grooming_method:
        _ratio_range = (-0.2, 2.2) if models else (0.1, 1.9)


    _results = _plot_pp_PbPb_comparison(
        hists=hists,
        models=models,
        grooming_method=grooming_method,
        set_zero_to_nan=False,
        event_activity_to_kt_range=event_activity_to_kt_range,
        plausible_stat_test_parameters=plausible_stat_test_parameters,
        calculate_hist_stat_tests_KS_chi2=calculate_hist_stat_tests_KS_chi2,
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
                    legend=pb.LegendConfig(location="lower left", font_size=22, anchor=(0.01, 0.02), ncol=2, marker_label_spacing=0.05, label_spacing=0.1, handle_height=1.3, column_spacing=0.30),
                    #legend=pb.LegendConfig(location="lower left", font_size=24, anchor=(0.01, 0.01), ncol=2, marker_label_spacing=0.05, label_spacing=0.1, column_spacing=0.20,
                    #    handler_map={
                    #        mpl.lines.Line2D: SymHandler2()
                    #    }
                    #),
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.15, "bottom": 0.095, "top": 0.975}),
        ),
        output_dir=output_dir,
    )

    return _results  # noqa: RET504


def _plot_simple_kt_with_systematics(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    set_zero_to_nan: bool,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """Plot PbPb with systematics for a set of grooming methods."""
    logger.info("Plotting grooming method comparison for kt with systematics")

    # fig, ax = plt.subplots(figsize=(9, 10))
    # Size is specified to make it convenient to compare against Hard Probes plots.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    grooming_styling = pb.define_grooming_styles()

    for grooming_method in grooming_methods:
        # Setup
        style = grooming_styling[grooming_method]

        # Axes: jet_pt, attr_name
        h = hists[grooming_method].data

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

        # Main data points
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
            # color=p[0].get_color(),
            color=style.color,
            linewidth=0,
        )

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_PbPb_systematics_simple(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    grooming_methods: Sequence[str],
    event_activity: str,
    output_dir: Path,
    kt_range: tuple[float, float] = (1.5, 15),
    jet_R: str = "R04",
) -> None:
    """Plot PbPb unfolded results with systematics."""
    jet_pt_bin = hists[grooming_methods[0]].ranges[0]
    event_activity_map = {
        "central": r"0-10\%",
        "semi_central": r"30-50\%",
    }

    text = pb.label_to_display_string["ALICE"]["work_in_progress"]
    if event_activity != "pp":
        text += (
            "\n" + pb.label_to_display_string["collision_system"]["PbPb"] + f", {event_activity_map[event_activity]}"
        )
    else:
        text += "\n" + pb.label_to_display_string["collision_system"]["pp_5TeV"]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_simple_kt_with_systematics(
        hists=hists,
        grooming_methods=grooming_methods,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_systematics_simple_{event_activity}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=kt_range),
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            range=(7e-3, 1),
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="lower left", font_size=22),
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.06}),
        ),
        output_dir=output_dir,
    )


def _plot_compare_kt_with_systematics(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    reference: Mapping[str, binned_data.BinnedData],
    grooming_methods: Sequence[str],
    set_zero_to_nan: bool,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """Plot PbPb with systematics for a set of grooming methods."""
    logger.info("Plotting grooming method comparison for kt with systematics")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax, ax_ratio = axes

    grooming_styling = pb.define_grooming_styles()

    for grooming_method in grooming_methods:
        # Setup
        style = grooming_styling[grooming_method]

        # Axes: jet_pt, attr_name
        h = hists[grooming_method].data

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

        # Main data points
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
            # color=p[0].get_color(),
            color=style.color,
            linewidth=0,
        )

        # Ratio + statistical error bars from unfolding
        ratio = h / reference[grooming_method]
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
        # Systematic errors.
        y_relative_error_low = full_results_helpers.relative_error(
            full_results_helpers.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].low)
        )
        y_relative_error_high = full_results_helpers.relative_error(
            full_results_helpers.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].high)
        )
        # From error prop, pythia has no systematic error, so we just convert the relative errors.
        ratio.metadata["y_systematic"] = full_results_helpers.AsymmetricErrors(
            low=y_relative_error_low * ratio.values,
            high=y_relative_error_high * ratio.values,
        )
        pachyderm.plot.error_boxes(
            ax=ax_ratio,
            x_data=ratio.axes[0].bin_centers,
            y_data=ratio.values,
            x_errors=ratio.axes[0].bin_widths / 2,
            y_errors=np.array([ratio.metadata["y_systematic"].low, ratio.metadata["y_systematic"].high]),
            color=style.color,
            linewidth=0,
            # label = "Background", color = plot_base.AnalysisColors.fit,
        )

    # Reference value for ratio
    ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_PbPb_systematics(
    hists: Mapping[str, unfolding_analysis.SingleResult],
    reference: Mapping[str, binned_data.BinnedData],
    grooming_methods: Sequence[str],
    event_activity: str,
    output_dir: Path,
    kt_range: tuple[float, float] = (1.5, 15),
    jet_R: str = "R04",
) -> None:
    """Plot PbPb unfolded results with systematics."""
    jet_pt_bin = hists[grooming_methods[0]].ranges[0]
    event_activity_map = {
        "central": r"0-10\%",
        "semi_central": r"30-50\%",
    }

    text = pb.label_to_display_string["ALICE"]["work_in_progress"]
    if event_activity != "pp":
        text += (
            "\n" + pb.label_to_display_string["collision_system"]["PbPb"] + f", {event_activity_map[event_activity]}"
        )
    else:
        text += "\n" + pb.label_to_display_string["collision_system"]["pp_5TeV"]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_compare_kt_with_systematics(
        hists=hists,
        grooming_methods=grooming_methods,
        set_zero_to_nan=False,
        reference=reference,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_systematics_{event_activity}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            range=(1e-3, 0.3),
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="lower left", font_size=22),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=kt_range),
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{data}}{\text{PYTHIA}}$",
                            range=(0.55, 1.45),
                        ),
                    ]
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.06}),
        ),
        output_dir=output_dir,
    )


def setup_unfolding_closures(
    substructure_variable: str,
    grooming_method: str,
    smeared_var_range: helpers.KtRange,
    smeared_untagged_var: helpers.KtRange,
    smeared_jet_pt_range: helpers.JetPtRange,
    collision_system: str,
    n_iter_compare: int,
    max_n_iter: int | None,
    suffix: str,
    double_counting_cut: str,
    input_dir_tag: str,
    output_dir: Path,
    pure_matches: bool = False,
    output_dir_tag: str | None = None,
) -> dict[str, unfolding_analysis.UnfoldingOutput]:
    # Setup the input files
    unfolding_outputs = {}
    unfolding_outputs["default"] = unfolding_analysis.UnfoldingOutput(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        base_dir=output_dir,
        n_iter_compare=n_iter_compare,
        max_n_iter=max_n_iter,
        pure_matches=pure_matches,
        suffix=suffix,
        input_dir_tag=input_dir_tag,
        output_dir_tag=output_dir_tag,
        double_counting_cut=double_counting_cut,
    )

    # These should always exist.
    unfolding_outputs["trivial_closure"] = unfolding_analysis.UnfoldingOutput(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        base_dir=output_dir,
        n_iter_compare=n_iter_compare,
        max_n_iter=max_n_iter,
        pure_matches=pure_matches,
        suffix=suffix,
        input_dir_tag=input_dir_tag,
        output_dir_tag=output_dir_tag,
        double_counting_cut=double_counting_cut,
        label="closure_trivial_hybrid_smeared_as_input",
        raw_hist_name="smeared",
    )

    unfolding_outputs["closure_later_iter"] = unfolding_analysis.UnfoldingOutput(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        base_dir=output_dir,
        n_iter_compare=n_iter_compare,
        max_n_iter=max_n_iter,
        pure_matches=pure_matches,
        suffix=suffix,
        input_dir_tag=input_dir_tag,
        output_dir_tag=output_dir_tag,
        double_counting_cut=double_counting_cut,
        label="closure_5_iter_5",
    )

    try:
        unfolding_outputs["split_MC"] = unfolding_analysis.UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            max_n_iter=max_n_iter,
            pure_matches=pure_matches,
            suffix=suffix,
            input_dir_tag=input_dir_tag,
            output_dir_tag=output_dir_tag,
            double_counting_cut=double_counting_cut,
            label="closure_split_MC",
            raw_hist_name="h2_pseudo_data",
            true_hist_name="h2_pseudo_true",
        )
    except FileNotFoundError:
        logger.debug("Skipping split MC because the output file doesn't exist.")

    try:
        unfolding_outputs["reweight_pseudo_data"] = unfolding_analysis.UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            max_n_iter=max_n_iter,
            pure_matches=pure_matches,
            suffix=suffix,
            input_dir_tag=input_dir_tag,
            output_dir_tag=output_dir_tag,
            double_counting_cut=double_counting_cut,
            label="closure_reweight_pseudo_data",
            raw_hist_name="h2_pseudo_data",
            true_hist_name="h2_pseudo_true",
        )
    except FileNotFoundError:
        logger.debug("Skipping reweighted pseudo data because the output file doesn't exist.")

    try:
        unfolding_outputs["reweight_response"] = unfolding_analysis.UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            max_n_iter=max_n_iter,
            pure_matches=pure_matches,
            suffix=suffix,
            input_dir_tag=input_dir_tag,
            output_dir_tag=output_dir_tag,
            double_counting_cut=double_counting_cut,
            label="closure_reweight_response",
            raw_hist_name="h2_pseudo_data",
            true_hist_name="h2_pseudo_true",
        )
    except FileNotFoundError:
        logger.debug("Skipping reweighted response because the output file doesn't exist.")

    try:
        unfolding_outputs["thermal_model"] = unfolding_analysis.UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            max_n_iter=max_n_iter,
            pure_matches=pure_matches,
            suffix=suffix,
            input_dir_tag=input_dir_tag,
            output_dir_tag=output_dir_tag,
            double_counting_cut=double_counting_cut,
            # Trivial closure
            #label="thermal_model_closure_trivial_hybrid_smeared_as_input",
            #raw_hist_name="smeared",
            # Split MC closure
            label="thermal_model_closure_split_MC",
            raw_hist_name="h2_pseudo_data",
            true_hist_name="h2_pseudo_true",
        )
    except FileNotFoundError:
        logger.debug("Skipping thermal model closure because the output file doesn't exist.")

    return unfolding_outputs


def setup_unfolding_outputs(  # noqa: C901
    substructure_variable: str,
    grooming_method: str,
    smeared_var_range: helpers.KtRange,
    smeared_untagged_var: helpers.KtRange,
    smeared_jet_pt_range: helpers.JetPtRange,
    collision_system: str,
    n_iter_compare: int,
    max_n_iter: int | None,
    suffix: str,
    double_counting_cut: str,
    input_dir_tag: str,
    output_dir: Path,
    truncation_shift: float = 5,
    displaced_untagged_above_range: bool = True,
    displaced_extremum: float | None = None,
    skip_reweighted_prior_in_systematics: bool = False,
    output_dir_tag: str | None = None,
    model_dependence_configuration: unfolding_analysis.ModelDependenceConfiguration | None = None,
) -> dict[str, unfolding_analysis.UnfoldingOutput]:
    # Validation
    # Keep the truncation positive so we know how we've shifted.
    if truncation_shift < 0:
        truncation_shift = np.abs(truncation_shift)
    if displaced_extremum is None:
        # NOTE: We set 20 externally (in the unfolding configuration in parsl). But it should work fine
        #       because it encompasses all possible PbPb ranges used so far.
        displaced_extremum = 20

    # Setup the input files
    unfolding_outputs = {}
    unfolding_outputs["default"] = unfolding_analysis.UnfoldingOutput(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        base_dir=output_dir,
        n_iter_compare=n_iter_compare,
        max_n_iter=max_n_iter,
        pure_matches=False,
        suffix=suffix,
        input_dir_tag=input_dir_tag,
        output_dir_tag=output_dir_tag,
        double_counting_cut=double_counting_cut,
    )
    logger.info(f"{grooming_method} default: {unfolding_outputs['default'].identifier}")

    try:
        unfolding_outputs["tracking_efficiency"] = unfolding_analysis.UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            max_n_iter=max_n_iter,
            pure_matches=False,
            suffix=suffix,
            input_dir_tag=input_dir_tag,
            output_dir_tag=output_dir_tag,
            double_counting_cut=double_counting_cut,
            label="tracking_efficiency",
        )
    except FileNotFoundError:
        logger.debug("Skipping tracking efficiency because the output file doesn't exist.")

    try:
        unfolding_outputs["truncation_low"] = unfolding_analysis.UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=helpers.JetPtRange(
                smeared_jet_pt_range.min - truncation_shift, smeared_jet_pt_range.max
            ),
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            max_n_iter=max_n_iter,
            pure_matches=False,
            suffix=suffix,
            input_dir_tag=input_dir_tag,
            output_dir_tag=output_dir_tag,
            double_counting_cut=double_counting_cut,
            label="truncation",
        )
        unfolding_outputs["truncation_high"] = unfolding_analysis.UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=helpers.JetPtRange(
                smeared_jet_pt_range.min + truncation_shift, smeared_jet_pt_range.max
            ),
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            max_n_iter=max_n_iter,
            pure_matches=False,
            suffix=suffix,
            input_dir_tag=input_dir_tag,
            output_dir_tag=output_dir_tag,
            double_counting_cut=double_counting_cut,
            label="truncation",
        )
    except FileNotFoundError:
        logger.debug("Skipping truncation because the output file doesn't exist.")

    try:
        unfolding_outputs["random_binning"] = unfolding_analysis.UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            max_n_iter=max_n_iter,
            pure_matches=False,
            suffix=suffix,
            input_dir_tag=input_dir_tag,
            output_dir_tag=output_dir_tag,
            double_counting_cut=double_counting_cut,
            label="random_binning",
        )
    except FileNotFoundError:
        logger.debug("Skipping random binning because the output file doesn't exist.")

    try:
        # If the untagged bin is disabled, then skip this
        if smeared_untagged_var.min != smeared_untagged_var.max:
            if displaced_untagged_above_range:
                displaced_untagged_var = helpers.KtRange(smeared_var_range.max, displaced_extremum)
            else:
                displaced_untagged_var = helpers.KtRange(displaced_extremum, smeared_var_range.min)

            unfolding_outputs["untagged_bin"] = unfolding_analysis.UnfoldingOutput(
                substructure_variable=substructure_variable,
                grooming_method=grooming_method,
                smeared_var_range=smeared_var_range,
                smeared_untagged_var=displaced_untagged_var,
                smeared_jet_pt_range=smeared_jet_pt_range,
                collision_system=collision_system,
                base_dir=output_dir,
                n_iter_compare=n_iter_compare,
                max_n_iter=max_n_iter,
                pure_matches=False,
                input_dir_tag=input_dir_tag,
                output_dir_tag=output_dir_tag,
                suffix=suffix,
                double_counting_cut=double_counting_cut,
            )
            #logger.debug(f"untagged_bin: {unfolding_outputs['untagged_bin'].identifier}")
        else:
            logger.info("Skipping untagged bin outputs because it is disabled")

    except FileNotFoundError:
        logger.debug("Skipping untagged bin location because the output file doesn't exist.")

    if not skip_reweighted_prior_in_systematics:
        try:
            unfolding_outputs["reweight_prior"] = unfolding_analysis.UnfoldingOutput(
                substructure_variable=substructure_variable,
                grooming_method=grooming_method,
                smeared_var_range=smeared_var_range,
                smeared_untagged_var=smeared_untagged_var,
                smeared_jet_pt_range=smeared_jet_pt_range,
                collision_system=collision_system,
                base_dir=output_dir,
                n_iter_compare=n_iter_compare,
                max_n_iter=max_n_iter,
                pure_matches=False,
                suffix=suffix,
                input_dir_tag=input_dir_tag,
                output_dir_tag=output_dir_tag,
                double_counting_cut=double_counting_cut,
                label="reweight_prior",
            )
        except FileNotFoundError:
            logger.debug("Skipping reweighted prior because the output file doesn't exist.")
    else:
        logger.debug(
            "Skipping reweighted prior because it was requested (probably for pp, where we take a model dependence instead)."
        )

    # Model dependence
    if model_dependence_configuration is not None:
        for model_name in model_dependence_configuration.all_models:
            # Validation
            if not model_name and not model_dependence_configuration.legacy_production:
                _msg = "Only allowed to load unlabeled model dependence if using a legacy production. Please check settings!"
                raise ValueError(_msg)

            # Handle the special case where the nominal is the default. In this case, no need to load it twice.
            if model_name == "default":
                continue

            label = ""
            # Handle the legacy case of the nominal being
            if model_name:
                label = f"_{model_name}"
            else:
                logger.warning("Loading unlabeled model dependence via legacy production.")
            try:
                # Careful here: the outputs in pp are not in the standard format. But this is a convenient fiction.
                unfolding_outputs[f"model_dependence{label}"] = unfolding_analysis.UnfoldingOutput(
                    substructure_variable=substructure_variable,
                    grooming_method=grooming_method,
                    smeared_var_range=smeared_var_range,
                    smeared_untagged_var=smeared_untagged_var,
                    smeared_jet_pt_range=smeared_jet_pt_range,
                    collision_system=collision_system,
                    base_dir=output_dir,
                    n_iter_compare=n_iter_compare,
                    max_n_iter=max_n_iter,
                    pure_matches=False,
                    suffix=suffix,
                    input_dir_tag=input_dir_tag,
                    output_dir_tag=output_dir_tag,
                    double_counting_cut=double_counting_cut,
                    label=f"model_dependence{label}",
                )
            except FileNotFoundError:
                logger.debug(f"Skipping model dependence '{model_name}' because the output file doesn't exist.")

    # Background subtraction
    # NOTE: We don't make this directly configurable because we just want it to grab all possible values.
    #       We'll sort which to use later.
    for background_setting in ["Rmax070", "Rmax050", "Rmax005"]:
        try:
            unfolding_outputs[background_setting] = unfolding_analysis.UnfoldingOutput(
                substructure_variable=substructure_variable,
                grooming_method=grooming_method,
                smeared_var_range=smeared_var_range,
                smeared_untagged_var=smeared_untagged_var,
                smeared_jet_pt_range=smeared_jet_pt_range,
                collision_system=collision_system,
                base_dir=output_dir,
                n_iter_compare=n_iter_compare,
                max_n_iter=max_n_iter,
                pure_matches=False,
                suffix=suffix,
                input_dir_tag=input_dir_tag,
                output_dir_tag=output_dir_tag,
                double_counting_cut=double_counting_cut,
                label=f"{background_setting}",
            )
        except FileNotFoundError:
            logger.debug(f"Skipping background setting {background_setting} because the output file doesn't exist.")

    return unfolding_outputs


def _load_unfolded_outputs(
    grooming_method: str,
    substructure_variable: str,
    smeared_var_range: helpers.KtRange,
    smeared_untagged_var: helpers.KtRange,
    smeared_jet_pt_range: helpers.JetPtRange,
    collision_system: str,
    event_activity: str,
    jet_R_str: str,
    n_iter_compare: int,
    max_n_iter: int | None,
    truncation_shift: int,
    displaced_extremum: float,
    input_dir_tag: str,
    output_dir: Path,
    double_counting_cut: str = "",
    tag_after_suffix: str = "",
    displaced_untagged_above_range: bool = True,
    skip_reweighted_prior_in_systematics: bool = False,
    output_dir_tag: str | None = None,
    model_dependence_configuration: unfolding_analysis.ModelDependenceConfiguration | None = None,
) -> tuple[dict[str, unfolding_analysis.UnfoldingOutput], dict[str, unfolding_analysis.UnfoldingOutput], dict[str, unfolding_analysis.UnfoldingOutput]]:
    # Validation
    suffix = f"{event_activity}_{jet_R_str}"
    if tag_after_suffix:
        suffix += f"_{tag_after_suffix}"

    logger.debug(f"{grooming_method}: Loading closures...")
    unfolding_closure_outputs = setup_unfolding_closures(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        n_iter_compare=n_iter_compare,
        max_n_iter=max_n_iter,
        suffix=suffix,
        input_dir_tag=input_dir_tag,
        output_dir_tag=output_dir_tag,
        output_dir=output_dir,
        double_counting_cut=double_counting_cut,
    )
    try:
        logger.debug(f"{grooming_method}: Attempting to load pure matches closures...")
        unfolding_closure_pure_matches_outputs = setup_unfolding_closures(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            n_iter_compare=n_iter_compare,
            max_n_iter=max_n_iter,
            suffix=suffix,
            input_dir_tag=input_dir_tag,
            output_dir_tag=output_dir_tag,
            output_dir=output_dir,
            double_counting_cut=double_counting_cut,
            pure_matches=True,
        )
    except FileNotFoundError:
        logger.debug("Skipping pure matches because the output file doesn't exist.")
        unfolding_closure_pure_matches_outputs = {}

    logger.debug(f"{grooming_method}: Loading systematics outputs...")
    unfolding_systematics_outputs = setup_unfolding_outputs(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        n_iter_compare=n_iter_compare,
        max_n_iter=max_n_iter,
        suffix=suffix,
        double_counting_cut=double_counting_cut,
        input_dir_tag=input_dir_tag,
        output_dir_tag=output_dir_tag,
        output_dir=output_dir,
        truncation_shift=truncation_shift,
        displaced_untagged_above_range=displaced_untagged_above_range,
        displaced_extremum=displaced_extremum,
        skip_reweighted_prior_in_systematics=skip_reweighted_prior_in_systematics,
        model_dependence_configuration=model_dependence_configuration,
    )

    return unfolding_closure_outputs, unfolding_closure_pure_matches_outputs, unfolding_systematics_outputs


def load_unfolded_outputs(
    grooming_methods: Sequence[str],
    substructure_variable: str,
    smeared_var_range: helpers.KtRange | Mapping[str, helpers.KtRange],
    smeared_untagged_var: helpers.KtRange | Mapping[str, helpers.KtRange],
    smeared_jet_pt_range: helpers.JetPtRange,
    collision_system: str,
    event_activity: str,
    jet_R_str: str,
    n_iter_compare: int | Mapping[str, int],
    truncation_shift: int,
    displaced_extremum: float,
    input_dir_tag: dict[str, str] | str,
    output_dir: Path,
    tag_after_suffix: str | Mapping[str, str] = "",
    double_counting_cut: dict[str, str] | str = "",
    displaced_untagged_above_range: bool = True,
    skip_reweighted_prior_in_systematics: bool = False,
    output_dir_tag: dict[str, str | None] | str | None = None,
    max_n_iter: dict[str, int | None] | int | None = None,
    model_dependence_configuration: dict[str, unfolding_analysis.ModelDependenceConfiguration | None] | unfolding_analysis.ModelDependenceConfiguration | None = None,
) -> tuple[
    dict[str, dict[str, unfolding_analysis.UnfoldingOutput]], dict[str, dict[str, unfolding_analysis.UnfoldingOutput]], dict[str, dict[str, unfolding_analysis.UnfoldingOutput]]
]:
    """Load the unfolding results for the provided configuration.

    This is a giant steering function. It's flexible, but complicated as a consequence.
    See the notebook(s) for how to call it.

    Returns:
        (unfolding_closure_outputs, unfolding_closure_pure_matches_outputs, unfolding_systematics_outputs) where each output includes
        a dict of grooming methods, and each grooming method includes a dict of unfolding outputs.
    """
    # Validation
    # Copy for every grooming method
    if isinstance(smeared_var_range, helpers.RangeSelector):
        smeared_var_range = {grooming_method: smeared_var_range for grooming_method in grooming_methods}
    if isinstance(smeared_untagged_var, helpers.RangeSelector):
        smeared_untagged_var = {grooming_method: smeared_untagged_var for grooming_method in grooming_methods}
    if isinstance(n_iter_compare, int):
        n_iter_compare = {grooming_method: n_iter_compare for grooming_method in grooming_methods}
    if isinstance(input_dir_tag, str):
        input_dir_tag = {grooming_method: input_dir_tag for grooming_method in grooming_methods}
    if isinstance(output_dir_tag, str) or output_dir_tag is None:
        output_dir_tag = {grooming_method: output_dir_tag for grooming_method in grooming_methods}
    if isinstance(tag_after_suffix, str):
        tag_after_suffix = {grooming_method: tag_after_suffix for grooming_method in grooming_methods}
    if isinstance(double_counting_cut, str):
        double_counting_cut = {grooming_method: double_counting_cut for grooming_method in grooming_methods}
    if isinstance(max_n_iter, int) or max_n_iter is None:
        max_n_iter = {grooming_method: max_n_iter for grooming_method in grooming_methods}
    if isinstance(model_dependence_configuration, unfolding_analysis.ModelDependenceConfiguration) or model_dependence_configuration is None:
        model_dependence_configuration = {grooming_method: model_dependence_configuration for grooming_method in grooming_methods}

    unfolding_closure_outputs = {}
    unfolding_closure_pure_matches_outputs = {}
    unfolding_systematics_outputs = {}
    for grooming_method in grooming_methods:
        (
            unfolding_closure_outputs[grooming_method],
            unfolding_closure_pure_matches_outputs[grooming_method],
            unfolding_systematics_outputs[grooming_method],
        ) = _load_unfolded_outputs(
            grooming_method=grooming_method,
            substructure_variable=substructure_variable,
            smeared_var_range=smeared_var_range[grooming_method],
            smeared_untagged_var=smeared_untagged_var[grooming_method],
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            event_activity=event_activity,
            jet_R_str=jet_R_str,
            n_iter_compare=n_iter_compare[grooming_method],
            truncation_shift=truncation_shift,
            displaced_extremum=displaced_extremum,
            input_dir_tag=input_dir_tag[grooming_method],
            output_dir=output_dir,
            tag_after_suffix=tag_after_suffix[grooming_method],
            double_counting_cut=double_counting_cut[grooming_method],
            displaced_untagged_above_range=displaced_untagged_above_range,
            skip_reweighted_prior_in_systematics=skip_reweighted_prior_in_systematics,
            output_dir_tag=output_dir_tag[grooming_method],
            max_n_iter=max_n_iter[grooming_method],
            model_dependence_configuration=model_dependence_configuration[grooming_method],
        )

    return (
        unfolding_closure_outputs,
        unfolding_closure_pure_matches_outputs,
        unfolding_systematics_outputs,
    )


def _unfolded_outputs_with_systematics(
    grooming_method: str,
    unfolding_systematics_outputs: dict[str, dict[str, unfolding_analysis.UnfoldingOutput]],
    true_jet_pt_range: helpers.JetPtRange,
    unfolding_related_systematic_treatment: str,
    smooth_systematic_uncertainty_contributions: list[str],
    model_dependence_configuration: unfolding_analysis.ModelDependenceConfiguration | None = None,
    non_closure_configuration: unfolding_analysis.NonClosureConfiguration | None = None,
    background_subtraction_configuration: unfolding_analysis.BackgroundSubtractionConfiguration | None = None,
) -> tuple[unfolding_analysis.SingleResult, binned_data.BinnedData]:
    """Steers the actual calculation of the systematic uncertainties.

    """
    logger.info(f"Calculating systematics for {grooming_method} with unfolding related sys. treatment of \"{unfolding_related_systematic_treatment}\".")
    # First, we need the nominal result.
    unfolded = unfolded_substructure_results(
        unfolding_outputs=unfolding_systematics_outputs[grooming_method],
        true_jet_pt_range=true_jet_pt_range,
        model_dependence_configuration=model_dependence_configuration,
    )

    # Then, we calculate the systematics based on the provided parameters.
    unfolded_with_systematics = calculate_systematics(
        unfolded=unfolded,
        unfolding_outputs=unfolding_systematics_outputs[grooming_method],
        true_jet_pt_range=true_jet_pt_range,
        unfolding_related_systematic_treatment=unfolding_related_systematic_treatment,
        smooth_systematic_uncertainty_contributions=smooth_systematic_uncertainty_contributions,
        model_dependence_configuration=model_dependence_configuration,
        non_closure_configuration=non_closure_configuration,
        background_subtraction_configuration=background_subtraction_configuration,
    )

    # And we include the true reference (ie. generally pythia true level). We could derive this elsewhere,
    # but it's extraordinarily useful to have it available, so we just dive in here.
    true_reference = unfolding_systematics_outputs[grooming_method]["default"].true_substructure(
        unfolding_systematics_outputs[grooming_method]["default"].true_hist_name, true_jet_pt_range=true_jet_pt_range
    )

    return unfolded_with_systematics, true_reference


def unfolded_outputs_with_systematics(
    grooming_methods: Sequence[str],
    unfolding_systematics_outputs: dict[str, dict[str, unfolding_analysis.UnfoldingOutput]],
    unfolding_closure_outputs: dict[str, dict[str, unfolding_analysis.UnfoldingOutput]],
    true_jet_pt_range: helpers.JetPtRange,
    unfolding_related_systematic_treatment: str,
    smooth_systematic_uncertainty_contributions: dict[str, list[str]] | None = None,
    model_dependence_configuration: dict[str, unfolding_analysis.ModelDependenceConfiguration | None] | unfolding_analysis.ModelDependenceConfiguration | None = None,
    non_closure_configuration: dict[str, unfolding_analysis.NonClosureConfiguration | None] | unfolding_analysis.NonClosureConfiguration | None = None,
    background_subtraction_configuration: dict[str, unfolding_analysis.BackgroundSubtractionConfiguration | None] | unfolding_analysis.BackgroundSubtractionConfiguration | None = None,
) -> tuple[dict[str, unfolding_analysis.SingleResult], dict[str, binned_data.BinnedData]]:
    """Calculates the unfolded results with systematic uncertainties.

    The values returned here are much more convenient to work with, but inherently are more limited since
    we've reduced the available information.

    Returns:
        (unfolded_with_systematics, true_reference) where each is a dict of grooming methods,
            and each grooming method includes an unfolding SingleResult
    """
    # Validation
    if smooth_systematic_uncertainty_contributions is None:
        smooth_systematic_uncertainty_contributions = {grooming_method: [] for grooming_method in grooming_methods}
    if isinstance(model_dependence_configuration, unfolding_analysis.ModelDependenceConfiguration) or model_dependence_configuration is None:
        model_dependence_configuration = {grooming_method: model_dependence_configuration for grooming_method in grooming_methods}
    if isinstance(non_closure_configuration, unfolding_analysis.NonClosureConfiguration) or non_closure_configuration is None:
        non_closure_configuration = {grooming_method: non_closure_configuration for grooming_method in grooming_methods}
    if isinstance(background_subtraction_configuration, unfolding_analysis.BackgroundSubtractionConfiguration) or background_subtraction_configuration is None:
        background_subtraction_configuration = {grooming_method: background_subtraction_configuration for grooming_method in grooming_methods}

    unfolded_with_systematics = {}
    true_reference = {}
    for grooming_method in grooming_methods:
        # Add in the closures relevant for non-closure into systematics
        if non_closure_configuration[grooming_method] is not None:
            for _contributor in non_closure_configuration[grooming_method].contributors:  # type: ignore[union-attr]
                unfolding_systematics_outputs[grooming_method][_contributor] = unfolding_closure_outputs[grooming_method][_contributor]

        (
            unfolded_with_systematics[grooming_method],
            true_reference[grooming_method],
        ) = _unfolded_outputs_with_systematics(
            grooming_method=grooming_method,
            unfolding_systematics_outputs=unfolding_systematics_outputs,
            true_jet_pt_range=true_jet_pt_range,
            unfolding_related_systematic_treatment=unfolding_related_systematic_treatment,
            smooth_systematic_uncertainty_contributions=smooth_systematic_uncertainty_contributions[grooming_method],
            model_dependence_configuration=model_dependence_configuration[grooming_method],
            non_closure_configuration=non_closure_configuration[grooming_method],
            background_subtraction_configuration=background_subtraction_configuration[grooming_method],
        )

    return unfolded_with_systematics, true_reference


def unfolded_substructure_results(
    unfolding_outputs: Mapping[str, unfolding_analysis.UnfoldingOutput],
    true_jet_pt_range: helpers.JetPtRange,
    model_dependence_configuration: unfolding_analysis.ModelDependenceConfiguration | None = None,
) -> dict[str, unfolding_analysis.SingleResult]:
    """Convert unfolded results into individual unfolded substructure results (selecting a particular iteration).

    This is useful for working with substructure systematics.

    Note:
        We always select the n iter from the default unfolded result.

    Args:
        unfolding_output: All unfolded outputs.
        true_jet_pt_range: True jet pt range for the substructure result.
    Returns:
        Unfolded substructure results.
    """
    # Validation
    skip_converting_model_dependence = False
    _entry_for_model_dependence = np.array(["model_dependence" in k for k in unfolding_outputs])

    if model_dependence_configuration is not None and model_dependence_configuration.legacy_production:
        # For the outputs from Leticia, which are in a different format than our usual.
        skip_converting_model_dependence = True
        # Cross check
        if np.count_nonzero(_entry_for_model_dependence) == 1 and not skip_converting_model_dependence:
            _msg = "Using legacy production, but too many model dependence entries. Check your input!"
            raise ValueError(_msg)

    unfolded = {}
    for k, v in unfolding_outputs.items():
        if skip_converting_model_dependence and k == "model_dependence":
            _msg = "Skipping conversion of model dependence due to legacy production. We'll handle it later."
            logger.info(_msg)
            # We have to handle this manually. See the systematics calculation.
            continue

        #logger.debug(f"Converted to single result for {k=}, {true_jet_pt_range=}")
        unfolded[k] = unfolding_analysis.SingleResult(
            # NOTE: We want to match the iter of the default case.
            data=v.unfolded_substructure(
                n_iter=unfolding_outputs["default"].n_iter_compare,
                true_jet_pt_range=true_jet_pt_range,
            ),
            n_iter=unfolding_outputs["default"].n_iter_compare,
            ranges=[true_jet_pt_range],
        )
    return unfolded


def _calculate_max_relative_error_from_contributions(
    relative_uncertainty_by_contribution: dict[str, npt.NDArray[np.float64]],
    n_values: int,
) -> npt.NDArray[np.float64]:
    """Simple helper for calculating the maximum contribution bin-by-bin"""
    final_relative_uncertainty = np.zeros(n_values, dtype=np.float64)
    for _name, relative_uncertainty in relative_uncertainty_by_contribution.items():
        entry_has_larger_uncertainties_mask = np.abs(relative_uncertainty) > np.abs(final_relative_uncertainty)
        final_relative_uncertainty[entry_has_larger_uncertainties_mask] = relative_uncertainty[entry_has_larger_uncertainties_mask]
        #logger.debug(f"Contribution {name} has contributions at {entry_has_larger_uncertainties_mask}")
        #logger.debug(f"{final_relative_uncertainty=}")

    return final_relative_uncertainty


def _calculate_quadrature_relative_error_from_contributions(
    relative_uncertainty_by_contribution: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Simple helper for calculating the maximum contribution bin-by-bin"""
    final_relative_uncertainty: npt.NDArray[np.float64] = np.sqrt(
        np.sum(
            [
                relative_uncertainty ** 2
                for relative_uncertainty in relative_uncertainty_by_contribution.values()
            ],
            axis=0,
        )
    )
    return final_relative_uncertainty


def calculate_systematics(  # noqa: C901
    unfolded: Mapping[str, unfolding_analysis.SingleResult],
    unfolding_outputs: Mapping[str, unfolding_analysis.UnfoldingOutput],
    true_jet_pt_range: helpers.JetPtRange,
    unfolding_related_systematic_treatment: str,
    smooth_systematic_uncertainty_contributions: list[str],
    truncation_iter: helpers.RangeSelector | None = None,
    model_dependence_configuration: unfolding_analysis.ModelDependenceConfiguration | None = None,
    non_closure_configuration: unfolding_analysis.NonClosureConfiguration | None = None,
    background_subtraction_configuration: unfolding_analysis.BackgroundSubtractionConfiguration | None = None,
) -> unfolding_analysis.SingleResult:
    # Validation
    if truncation_iter is None:
        truncation_iter = helpers.RangeSelector(1, 1)
    if truncation_iter.min < 0:
        truncation_iter = helpers.RangeSelector(-1 * truncation_iter.min, truncation_iter.max)
    if unfolding_related_systematic_treatment not in ["all", "max", "std_dev"]:
        _msg = f"Unrecognized treatment of unfolding systematics: {unfolding_related_systematic_treatment}"
        raise ValueError(_msg)
    # Setup
    unfolded["default"].data.metadata["y_systematic"] = {}

    # Tracking efficiency
    # This is treated as a symmetric uncertainty.
    # However, we store it as asymmetric errors objects for consistency with everything else.
    try:
        # NOTE: Unlike the others, we take the abs and set the values here directly because
        #       we want them to be symmetric.
        tracking_efficiency_sym = np.abs(unfolded["tracking_efficiency"].data.values - unfolded["default"].data.values)
        unfolded["default"].data.metadata["y_systematic"]["tracking_efficiency"] = full_results_helpers.AsymmetricErrors(
            tracking_efficiency_sym, tracking_efficiency_sym
        )
    except KeyError as e:
        logger.debug(f"Skipping tracking efficiency because of {e}")

    # Everything else is treated asymmetrically, potentially one-sided.
    unfolding_related_uncertainties = {}
    # Per Rey, he has truncation, binning, prior, and regularization. I include:
    # "truncation"
    # "regularization"
    # "random_binning"
    # "untagged_bin"
    # "reweight_prior"
    ## Truncation
    try:
        unfolding_related_uncertainties[
            "truncation"
        ] = full_results_helpers.AsymmetricErrors.calculate_errors(
            unfolded["truncation_low"].data.values - unfolded["default"].data.values,
            unfolded["truncation_high"].data.values - unfolded["default"].data.values,
        )
    except KeyError as e:
        logger.debug(f"Skipping truncation because of {e}")

    # Regularization
    # +/- iterations
    unfolding_related_uncertainties[
        "regularization"
    ] = full_results_helpers.AsymmetricErrors.calculate_errors(
        unfolded["default"].data.values
        - unfolding_outputs["default"]
        .unfolded_substructure(
            n_iter=unfolding_outputs["default"].n_iter_compare - truncation_iter.min,  # type: ignore[arg-type]
            true_jet_pt_range=true_jet_pt_range,
        )
        .values,
        unfolded["default"].data.values
        - unfolding_outputs["default"]
        .unfolded_substructure(
            n_iter=unfolding_outputs["default"].n_iter_compare + truncation_iter.max,  # type: ignore[arg-type]
            true_jet_pt_range=true_jet_pt_range,
        )
        .values,
    )

    # An aside from systematics, but useful: determine how much we change when we go to the next iteration.
    # This is useful for a line in the paper, and it's really easy to calculate here, so just go for it and print it.
    log_change_from_next_iteration = True
    if log_change_from_next_iteration:
        difference_n_iter = (
            unfolded["default"].data.values
            - unfolding_outputs["default"]
            .unfolded_substructure(
                n_iter=unfolding_outputs["default"].n_iter_compare + truncation_iter.max,  # type: ignore[arg-type]
                true_jet_pt_range=true_jet_pt_range,
            )
            .values
        ) / unfolded["default"].data.values
        logger.warning(f"Max difference from n_iter: {np.max(difference_n_iter)}")

    # Random binning
    # Take as a symmetric uncertainty because it's not clear why it should be one sided given that
    # we're taking only one variation.
    try:
        random_binning_sym = unfolded["random_binning"].data.values - unfolded["default"].data.values
        unfolding_related_uncertainties[
            "random_binning"
        ] = full_results_helpers.AsymmetricErrors(
            random_binning_sym, random_binning_sym,
        )
    except KeyError as e:
        _msg = f"Skipping random binning because of KeyError {e}"
        logger.debug(_msg)

    # Untagged bin location
    try:
        unfolding_related_uncertainties[
            "untagged_bin"
        ] = full_results_helpers.AsymmetricErrors.calculate_errors(
            unfolded["untagged_bin"].data.values - unfolded["default"].data.values
        )
    except KeyError as e:
        _msg = f"Skipping untagged bin location because of KeyError {e}"
        logger.debug(_msg)

    # Reweight prior
    try:
        unfolding_related_uncertainties[
            "reweight_prior"
        ] = full_results_helpers.AsymmetricErrors.calculate_errors(
            unfolded["reweight_prior"].data.values - unfolded["default"].data.values
        )
    except KeyError as e:
        _msg = f"Skipping reweighting prior because of KeyError {e}"
        logger.debug(_msg)

    # Now include the unfolding related contributions as appropriate
    if unfolding_related_systematic_treatment == "all":
        unfolded["default"].data.metadata["y_systematic"].update(unfolding_related_uncertainties)
    elif unfolding_related_systematic_treatment == "max":
        # Calculate as max_i=1^N (sigma_i)
        low_max_values = np.zeros_like(unfolded["default"].data.values)
        high_max_values = np.zeros_like(unfolded["default"].data.values)
        for _error_values in unfolding_related_uncertainties.values():
            low_max_values = np.maximum(
                low_max_values,
                np.abs(_error_values.low),
            )
            high_max_values = np.maximum(
                high_max_values,
                np.abs(_error_values.high),
            )
        unfolded["default"].data.metadata["y_systematic"]["unfolding_max"] = full_results_helpers.AsymmetricErrors(
            low_max_values,
            high_max_values,
        )
    elif unfolding_related_systematic_treatment == "std_dev":
        # Calculate asymmetrically as sqrt(1/N * sum_i=1^N (sigma_i)^2)
        low_std_dev_values = np.zeros_like(unfolded["default"].data.values)
        high_std_dev_values = np.zeros_like(unfolded["default"].data.values)
        for _error_values in unfolding_related_uncertainties.values():
            low_std_dev_values += _error_values.low ** 2
            high_std_dev_values += _error_values.high ** 2
        low_std_dev_values = np.sqrt(1./len(unfolding_related_uncertainties) * low_std_dev_values)
        high_std_dev_values = np.sqrt(1./len(unfolding_related_uncertainties) * high_std_dev_values)
        unfolded["default"].data.metadata["y_systematic"]["unfolding_std_dev"] = full_results_helpers.AsymmetricErrors(
            low_std_dev_values,
            high_std_dev_values,
        )
    else:
        _msg = f"Unrecognized treatment of unfolding-related systematics: {unfolding_related_systematic_treatment}"
        raise ValueError(_msg)

    # Background subtraction systematics.
    background_systematics = {}
    if background_subtraction_configuration is not None:
        # (Usual) possible values: ["Rmax070", "Rmax050", "Rmax005"]:
        for background_setting in background_subtraction_configuration.contributors:
            # Since we're explicitly passing the configuration, we should expect that the outputs are there!
            background_systematics[background_setting] = (
                unfolded[background_setting].data.values - unfolded["default"].data.values
            )

        if len(background_systematics) >= 3:
            _msg = f"Found too many background sub systematics - it's ambiguous. Please check! {list(background_systematics.keys())=}"
            raise ValueError(_msg)
        _background_subtraction_values = list(background_systematics.values())
        # NOTE: This is important to append for there to be two values! Otherwise, the uncertainties will
        #       be treated as one sided! If we for some reason only had one side, we would want to duplicate
        #       them since the background can be asymmetric, but they should always be treated as two sided.
        if len(_background_subtraction_values) == 1:
            _background_subtraction_values.append(*_background_subtraction_values)
        unfolded["default"].data.metadata["y_systematic"][
            "background_sub"
        ] = full_results_helpers.AsymmetricErrors.calculate_errors(
            *_background_subtraction_values
        )
        #logger.info(f"Bin edges: {unfolded['default'].data.axes[0].bin_edges}")
        #for _bkg_label, _bkg_value in background_systematics.items():
        #    logger.info(f"{_bkg_label} relative: {_bkg_value / unfolded['default'].data.values}")
        #logger.info(
        #    f'low relative: {unfolded["default"].data.metadata["y_systematic"]["background_sub"].low / unfolded["default"].data.values}\n'
        #    f'high relative: {unfolded["default"].data.metadata["y_systematic"]["background_sub"].high / unfolded["default"].data.values}'
        #)
    else:
        logger.info("Skipping background subtraction because background sub config was not passed.")

    # Non-closure
    # This is treated as a symmetric uncertainty.
    # However, we store it as asymmetric errors objects for consistency with everything else.
    if non_closure_configuration is not None:
        # We loop here since we could there could be multiple contributors to the non-closure.
        non_closure_relative_errors_by_contribution = {}
        for _name in non_closure_configuration.contributors:
            # Intermediate step: Find the asymmetric relative errors of each non-closure variation
            # NOTE: Unlike the others, we take the abs and set the values here directly because
            #       we want them to be symmetric.
            # NOTE: Since the non-closure is generated via a closure (by definition), the reference
            #       needs to be to the PseudoTrue, so we need to retrieve it here.
            # NOTE: We calculate this as a relative error because the scales could be (quite) different.
            #       We then scale the default values by this relative error to determine the non-closure.
            _pseudo_true = unfolding_outputs[_name].true_substructure(
                unfolding_outputs[_name].true_hist_name, true_jet_pt_range=true_jet_pt_range
            )
            non_closure_relative_errors_by_contribution[_name] = np.abs(unfolded[_name].data.values - _pseudo_true.values) / _pseudo_true.values
            #logger.info(f"true name: {unfolding_outputs[_name].true_hist_name}")
            #non_closure_sym_relative =
            #logger.info(f"non_closure values: {unfolded['non_closure'].data.values}")
            #logger.info(f"pseudo true: {pseudo_true.values}")
            ## non_closure_sym = (1 - non_closure_sym_relative) * unfolded["default"].data.values
            #logger.info(f"non_closure_sym_relative: {non_closure_sym_relative}")
            #logger.info(f"non_closure bin edges: {unfolded['non_closure'].data.axes[0].bin_edges}")
            #logger.info(f"pseudo_true bin edges: {pseudo_true.axes[0].bin_edges}")

        # Now calculate the contributions
        if non_closure_configuration.approach_to_combining == "max":
            non_closure_sym_relative = _calculate_max_relative_error_from_contributions(
                relative_uncertainty_by_contribution=non_closure_relative_errors_by_contribution,
                n_values=len(unfolded["default"].data.values),
            )
        elif non_closure_configuration.approach_to_combining == "quadrature":
            non_closure_sym_relative = _calculate_quadrature_relative_error_from_contributions(
                relative_uncertainty_by_contribution=non_closure_relative_errors_by_contribution,
            )
        else:
            _msg = f"Non-closure dependence approach {non_closure_configuration.approach_to_combining} is not recognized and is not implemented."
            raise NotImplementedError(_msg)

        # Treat symmetrically since we don't have an obvious source of this non-closure
        unfolded["default"].data.metadata["y_systematic"]["non_closure"] = full_results_helpers.AsymmetricErrors(
            non_closure_sym_relative * unfolded["default"].data.values,
            non_closure_sym_relative * unfolded["default"].data.values,
        )
    else:
        logger.info("Skipping non closure systematic because no configuration was provided.")

    # Model dependence.
    # The output should include _either_ the model dependence or the prior
    if model_dependence_configuration is not None:
        _entry_for_model_dependence = np.array(["model_dependence" in k for k in unfolding_outputs])
        if model_dependence_configuration.legacy_production:
            # Cross check
            assert np.count_nonzero(_entry_for_model_dependence) == 1
            # This is the original model dependence from Leticia. We had to do special things here because the
            # output was not in our standard format.
            # NOTE: As of 6 Mar 2023, I can't exactly trace back what Leticia gave me, but I can only assume
            #       that it was the difference between the HERWIG and PYTHIA fastsim. Otherwise, I can't make sense
            #       of what I did here. Will handle it properly for the paper since I've re-analyzed these, but am
            #       definitely a bit confused here.
            logger.warning("Handling model dependence from Leticia.")
            # First, extract the model dependence graph
            # NOTE: This output is quite different, so we just need to handle the graph (not hist!) directly.
            graph = unfolding_outputs["model_dependence"].hists[
                f'bayesian_unfolded_iter_{unfolding_outputs["model_dependence"].n_iter_compare}'
            ]

            # Then use the information
            relative_errors_on_model_dependence_low = graph.metadata["y_errors"]["low"] / graph.values
            relative_errors_on_model_dependence_high = graph.metadata["y_errors"]["high"] / graph.values

            #logger.debug(
            #    f"\nmodel_dependence bin_edges: {graph.axes[0].bin_edges}"
            #    f"\nnominal bin_edges: {unfolded['default'].data.axes[0].bin_edges}"
            #)
            unfolded["default"].data.metadata["y_systematic"]["model_dependence"] = full_results_helpers.AsymmetricErrors(
                relative_errors_on_model_dependence_low * unfolded["default"].data.values,
                relative_errors_on_model_dependence_high * unfolded["default"].data.values,
            )
            #logger.debug(
            #    f"\n\tlow (relative): {relative_errors_on_model_dependence_low}"
            #    f"\n\thigh (relative): {relative_errors_on_model_dependence_high}"
            #    f'\n\tmodel_dependence errors: {unfolded["default"].data.metadata["y_systematic"]["model_dependence"]}'
            #)
        elif np.any(_entry_for_model_dependence):
            ###############################################
            # Updated handling for more recent calculations
            ###############################################
            # Validation
            # NOTE: We may only have one if we're comparing to the nominal value, but we may also have
            #       more (eg. fastsim pythia (nominal) vs fastsim herwig (alternative)).
            assert np.count_nonzero(_entry_for_model_dependence) >= 1

            # Retrieve common values and calculate relative uncertainties
            _nominal_name_label = model_dependence_configuration.nominal
            if _nominal_name_label:
                _nominal_name_label = f"_{_nominal_name_label}"

            nominal_values = unfolded[f"model_dependence{_nominal_name_label}"].data.values
            relative_errors_by_model = {}
            # We loop here since we could imagine multiple possible model dependence contributions.
            for model_name in model_dependence_configuration.variations:
                # Intermediate step: Find the asymmetric relative errors of each model variation
                _model_name_label = f"_{model_name}" if model_name else ""
                relative_errors_by_model[model_name] = (
                    unfolded[f"model_dependence{_model_name_label}"].data.values - nominal_values
                ) / nominal_values

            if model_dependence_configuration.approach_to_combining == "max":
                model_dependence_relative = _calculate_max_relative_error_from_contributions(
                    relative_uncertainty_by_contribution=relative_errors_by_model,
                    n_values=len(nominal_values),
                )
            elif model_dependence_configuration.approach_to_combining == "quadrature":
                model_dependence_relative = _calculate_quadrature_relative_error_from_contributions(
                    relative_uncertainty_by_contribution=relative_errors_by_model,
                )
            else:
                _msg = f"Model dependence approach {model_dependence_configuration.approach_to_combining} is not recognized and is not implemented."
                raise NotImplementedError(_msg)

            # Treat asymmetrically since the model goes in a particular direction
            unfolded["default"].data.metadata["y_systematic"][
                "model_dependence"
            ] = full_results_helpers.AsymmetricErrors.calculate_errors(
                # We need the absolute error, so multiply the difference by the default value
                model_dependence_relative * unfolded["default"].data.values,
            )

            #logger.debug(f"Relative error: {model_dependence_relative}")
            #logger.debug(f"{unfolded[f'model_dependence{_nominal_name_label}'].data.axes[0].bin_edges}")
            #logger.debug(f'model_dependence errors: {unfolded["default"].data.metadata["y_systematic"]["model_dependence"]}')
            #logger.debug(
            #    f'\n\tmodel_dependence errors: {unfolded["default"].data.metadata["y_systematic"]["model_dependence"]}'
            #)

    # Smooth selected systematic uncertainties
    # First, a cross check:
    for _smoothed_uncertainty_name in smooth_systematic_uncertainty_contributions:
        if _smoothed_uncertainty_name not in unfolded["default"].data.metadata["y_systematic"]:
            msg = f"Requested to smooth {_smoothed_uncertainty_name}, but it's not available! Cross check this"
            logger.warning(msg)
    # Then, do the actual smoothing
    for _name, _error_values in unfolded["default"].data.metadata["y_systematic"].items():
        if _name in smooth_systematic_uncertainty_contributions:
            _error_values.smooth()

    # Cross check to make sure that I haven't copied and pasted incorrectly.
    assert not any(
        np.allclose(a.low, b.low)
            for k_a, a in unfolded["default"].data.metadata["y_systematic"].items()
            for k_b, b in unfolded["default"].data.metadata["y_systematic"].items()
            if k_a != k_b
    )
    assert not any(
        np.allclose(a.high, b.high)
            for k_a, a in unfolded["default"].data.metadata["y_systematic"].items()
            for k_b, b in unfolded["default"].data.metadata["y_systematic"].items()
            if k_a != k_b
    )

    # Sum in quadrature
    # We protect against including quadrature in case we already calculated the systematics.
    unfolded["default"].data.metadata["y_systematic"]["quadrature"] = full_results_helpers.AsymmetricErrors(
        low=np.sqrt(
            np.sum(
                [
                    v.low ** 2
                    for k, v in unfolded["default"].data.metadata["y_systematic"].items()
                    if k != "quadrature"
                ],
                axis=0,
            )
        ),
        high=np.sqrt(
            np.sum(
                [
                    v.high ** 2
                    for k, v in unfolded["default"].data.metadata["y_systematic"].items()
                    if k != "quadrature"
                ],
                axis=0,
            )
        ),
    )

    # We could already retrieve this from the input, but return it for convenience.
    return unfolded["default"]


def _colors_for_plotting(n_colors: int) -> list[tuple[float, float, float]]:
    colors = []
    if n_colors > 10:
        colors.extend(sns.color_palette("Paired", n_colors=10))
        # "Spectral" and "Set2" are both good options!
        # "Spectral" seems slightly easier to see, so we stick with it
        colors.extend(sns.color_palette("Spectral", n_colors=n_colors - 10))
    else:
        colors.extend(sns.color_palette("Paired", n_colors=n_colors))

    return colors


def plot_unfolded(
    unfolding_output: unfolding_analysis.UnfoldingOutput,
    hist_true: binned_data.BinnedData,
    hist_n_iter_compare: binned_data.BinnedData,
    unfolded_hists: Mapping[int, binned_data.BinnedData],
    plot_config: pb.PlotConfig,
    plot_png: bool = False,
) -> None:
    """Plot unfolded."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")

    # Make it easier to see individual points when necessary
    _jitter = 0.001

    with sns.color_palette(_colors_for_plotting(n_colors=len(unfolded_hists))):
        # Setup
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10, 12),
            gridspec_kw={"height_ratios": [4, 1, 1]},
            sharex=True,
        )
        ax_upper, ax_ratio_iter, ax_ratio_true = axes

        for _plot_counter, (i, hist) in enumerate(unfolded_hists.items()):
            # NOTE: We only apply the jitter in the ratios since that's where we need to see in detail
            _jitter_per_iter = (-1) ** _plot_counter * (_jitter * _plot_counter)
            ax_upper.errorbar(
                hist.axes[0].bin_centers,
                hist.values,
                xerr=hist.axes[0].bin_widths / 2,
                yerr=hist.errors,
                label=f"Bayes {i}",
                marker="o",
                linestyle="",
                alpha=0.8,
                # NOTE: We plot the earliest on top so we can keep better track of the error bars (because
                #       the later iterations have larger error bars).
                # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
                zorder=3 + len(unfolded_hists) - _plot_counter,
            )

            # Plot ratio with selected iter (in principle could also be with true, but now it's
            # not necessary because we have another panel with the true).
            ratio = hist / hist_n_iter_compare
            ax_ratio_iter.errorbar(
                ratio.axes[0].bin_centers + _jitter_per_iter,
                ratio.values,
                xerr=ratio.axes[0].bin_widths / 2,
                yerr=ratio.errors,
                marker="o",
                linestyle="",
                alpha=0.8,
                zorder=3 + len(unfolded_hists) - _plot_counter,
            )

            # Plot ratio with true
            ratio_true = hist / hist_true
            ax_ratio_true.errorbar(
                ratio_true.axes[0].bin_centers + _jitter_per_iter,
                ratio_true.values,
                xerr=ratio_true.axes[0].bin_widths / 2,
                yerr=ratio_true.errors,
                marker="o",
                linestyle="",
                alpha=0.8,
                zorder=3 + len(unfolded_hists) - _plot_counter,
            )

        # Cross check.
        # Plot truth
        ax_upper.errorbar(
            hist_true.axes[0].bin_centers,
            hist_true.values,
            xerr=hist_true.axes[0].bin_widths / 2,
            yerr=hist_true.errors,
            label="True",
            marker="o",
            linestyle="",
            color="black",
            alpha=0.8,
        )
        ## And the ratio too
        # ratio = hist_true / h_ratio_denominator
        # ax_lower.errorbar(
        #    ratio.axes[0].bin_centers,
        #    ratio.values,
        #    xerr=ratio.axes[0].bin_widths / 2,
        #    yerr=ratio.errors,
        #    marker="o",
        #    linestyle="",
        #    color="black",
        #    alpha=0.8,
        # )

        # Plot truth and compare to the full efficient truth.
        ## Compare to the full efficiency to make sure that have the right shape...
        # full_eff_true = projection_func(hists["truef"], true_bin)
        ## Then normalize by the integral (sum) and bin width.
        ## Don't need to correct for the kinematic efficiency here because it's already fully efficient.
        # full_eff_true /= np.sum(full_eff_true.values)
        # full_eff_true /= full_eff_true.axes[0].bin_widths
        # ax_upper.errorbar(full_eff_true.axes[0].bin_centers, full_eff_true.values, xerr=full_eff_true.axes[0].bin_widths / 2, yerr=full_eff_true.errors, label = "True fully eff",
        #                  marker="o", linestyle="", alpha=0.8)
        ## Add ratio...
        # ratio = hist_true / full_eff_true
        # ax_lower.errorbar(
        #    ratio.axes[0].bin_centers,
        #    ratio.values,
        #    xerr=ratio.axes[0].bin_widths / 2,
        #    yerr=ratio.errors,
        #    marker="o",
        #    linestyle="",
        #    alpha=0.8,
        #    color="black",
        # )

        # Draw reference line for ratio
        ax_ratio_iter.axhline(y=1, color="black", linestyle="dashed", zorder=1)
        ax_ratio_true.axhline(y=1, color="black", linestyle="dashed", zorder=1)

        # Label and layout
        # First, tweak the label for the ratio
        true_hist_name_to_ratio_label = {
            "true": "true",
            "h2_pseudo_true": "pseudo true",
        }
        plot_config.panels[2].axes[1].label = (
            plot_config.panels[2]
            .axes[1]
            .label.format(true_label=true_hist_name_to_ratio_label[unfolding_output.true_hist_name])
        )
        plot_config.apply(fig=fig, axes=[ax_upper, ax_ratio_iter, ax_ratio_true])

        figure_name = f"{plot_config.name}"
        logger.info(f"Writing plot to {unfolding_output.output_dir / figure_name}.pdf")
        fig.savefig(unfolding_output.output_dir / f"{figure_name}.pdf")
        if plot_png:
            output_dir_png = unfolding_output.output_dir / "png"
            output_dir_png.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)


def plot_refolded(
    unfolding_output: unfolding_analysis.UnfoldingOutput,
    hist_raw: binned_data.BinnedData,
    hist_smeared: binned_data.BinnedData,
    refolded_hists: Mapping[int, binned_data.BinnedData],
    plot_config: pb.PlotConfig,
    plot_png: bool = False,
) -> None:
    """Plot refolded."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")

    # Make it easier to see individual points when necessary
    _jitter = 0.001

    with sns.color_palette(_colors_for_plotting(n_colors=len(refolded_hists))):
        # Setup
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        ax_upper, ax_lower = axes

        # Raw
        # Only plot if there's something meaningful to plot
        if hist_raw.values.any():
            ax_upper.errorbar(
                hist_raw.axes[0].bin_centers,
                hist_raw.values,
                xerr=hist_raw.axes[0].bin_widths / 2,
                yerr=hist_raw.errors,
                label="Raw",
                marker="o",
                linestyle="",
                color="red",
                # Arbitrarily large
                zorder=49,
            )

        # Smeared
        ax_upper.errorbar(
            hist_smeared.axes[0].bin_centers,
            hist_smeared.values,
            xerr=hist_smeared.axes[0].bin_widths / 2,
            yerr=hist_smeared.errors,
            label="Smeared",
            marker="o",
            linestyle="",
            color="green",
            # Arbitrarily large
            zorder=50,
        )

        raw_is_smeared = unfolding_output.raw_hist_name == "smeared"
        ratio_denominator = hist_smeared if raw_is_smeared else hist_raw
        for _plot_counter, (i, hist) in enumerate(refolded_hists.items()):
            # NOTE: We only apply the jitter in the ratios since that's where we need to see in detail
            _jitter_per_iter = (-1) ** _plot_counter * (_jitter * _plot_counter)

            ax_upper.errorbar(
                hist.axes[0].bin_centers + _jitter_per_iter,
                hist.values,
                xerr=hist.axes[0].bin_widths / 2,
                yerr=hist.errors,
                label=f"Bayes {i}",
                marker="o",
                linestyle="",
                alpha=0.8,
                # NOTE: We plot the earliest on top so we can keep better track of the error bars (because
                #       the later iterations have larger error bars).
                # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
                zorder=3 + len(refolded_hists) - _plot_counter,
            )

            ratio = hist / ratio_denominator
            ax_lower.errorbar(
                ratio.axes[0].bin_centers + _jitter_per_iter,
                ratio.values,
                xerr=ratio.axes[0].bin_widths / 2,
                yerr=ratio.errors,
                marker="o",
                linestyle="",
                alpha=0.8,
                # NOTE: We plot the earliest on top so we can keep better track of the error bars (because
                #       the later iterations have larger error bars).
                # NOTE: Minimum of 3 is important for the error bars to show up on top of points properly
                zorder=3 + len(refolded_hists) - _plot_counter,
            )

        # Add smeared ratio in the right circumstances.
        if not raw_is_smeared:
            r = hist_smeared / ratio_denominator
            ax_lower.errorbar(
                r.axes[0].bin_centers,
                r.values,
                xerr=r.axes[0].bin_widths / 2,
                yerr=r.errors,
                marker="o",
                linestyle="",
                color="green",
                alpha=0.8,
            )

    # Draw reference line for ratio
    ax_lower.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Label and layout
    # First, tweak the label for the ratio
    raw_hist_name_to_ratio_label = {
        "raw": "data",
        "smeared": "smeared",
        "h2_pseudo_data": "pseudo data",
    }
    plot_config.panels[1].axes[1].label = (
        plot_config.panels[1]
        .axes[1]
        .label.format(refold_label=raw_hist_name_to_ratio_label[unfolding_output.raw_hist_name])
    )
    plot_config.apply(fig=fig, axes=[ax_upper, ax_lower])

    figure_name = f"{plot_config.name}"
    # if tag:
    #    figure_name = f"{tag}_{figure_name}"
    fig.savefig(unfolding_output.output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = unfolding_output.output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)


def plot_response(
    hists: Mapping[str, binned_data.BinnedData],
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    # Setup
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")

    h = binned_data.BinnedData.from_existing_data(hists["h2_substructure_variable"])

    # Normalize the response.
    normalization_values = h.values.sum(axis=0, keepdims=True)
    h.values = np.divide(h.values, normalization_values, out=np.zeros_like(h.values), where=normalization_values != 0)

    # Finish setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        # "vmin": h_proj.values[h_proj.values > 0].min(),
        "vmin": max(1e-4, h.values[h.values > 0].min()),
        # "vmax": h.values.max(),
        "vmax": 1,
    }

    # Plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T,
        h.axes[1].bin_edges.T,
        h.values.T,
        norm=mpl.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")
    plt.close(fig)


def plot_jet_pt_vs_substructure(
    hists: Mapping[str, binned_data.BinnedData],
    hist_name: str,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    # Setup
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")

    h = binned_data.BinnedData.from_existing_data(hists[hist_name])

    # Finish setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        "vmin": h.values[h.values > 0].min(),
        "vmax": h.values.max(),
    }

    # Plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T,
        h.axes[1].bin_edges.T,
        h.values.T,
        norm=mpl.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")
    plt.close(fig)


def plot_efficiency(
    hists: Mapping[str, binned_data.BinnedData],
    efficiency_func: Callable[[Mapping[str, binned_data.BinnedData], helpers.RangeSelector], binned_data.BinnedData],
    true_bins: Sequence[helpers.RangeSelector],
    true_bin_label: str,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    """Plot kinematic efficiency."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    for true_bin in true_bins:
        # Project
        # We need the efficiency in the true bin that we actually want to measure.
        hist = efficiency_func(hists, true_bin)

        # Plot
        ax.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            yerr=hist.errors,
            label=fr"${true_bin.min} < {true_bin_label}_{{\text{{T,jet}}}}^{{\text{{true}}}} < {true_bin.max}$",
            marker="o",
            linestyle="",
            alpha=0.8,
        )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    fig.savefig(output_dir / f"{plot_config.name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{plot_config.name}.png")
    plt.close(fig)


def plot_select_iteration(
    unfolding_output: unfolding_analysis.UnfoldingOutput,
    projection_func: Callable[[unfolding_analysis.UnfoldingOutput, int, helpers.RangeSelector], binned_data.BinnedData],
    max_iter: int,
    true_bin: helpers.RangeSelector,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
    prior_variation_output: unfolding_analysis.UnfoldingOutput | None = None,
) -> None:
    """Plot selected iteration."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # -2 because we go two above, and then -2 because we start at 2
    n_bins = max_iter - 2 - 2
    hist_reg = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )
    hist_stat = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )
    hist_prior = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )
    hist_total = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )

    for i, iter in enumerate(range(2, max_iter - 2)):
        # Current iteration
        current_iter_hist = projection_func(unfolding_output, iter, true_bin)
        # Previous iter hist
        previous_iter_hist = projection_func(unfolding_output, iter - 1, true_bin)
        # Iter + 2 hist
        forward_iter_hist = projection_func(unfolding_output, iter + 2, true_bin)

        # Calculate and store regularization error
        regularization_value = np.sum(

                np.divide(
                    np.maximum(
                        np.abs(previous_iter_hist.values - current_iter_hist.values),
                        np.abs(forward_iter_hist.values - current_iter_hist.values),
                    ),
                    # TEMP: Try excluding the untagged bin.
                    # / current_iter_hist.values)[1:]
                    current_iter_hist.values,
                    out=np.zeros_like(current_iter_hist.values),
                    where=current_iter_hist.values != 0,
                )

        )
        hist_reg.values[i] = regularization_value
        # Calculate and store stat error
        # Skip the untagged since it tends to blow up the stat error in a way that's not meaningful
        lower_edge = None if unfolding_output.disabled_untagged_bin else 1
        stat_value = np.sum(
            np.divide(
                current_iter_hist.errors[lower_edge:],
                current_iter_hist.values[lower_edge:],
                out=np.zeros_like(current_iter_hist.values[lower_edge:]),
                where=current_iter_hist.values[lower_edge:] != 0,
            )
        )
        hist_stat.values[i] = stat_value
        # If prior is provided, calculate.
        prior_value = 0
        if prior_variation_output:
            prior = projection_func(prior_variation_output, iter, true_bin)
            prior_value = np.sum(
                np.divide(
                    np.abs(current_iter_hist.values - prior.values),
                    current_iter_hist.values,
                    out=np.zeros_like(current_iter_hist.values),
                    where=current_iter_hist.values != 0,
                )
            )
            hist_prior.values[i] = prior_value

        # Total
        hist_total.values[i] = np.sqrt(regularization_value ** 2 + stat_value ** 2 + prior_value ** 2)

    # Plot the total errors
    ax.errorbar(
        hist_total.axes[0].bin_centers,
        hist_total.values,
        xerr=hist_total.axes[0].bin_widths / 2,
        label="Total",
        marker="o",
        linestyle="",
    )
    # The regularization errors
    ax.errorbar(
        hist_reg.axes[0].bin_centers,
        hist_reg.values,
        xerr=hist_reg.axes[0].bin_widths / 2,
        label="Regularization",
        marker="o",
        linestyle="",
    )
    # Plot the stat errors
    ax.errorbar(
        hist_stat.axes[0].bin_centers,
        hist_stat.values,
        xerr=hist_stat.axes[0].bin_widths / 2,
        label="Statistical",
        marker="o",
        linestyle="",
    )
    # And the prior values, if they were provided
    if prior_variation_output:
        ax.errorbar(
            hist_prior.axes[0].bin_centers,
            hist_prior.values,
            xerr=hist_prior.axes[0].bin_widths / 2,
            label="Prior",
            marker="o",
            linestyle="",
        )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)
    # Additional tweaks
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2.0))

    figure_name = f"{plot_config.name}"
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")
    plt.close(fig)


def plot_kt_unfolding(
    unfolding_output: unfolding_analysis.UnfoldingOutput,
    plot_png: bool = False,
    prior_variation_output: unfolding_analysis.UnfoldingOutput | None = None,
    unfolding_kt_display_range: tuple[float, float] | None = None,
) -> Path:
    if unfolding_kt_display_range is None:
        unfolding_kt_display_range = (-0.5, unfolding_output.smeared_var_range.max)
    logger.info(f"Plotting {unfolding_output.identifier}")

    # Main unfolded plot.
    true_jet_pt_range = helpers.JetPtRange(60, 80)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_substructure(
            unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_substructure(
            unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
            for n_iter in unfolding_output.n_iter_range_to_plot()
            # for n_iter in range(1, unfolding_output.n_iter_compare + 5)
        },
        plot_config=pb.PlotConfig(
            name=f"unfolded_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"$\text{{d}}N/\text{{d}}k_{{\text{{T}}}}\:(\text{{GeV}}/c)^{{-1}}$",  # noqa: F541
                            log=True,
                            range=(8e-4, None),
                        )
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
                pb.Panel(
                    axes=[
                        # pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(-0.5, 15)),
                        # Take advantage of the smeared and true level substructure var being the same range.
                        pb.AxisConfig(
                            "x",
                            label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
                            range=unfolding_kt_display_range,
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Check a broader true jet pt range: 40-120
    true_jet_pt_range = helpers.JetPtRange(40, 120)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_substructure(
            unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_substructure(
            unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
            # for n_iter in unfolding_output.n_iter_range_to_plot()
            for n_iter in range(1, unfolding_output.n_iter_compare + 5)
        },
        plot_config=pb.PlotConfig(
            name=f"unfolded_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"$\text{{d}}N/\text{{d}}k_{{\text{{T}}}}\:(\text{{GeV}}/c)^{{-1}}$",  # noqa: F541
                            log=True,
                            range=(1e-4, None),
                        )
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
                pb.Panel(
                    axes=[
                        # Take advantage of the smeared and true level substructure var being the same range.
                        pb.AxisConfig(
                            "x",
                            label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
                            range=unfolding_kt_display_range,
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Check a higher jet pt bin: 80-100
    true_jet_pt_range = helpers.JetPtRange(80, 100)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_substructure(
            unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_substructure(
            unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
            for n_iter in unfolding_output.n_iter_range_to_plot()
            # for n_iter in range(1, unfolding_output.n_iter_compare + 5)
        },
        plot_config=pb.PlotConfig(
            name=f"unfolded_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"$\text{{d}}N/\text{{d}}k_{{\text{{T}}}}\:(\text{{GeV}}/c)^{{-1}}$",  # noqa: F541
                            log=True,
                            range=(1e-3, None),
                        )
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
                pb.Panel(
                    axes=[
                        # Take advantage of the smeared and true level substructure var being the same range.
                        pb.AxisConfig(
                            "x",
                            label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
                            range=unfolding_kt_display_range,
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Unfolded jet pt
    # First, over the full kt range.
    true_substructure_variable_range = helpers.KtRange(-1, 100)
    text = f"${true_substructure_variable_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_jet_pt(
            unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
            unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_jet_pt(
                n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
            )
            # for n_iter in unfolding_output.n_iter_range_to_plot()
            for n_iter in range(1, unfolding_output.n_iter_compare + 5)
        },
        plot_config=pb.PlotConfig(
            name="unfolded_pt",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Since our smeared and true kt ranges usually match, we'll restrict it here.
    # NOTE: Careful here, this doesn't actually apply for the main semi-central and central ranges...
    true_substructure_variable_range = unfolding_output.smeared_var_range  # type: ignore[assignment]
    text = f"${true_substructure_variable_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_jet_pt(
            unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
            unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_jet_pt(
                n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
            )
            # for n_iter in unfolding_output.n_iter_range_to_plot()
            for n_iter in range(1, unfolding_output.n_iter_compare + 5)
        },
        plot_config=pb.PlotConfig(
            # Display with f"unfolded_pt_true_{unfolding_output.smeared_var_range}"
            name=f"unfolded_pt_true_{true_substructure_variable_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )

    # Now, on to the refolded.
    text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='data')}$"
    plot_refolded(
        unfolding_output=unfolding_output,
        hist_raw=unfolding_output.smeared_substructure(
            hist_name=unfolding_output.raw_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
        ),
        hist_smeared=unfolding_output.smeared_substructure(
            hist_name=unfolding_output.smeared_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
        ),
        refolded_hists={
            n_iter: unfolding_output.refolded_substructure(
                n_iter=n_iter, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
            )
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name=f"refolded_{unfolding_output.substructure_variable}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2, anchor=(0.025, 0.025)),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                        # y label is set in the function.
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {refold_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Jet pt
    text = f"${unfolding_output.smeared_var_range.display_str(label='data')}$"
    plot_refolded(
        unfolding_output=unfolding_output,
        hist_raw=unfolding_output.smeared_jet_pt(
            hist_name=unfolding_output.raw_hist_name,
            smeared_substructure_variable_range=unfolding_output.smeared_var_range,
        ),
        hist_smeared=unfolding_output.smeared_jet_pt(
            hist_name=unfolding_output.smeared_hist_name,
            smeared_substructure_variable_range=unfolding_output.smeared_var_range,
        ),
        refolded_hists={
            n_iter: unfolding_output.refolded_jet_pt(
                n_iter=n_iter, smeared_substructure_variable_range=unfolding_output.smeared_var_range
            )
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name="refolded_pt",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="upper right", ncol=2, anchor=(0.975, 0.90)),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                        # y label is set in the function.
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {refold_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )

    # Slice the refolded in jet pt just to get a sense of what they look like.
    # pp
    # Apparently I need to go to 90 to avoid slicing issues. I thought it would peak at 85...?
    _small_jet_pt_bins = np.array([unfolding_output.smeared_jet_pt_range.min, 30, 40, 50, 60, 90])
    if unfolding_output.collision_system != "pp":
        # PbPb needs somewhat different binning
        _small_jet_pt_bins = np.array([unfolding_output.smeared_jet_pt_range.min, 60, 80, 100, 120])
    for _low, _high in zip(_small_jet_pt_bins[:-1], _small_jet_pt_bins[1:]):
        _small_jet_pt_range = helpers.JetPtRange(_low, _high)
        text = f"${_small_jet_pt_range.display_str(label='data')}$"
        plot_refolded(
            unfolding_output=unfolding_output,
            hist_raw=unfolding_output.smeared_substructure(
                hist_name=unfolding_output.raw_hist_name, smeared_jet_pt_range=_small_jet_pt_range
            ),
            hist_smeared=unfolding_output.smeared_substructure(
                hist_name=unfolding_output.smeared_hist_name, smeared_jet_pt_range=_small_jet_pt_range
            ),
            refolded_hists={
                n_iter: unfolding_output.refolded_substructure(
                    n_iter=n_iter, smeared_jet_pt_range=_small_jet_pt_range
                )
                for n_iter in unfolding_output.n_iter_range_to_plot()
            },
            plot_config=pb.PlotConfig(
                name=f"refolded_{unfolding_output.substructure_variable}_{_small_jet_pt_range.histogram_str(label='smeared')}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y", label=r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True
                            )
                        ],
                        legend=pb.LegendConfig(location="lower left", ncol=2, anchor=(0.025, 0.025)),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                            # y label is set in the function.
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {refold_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding={"bottom": 0.06}),
            ),
            plot_png=plot_png,
        )

    # Plot the response
    if "h2_substructure_variable" in unfolding_output.hists:
        text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='hybrid')}$"
        plot_response(
            hists=unfolding_output.hists,
            plot_config=pb.PlotConfig(
                name=f"response_{unfolding_output.substructure_variable}_hybrid_{unfolding_output.smeared_jet_pt_range}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                        # Use the smeared variable max value as a proxy for the max true value of interest.
                        pb.AxisConfig(
                            "y",
                            label=r"$k_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$",
                            range=(0, unfolding_output.smeared_var_range.max),
                        ),
                    ],
                    text=pb.TextConfig(text, 0.97, 0.03),
                ),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
        )

    # Plot kt vs jet pt
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="smeared",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_hybrid",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    # True
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="true",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_true",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$", range=(None, 20)),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    # Select the n_iter iteration
    for true_jet_pt_range in [helpers.JetPtRange(60, 80), helpers.JetPtRange(80, 100)]:
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_select_iteration(
            unfolding_output=unfolding_output,
            projection_func=unfolding_analysis.UnfoldingOutput.unfolded_substructure,  # type: ignore[arg-type]
            max_iter=unfolding_output.max_n_iter,
            true_bin=true_jet_pt_range,
            plot_config=pb.PlotConfig(
                name=f"select_iteration_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label="Iteration"),
                        pb.AxisConfig("y", label="Summed Error", range=(0, None)),
                    ],
                    legend=pb.LegendConfig(location="center right"),
                    text=pb.TextConfig(text, 0.03, 0.03),
                ),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
            prior_variation_output=prior_variation_output,
        )

    # Efficiency
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=unfolding_analysis.efficiency_substructure_variable,
        true_bins=[
            helpers.JetPtRange(40, 120),
            helpers.JetPtRange(40, 60),
            helpers.JetPtRange(60, 80),
            helpers.JetPtRange(80, 100),
            helpers.JetPtRange(80, 120),
        ],
        true_bin_label="p",
        plot_config=pb.PlotConfig(
            name=f"efficiency_{unfolding_output.substructure_variable}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", log=True),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower left"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    # Cleaned up kt efficiency, focused on the ranges that we will measure.
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=unfolding_analysis.efficiency_substructure_variable,
        true_bins=[
            helpers.JetPtRange(60, 80),
        ],
        true_bin_label="p",
        plot_config=pb.PlotConfig(
            name=f"efficiency_{unfolding_output.substructure_variable}_true_pt_60_80",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=unfolding_kt_display_range),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower left"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=unfolding_analysis.efficiency_pt,
        true_bins=[
            unfolding_output.smeared_var_range,
            # helpers.RangeSelector(unfolding_output.smeared_var_range.min, unfolding_output.smeared_var_range.max),
            # helpers.RangeSelector(1, 15),
            # helpers.RangeSelector(2, 13),
            # helpers.RangeSelector(2, 15),
        ],
        true_bin_label="k",
        plot_config=pb.PlotConfig(
            name="efficiency_pt",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower right"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    # plot_spectra_comparison(hists, output_dir)
    # plot_spectra_comparison_fine_binned(hists, output_dir)
    # plot_response_matrix(hists["responseUnscaled"], "response", output_dir)

    return unfolding_output.output_dir


def run(collision_system: str) -> None:
    base_dir = Path("output")
    for unfolding_output in [
        ###################### kt smeared = 2-10 ##########################
        ## 2-10, 1-2, 30-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        #    smeared_input=True,
        # ),
        ## 2-10, 1-2, 40-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ## 2-10, 10-13, 30-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        #    smeared_input=True,
        # ),
        ## 2-10, 10-13, 40-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ###################### kt smeared = 3-10 ##########################
        ## 3-10, 2-3, 30-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    smeared_input=True,
        # ),
        # 3-10, 2-3, 40-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ## 3-10, 10-13, 30-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    smeared_input=True,
        # ),
        ## 3-10, 10-13, 40-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        # 3-10, 2-3, 40-120, pure matches
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    pure_matches=True,
        #    n_iter_compare=11,
        #    max_iter=15,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    pure_matches=True,
        #    n_iter_compare=11,
        #    max_iter=15,
        #    smeared_input=True,
        # ),
        ###################### kt smeared = 3-10, broad true bins ##########################
        ## 3-10, 2-3, 30-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    suffix="broadTrueBins",
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    suffix="broadTrueBins",
        #    smeared_input=True,
        # ),
        ## 3-10, 2-3, 40-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    suffix="broadTrueBins",
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    suffix="broadTrueBins",
        #    smeared_input=True,
        # ),
        ## 3-11, 30-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    smeared_input=True,
        # ),
        ## 3-11, 40-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    smeared_input=True,
        # ),
        ## 3-15, 30-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    smeared_input=True,
        # ),
        ## 3-15, 40-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    pure_matches=True,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    pure_matches=True,
        #    smeared_input=True,
        # ),
        ###################### kt smeared = 5-15 ##########################
        ## 4-15, 3-4, 40-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(4, 15),
        #    smeared_untagged_var=helpers.KtRange(3, 4),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(4, 15),
        #    smeared_untagged_var=helpers.KtRange(3, 4),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ## 5-15, 4-5, 40-120
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(5, 15),
        #    smeared_untagged_var=helpers.KtRange(4, 5),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # unfolding_configuration.UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(5, 15),
        #    smeared_untagged_var=helpers.KtRange(4, 5),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ####### Dynamical kt ##########
        # 3-15, 2-3, 40-120
        unfolding_analysis.UnfoldingOutput(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        unfolding_analysis.UnfoldingOutput(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        # 2-15, 1-2, 30-120
        unfolding_analysis.UnfoldingOutput(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(2, 15),
            smeared_untagged_var=helpers.KtRange(1, 2),
            smeared_jet_pt_range=helpers.JetPtRange(30, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        unfolding_analysis.UnfoldingOutput(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(2, 15),
            smeared_untagged_var=helpers.KtRange(1, 2),
            smeared_jet_pt_range=helpers.JetPtRange(30, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        ####### Dynamical time ##########
        # 3-15, 2-3, 40-120
        unfolding_analysis.UnfoldingOutput(
            "kt",
            "dynamical_time",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        unfolding_analysis.UnfoldingOutput(
            "kt",
            "dynamical_time",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        ####### Leading kt ##########
        # 3-15, 2-3, 40-120
        unfolding_analysis.UnfoldingOutput(
            "kt",
            "leading_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        unfolding_analysis.UnfoldingOutput(
            "kt",
            "leading_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
    ]:
        plot_kt_unfolding(unfolding_output=unfolding_output)


def plot_delta_R_unfolding(
    unfolding_output: unfolding_analysis.UnfoldingOutput,
    plot_png: bool = False,
    prior_variation_output: unfolding_analysis.UnfoldingOutput | None = None,
    unfolding_var_display_range: tuple[float, float] | None = None,
) -> Path:
    if unfolding_var_display_range is None:
        unfolding_var_display_range = (-0.5, unfolding_output.smeared_var_range.max)
    logger.info(f"Plotting {unfolding_output.identifier}")

    true_jet_pt_range = helpers.JetPtRange(60, 80)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_substructure(
            unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_substructure(
            unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name=f"unfolded_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$\text{d}N/\text{d}R_{\text{g}}$",
                        )
                    ],
                    # legend=pb.LegendConfig(location="lower left"),
                    legend=pb.LegendConfig(location="center right", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$R_{\text{g}}$",
                            range=unfolding_var_display_range,
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Check a broader true jet pt range: 40-120
    true_jet_pt_range = helpers.JetPtRange(40, 120)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_substructure(
            unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_substructure(
            unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name=f"unfolded_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$\text{d}N/\text{d}R_{\text{g}}$",
                        )
                    ],
                    legend=pb.LegendConfig(location="center right", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$R_{\text{g}}$",
                            range=unfolding_var_display_range
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Check a higher jet pt bin: 80-100
    true_jet_pt_range = helpers.JetPtRange(80, 100)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_substructure(
            unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_substructure(
            unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name=f"unfolded_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$\text{d}N/\text{d}R_{\text{g}}$",
                        )
                    ],
                    legend=pb.LegendConfig(location="center right", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$R_{\text{g}}$",
                            range=unfolding_var_display_range
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Unfolded jet pt
    # First, over the full kt range.
    true_substructure_variable_range = helpers.RgRange(-1, 100)
    text = f"${true_substructure_variable_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_jet_pt(
            unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
            unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_jet_pt(
                n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
            )
            #for n_iter in unfolding_output.n_iter_range_to_plot()
            for n_iter in range(1, unfolding_output.n_iter_compare + 5)
        },
        plot_config=pb.PlotConfig(
            name="unfolded_pt",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Since our smeared and true Rg ranges can match, we'll restrict it here.
    # NOTE: Careful here, this doesn't actually apply for the main semi-central and central ranges...
    true_substructure_variable_range = unfolding_output.smeared_var_range  # type: ignore[assignment]
    text = f"${true_substructure_variable_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_jet_pt(
            unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
            unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_jet_pt(
                n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
            )
            #for n_iter in unfolding_output.n_iter_range_to_plot()
            for n_iter in range(1, unfolding_output.n_iter_compare + 5)
        },
        plot_config=pb.PlotConfig(
            # Display with f"unfolded_pt_true_{unfolding_output.smeared_var_range}"
            name=f"unfolded_pt_true_{true_substructure_variable_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )

    # Now, on to the refolded.
    text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='data')}$"
    plot_refolded(
        unfolding_output=unfolding_output,
        hist_raw=unfolding_output.smeared_substructure(
            hist_name=unfolding_output.raw_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
        ),
        hist_smeared=unfolding_output.smeared_substructure(
            hist_name=unfolding_output.smeared_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
        ),
        refolded_hists={
            n_iter: unfolding_output.refolded_substructure(
                n_iter=n_iter, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
            )
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name=f"refolded_{unfolding_output.substructure_variable}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[pb.AxisConfig("y", label=r"$\text{d}N/\text{d}R_{\text{g}}$")],
                    legend=pb.LegendConfig(location="lower center", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$R_{\text{g}}$"),
                        # y label is set in the function.
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {refold_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Jet pt
    text = f"${unfolding_output.smeared_var_range.display_str(label='data')}$"
    plot_refolded(
        unfolding_output=unfolding_output,
        hist_raw=unfolding_output.smeared_jet_pt(
            hist_name=unfolding_output.raw_hist_name,
            smeared_substructure_variable_range=unfolding_output.smeared_var_range,
        ),
        hist_smeared=unfolding_output.smeared_jet_pt(
            hist_name=unfolding_output.smeared_hist_name,
            smeared_substructure_variable_range=unfolding_output.smeared_var_range,
        ),
        refolded_hists={
            n_iter: unfolding_output.refolded_jet_pt(
                n_iter=n_iter, smeared_substructure_variable_range=unfolding_output.smeared_var_range
            )
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name="refolded_pt",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="upper right", ncol=2, anchor=(0.975, 0.90)),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                        # y label is set in the function.
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {refold_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # NOTE: Going to skip slicing in the refolded here since I think we don't need
    #       to do such exploration...

    # Plot the response
    if "h2_substructure_variable" in unfolding_output.hists:
        text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='hybrid')}$"
        plot_response(
            hists=unfolding_output.hists,
            plot_config=pb.PlotConfig(
                name=f"response_{unfolding_output.substructure_variable}_hybrid_{unfolding_output.smeared_jet_pt_range}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$R_{\text{g}}^{\text{hybrid}}$",
                            range=(0, unfolding_output.smeared_var_range.max),
                        ),
                        pb.AxisConfig("y", label=r"$R_{\text{g}}^{\text{true}}$",
                            range=(0, unfolding_output.smeared_var_range.max),
                        ),
                    ],
                    text=pb.TextConfig(text, 0.97, 0.03),
                ),
                figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.10}),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
        )

    # Plot Rg vs jet pt
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="smeared",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_hybrid",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$R_{\text{g}}^{\text{hybrid}}$"),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    # True
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="true",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_true",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$R_{\text{g}}^{\text{true}}$"),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    # Select the n_iter iteration
    for true_jet_pt_range in [helpers.JetPtRange(60, 80), helpers.JetPtRange(80, 100)]:
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_select_iteration(
            unfolding_output=unfolding_output,
            projection_func=unfolding_analysis.UnfoldingOutput.unfolded_substructure,  # type: ignore[arg-type]
            max_iter=unfolding_output.max_n_iter,
            true_bin=true_jet_pt_range,
            plot_config=pb.PlotConfig(
                name=f"select_iteration_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label="Iteration"),
                        pb.AxisConfig("y", label="Summed Error", range=(0, None)),
                    ],
                    legend=pb.LegendConfig(location="center right"),
                    text=pb.TextConfig(text, 0.03, 0.03),
                ),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
            prior_variation_output=prior_variation_output,
        )

    # Efficiency
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=unfolding_analysis.efficiency_substructure_variable,
        true_bins=[
            helpers.JetPtRange(40, 120),
            helpers.JetPtRange(40, 60),
            helpers.JetPtRange(60, 80),
            helpers.JetPtRange(80, 100),
            helpers.JetPtRange(80, 120),
        ],
        true_bin_label="p",
        plot_config=pb.PlotConfig(
            name=f"efficiency_{unfolding_output.substructure_variable}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$R_{\text{g}}$"),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower left"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    # Cleaned up Rg efficiency, focused on the ranges that we will measure.
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=unfolding_analysis.efficiency_substructure_variable,
        true_bins=[
            helpers.JetPtRange(60, 80),
        ],
        true_bin_label="p",
        plot_config=pb.PlotConfig(
            name=f"efficiency_{unfolding_output.substructure_variable}_true_pt_60_80",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$R_{\text{g}}$", range=unfolding_var_display_range),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower left"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=unfolding_analysis.efficiency_pt,
        true_bins=[
            unfolding_output.smeared_var_range,
            # helpers.RangeSelector(unfolding_output.smeared_var_range.min, unfolding_output.smeared_var_range.max),
            # helpers.RangeSelector(1, 15),
            # helpers.RangeSelector(2, 13),
            # helpers.RangeSelector(2, 15),
        ],
        true_bin_label="Rg",
        plot_config=pb.PlotConfig(
            name="efficiency_pt",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower right"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    return unfolding_output.output_dir


def run_delta_R(collision_system: str) -> None:
    base_dir = Path("output")
    for unfolding_output in [
        unfolding_analysis.UnfoldingOutput(
            "delta_R",
            "leading_kt_z_cut_02",
            # Hack until the labeling is fixed...
            smeared_var_range=helpers.RgRange(0, 350),
            smeared_untagged_var=helpers.RgRange(-50, 0),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        unfolding_analysis.UnfoldingOutput(
            "delta_R",
            "leading_kt_z_cut_02",
            # Hack until the labeling is fixed...
            smeared_var_range=helpers.RgRange(0, 350),
            smeared_untagged_var=helpers.RgRange(-50, 0),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
    ]:
        plot_delta_R_unfolding(unfolding_output=unfolding_output)


def plot_z_unfolding(
    unfolding_output: unfolding_analysis.UnfoldingOutput,
    plot_png: bool = False,
    prior_variation_output: unfolding_analysis.UnfoldingOutput | None = None,
    unfolding_var_display_range: tuple[float, float] | None = None,
) -> Path:
    if unfolding_var_display_range is None:
        unfolding_var_display_range = (-0.5, unfolding_output.smeared_var_range.max)
    logger.info(f"Plotting {unfolding_output.identifier}")

    true_jet_pt_range = helpers.JetPtRange(60, 80)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_substructure(
            unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_substructure(
            unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name=f"unfolded_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$\text{d}N/\text{d}z_{\text{g}}$",
                        )
                    ],
                    # legend=pb.LegendConfig(location="lower left"),
                    legend=pb.LegendConfig(location="center right", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$z_{\text{g}}$",
                            range=unfolding_var_display_range,
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Check a broader true jet pt range: 40-120
    true_jet_pt_range = helpers.JetPtRange(40, 120)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_substructure(
            unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_substructure(
            unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name=f"unfolded_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$\text{d}N/\text{d}z_{\text{g}}$",
                        )
                    ],
                    legend=pb.LegendConfig(location="center right", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$z_{\text{g}}$",
                            range=unfolding_var_display_range
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Check a higher jet pt bin: 80-100
    true_jet_pt_range = helpers.JetPtRange(80, 100)
    text = f"${true_jet_pt_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_substructure(
            unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_substructure(
            unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name=f"unfolded_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$\text{d}N/\text{d}z_{\text{g}}$",
                        )
                    ],
                    legend=pb.LegendConfig(location="center right", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            range=(0.5, 1.5),
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$z_{\text{g}}$",
                            range=unfolding_var_display_range
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Unfolded jet pt
    # First, over the full z range.
    true_substructure_variable_range = helpers.ZgRange(-1, 0.6)
    text = f"${true_substructure_variable_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_jet_pt(
            unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
            unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_jet_pt(
                n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
            )
            #for n_iter in unfolding_output.n_iter_range_to_plot()
            for n_iter in range(1, unfolding_output.n_iter_compare + 5)
        },
        plot_config=pb.PlotConfig(
            name="unfolded_pt",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Since our smeared and true zg ranges can match, we'll restrict it here.
    # NOTE: Careful here, this doesn't actually apply for the main semi-central and central ranges...
    true_substructure_variable_range = unfolding_output.smeared_var_range  # type: ignore[assignment]
    text = f"${true_substructure_variable_range.display_str(label='true')}$"
    plot_unfolded(
        unfolding_output=unfolding_output,
        hist_true=unfolding_output.true_jet_pt(
            unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
        ),
        hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
            unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
        ),
        unfolded_hists={
            n_iter: unfolding_output.unfolded_jet_pt(
                n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
            )
            #for n_iter in unfolding_output.n_iter_range_to_plot()
            for n_iter in range(1, unfolding_output.n_iter_compare + 5)
        },
        plot_config=pb.PlotConfig(
            # Display with f"unfolded_pt_true_{unfolding_output.smeared_var_range}"
            name=f"unfolded_pt_true_{true_substructure_variable_range!s}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="lower left", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                        )
                    ],
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {true_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )

    # Now, on to the refolded.
    text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='data')}$"
    plot_refolded(
        unfolding_output=unfolding_output,
        hist_raw=unfolding_output.smeared_substructure(
            hist_name=unfolding_output.raw_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
        ),
        hist_smeared=unfolding_output.smeared_substructure(
            hist_name=unfolding_output.smeared_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
        ),
        refolded_hists={
            n_iter: unfolding_output.refolded_substructure(
                n_iter=n_iter, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
            )
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name=f"refolded_{unfolding_output.substructure_variable}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[pb.AxisConfig("y", label=r"$\text{d}N/\text{d}z_{\text{g}}$")],
                    legend=pb.LegendConfig(location="lower center", ncol=2),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$z_{\text{g}}$"),
                        # y label is set in the function.
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {refold_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # Jet pt
    text = f"${unfolding_output.smeared_var_range.display_str(label='data')}$"
    plot_refolded(
        unfolding_output=unfolding_output,
        hist_raw=unfolding_output.smeared_jet_pt(
            hist_name=unfolding_output.raw_hist_name,
            smeared_substructure_variable_range=unfolding_output.smeared_var_range,
        ),
        hist_smeared=unfolding_output.smeared_jet_pt(
            hist_name=unfolding_output.smeared_hist_name,
            smeared_substructure_variable_range=unfolding_output.smeared_var_range,
        ),
        refolded_hists={
            n_iter: unfolding_output.refolded_jet_pt(
                n_iter=n_iter, smeared_substructure_variable_range=unfolding_output.smeared_var_range
            )
            for n_iter in unfolding_output.n_iter_range_to_plot()
        },
        plot_config=pb.PlotConfig(
            name="refolded_pt",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                    ],
                    legend=pb.LegendConfig(location="upper right", ncol=2, anchor=(0.975, 0.90)),
                    text=pb.TextConfig(text, 0.97, 0.97),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                        # y label is set in the function.
                        pb.AxisConfig(
                            "y",
                            label="Ratio to {refold_label}",
                            range=(0.5, 1.5),
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding={"bottom": 0.06}),
        ),
        plot_png=plot_png,
    )
    # NOTE: Going to skip slicing in the refolded here since I think we don't need
    #       to do such exploration...

    # Plot the response
    if "h2_substructure_variable" in unfolding_output.hists:
        text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='hybrid')}$"
        plot_response(
            hists=unfolding_output.hists,
            plot_config=pb.PlotConfig(
                name=f"response_{unfolding_output.substructure_variable}_hybrid_{unfolding_output.smeared_jet_pt_range}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$z_{\text{g}}^{\text{hybrid}}$",
                            range=(0, unfolding_output.smeared_var_range.max),
                        ),
                        pb.AxisConfig("y", label=r"$z_{\text{g}}^{\text{true}}$",
                            range=(0, unfolding_output.smeared_var_range.max),
                        ),
                    ],
                    text=pb.TextConfig(text, 0.97, 0.03),
                ),
                figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.10}),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
        )

    # Plot zg vs jet pt
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="smeared",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_hybrid",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$z_{\text{g}}^{\text{hybrid}}$"),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    # True
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="true",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_true",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$z_{\text{g}}^{\text{true}}$"),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    # Select the n_iter iteration
    for true_jet_pt_range in [helpers.JetPtRange(60, 80), helpers.JetPtRange(80, 100)]:
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_select_iteration(
            unfolding_output=unfolding_output,
            projection_func=unfolding_analysis.UnfoldingOutput.unfolded_substructure,  # type: ignore[arg-type]
            max_iter=unfolding_output.max_n_iter,
            true_bin=true_jet_pt_range,
            plot_config=pb.PlotConfig(
                name=f"select_iteration_{unfolding_output.substructure_variable}_true_{true_jet_pt_range!s}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label="Iteration"),
                        pb.AxisConfig("y", label="Summed Error", range=(0, None)),
                    ],
                    legend=pb.LegendConfig(location="center right"),
                    text=pb.TextConfig(text, 0.03, 0.03),
                ),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
            prior_variation_output=prior_variation_output,
        )

    # Efficiency
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=unfolding_analysis.efficiency_substructure_variable,
        true_bins=[
            helpers.JetPtRange(40, 120),
            helpers.JetPtRange(40, 60),
            helpers.JetPtRange(60, 80),
            helpers.JetPtRange(80, 100),
            helpers.JetPtRange(80, 120),
        ],
        true_bin_label="p",
        plot_config=pb.PlotConfig(
            name=f"efficiency_{unfolding_output.substructure_variable}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$z_{\text{g}}$"),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower left"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    # Cleaned up zg efficiency, focused on the ranges that we will measure.
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=unfolding_analysis.efficiency_substructure_variable,
        true_bins=[
            helpers.JetPtRange(60, 80),
        ],
        true_bin_label="p",
        plot_config=pb.PlotConfig(
            name=f"efficiency_{unfolding_output.substructure_variable}_true_pt_60_80",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$z_{\text{g}}$", range=unfolding_var_display_range),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower left"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=unfolding_analysis.efficiency_pt,
        true_bins=[
            unfolding_output.smeared_var_range,
            # helpers.RangeSelector(unfolding_output.smeared_var_range.min, unfolding_output.smeared_var_range.max),
            # helpers.RangeSelector(1, 15),
            # helpers.RangeSelector(2, 13),
            # helpers.RangeSelector(2, 15),
        ],
        true_bin_label="Zg",
        plot_config=pb.PlotConfig(
            name="efficiency_pt",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower right"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    return unfolding_output.output_dir

def steer_plotting_of_substructure_var_unfolding_outputs(  # noqa: C901
    substructure_variable: str,
    grooming_methods: list[str],
    unfolded_with_systematics: dict[str, unfolding_analysis.SingleResult],
    unfolding_systematics_outputs: dict[str, dict[str, unfolding_analysis.UnfoldingOutput]],
    unfolding_closure_outputs: dict[str, dict[str, unfolding_analysis.UnfoldingOutput]],
    plot: bool,
    plot_png: bool,
    plot_systematic_breakdown: bool,
    plot_systematics: bool,
    plot_closures: bool,
    unfolding_display_range: dict[str, tuple[float, float]] | tuple[float, float],
    prior_variation_output_name: dict[str, str | None] | str | None = None,
    relative_individual_systematic_ratio_range: dict[str, tuple[float, float]] | tuple[float, float] | None = None,
) -> None:
    # Validation
    if isinstance(unfolding_display_range, tuple):
        unfolding_display_range = {grooming_method: unfolding_display_range for grooming_method in grooming_methods}
    if isinstance(prior_variation_output_name, str) or prior_variation_output_name is None:
        prior_variation_output_name = {grooming_method: prior_variation_output_name for grooming_method in grooming_methods}
    if isinstance(relative_individual_systematic_ratio_range, tuple) or relative_individual_systematic_ratio_range is None:
        relative_individual_systematic_ratio_range = {
            grooming_method: relative_individual_systematic_ratio_range if relative_individual_systematic_ratio_range is not None else (0.5, 1.5)
            for grooming_method in grooming_methods
        }

    # Skip out immediately if we can
    if not plot:
        return

    # Setup
    if substructure_variable == "kt":
        axis_label = r"k_{\text{T}}"
    elif substructure_variable == "delta_R":
        axis_label = r"R_{\text{g}}"
    elif substructure_variable == "z":
        axis_label = r"z_{\text{g}}"
    else:
        msg = f"Unknown substructure variable: {substructure_variable}"
        raise ValueError(msg)

    for grooming_method in grooming_methods:
        logger.info(f"Plotting '{grooming_method}'")
        if plot_systematic_breakdown:
            # Plot the individual relative systematics
            plot_relative_individual_systematics(
                unfolded=unfolded_with_systematics[grooming_method],
                plot_config=pb.PlotConfig(
                    name="unfolded_systematic_relative",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$" + axis_label + r"\:(\text{GeV}/c)$",
                                    range=unfolding_display_range[grooming_method],
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label="Relative error",
                                    range=relative_individual_systematic_ratio_range[grooming_method],
                                ),
                            ],
                            legend=pb.LegendConfig(location="upper right", ncol=2),
                            #text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                    ],
                ),
                output_dir=unfolding_systematics_outputs[grooming_method]["default"].output_dir,
                plot_png=plot_png,
            )

        _prior_variation_name = prior_variation_output_name[grooming_method]
        if _prior_variation_name is not None:
            _prior_variation_output = unfolding_systematics_outputs[grooming_method][_prior_variation_name]
        else:
            _prior_variation_output = None
        if substructure_variable == "delta_R":
            plot_delta_R_unfolding(
                unfolding_output=unfolding_systematics_outputs[grooming_method]["default"],
                plot_png=plot_png,
                prior_variation_output=_prior_variation_output,
                unfolding_var_display_range=unfolding_display_range[grooming_method],
            )
        elif substructure_variable == "z":
            plot_z_unfolding(
                unfolding_output=unfolding_systematics_outputs[grooming_method]["default"],
                plot_png=plot_png,
                prior_variation_output=_prior_variation_output,
                unfolding_var_display_range=unfolding_display_range[grooming_method],
            )

        _unfolding_func = plot_delta_R_unfolding
        if substructure_variable == "z":
            _unfolding_func = plot_z_unfolding
        for _outputs in [
            unfolding_closure_outputs if plot_closures else {grooming_method: {}},
            unfolding_systematics_outputs if plot_systematics else {grooming_method: {}},
        ]:
            for name, _unfolding_output in _outputs[grooming_method].items():
                # Skip, since we already plotted above.
                if name == "default":
                    continue
                _unfolding_func(
                    unfolding_output=_unfolding_output,
                    plot_png=plot_png,
                    unfolding_var_display_range=unfolding_display_range[grooming_method],
                )


def steer_plotting_of_kt_unfolding_outputs(
    grooming_methods: list[str],
    unfolded_with_systematics: dict[str, unfolding_analysis.SingleResult],
    unfolding_systematics_outputs: dict[str, dict[str, unfolding_analysis.UnfoldingOutput]],
    unfolding_closure_outputs: dict[str, dict[str, unfolding_analysis.UnfoldingOutput]],
    plot: bool,
    plot_png: bool,
    plot_systematic_breakdown: bool,
    plot_systematics: bool,
    plot_closures: bool,
    unfolding_kt_display_range: dict[str, tuple[float, float]] | tuple[float, float],
    unfolding_related_systematic_treatment: str,
    prior_variation_output_name: dict[str, str | None] | str | None = None,
    relative_individual_systematic_ratio_range: dict[str, tuple[float, float]] | tuple[float, float] | None = None,
) -> None:
    # Validation
    if isinstance(unfolding_kt_display_range, tuple):
        unfolding_kt_display_range = {grooming_method: unfolding_kt_display_range for grooming_method in grooming_methods}
    if isinstance(prior_variation_output_name, str) or prior_variation_output_name is None:
        prior_variation_output_name = {grooming_method: prior_variation_output_name for grooming_method in grooming_methods}
    if isinstance(relative_individual_systematic_ratio_range, tuple) or relative_individual_systematic_ratio_range is None:
        relative_individual_systematic_ratio_range = {
            grooming_method: relative_individual_systematic_ratio_range if relative_individual_systematic_ratio_range is not None else (0.5, 1.5)
            for grooming_method in grooming_methods
        }

    # Skip out immediately if we can
    if not plot:
        return

    for grooming_method in grooming_methods:
        logger.info(f"Plotting '{grooming_method}'")
        if plot_systematic_breakdown:
            # Plot the individual relative systematics
            plot_relative_individual_systematics(
                unfolded=unfolded_with_systematics[grooming_method],
                plot_config=pb.PlotConfig(
                    name=f"unfolded_systematic_relative__unfolding_treatment_{unfolding_related_systematic_treatment}",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
                                    range=unfolding_kt_display_range[grooming_method],
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label="Relative error",
                                    range=relative_individual_systematic_ratio_range[grooming_method],
                                ),
                            ],
                            legend=pb.LegendConfig(location="upper right", ncol=2),
                            #text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                    ],
                ),
                output_dir=unfolding_systematics_outputs[grooming_method]["default"].output_dir,
                plot_png=plot_png,
            )

        _prior_variation_name = prior_variation_output_name[grooming_method]
        if _prior_variation_name is not None:
            _prior_variation_output = unfolding_systematics_outputs[grooming_method][_prior_variation_name]
        else:
            _prior_variation_output = None
        plot_kt_unfolding(
            unfolding_output=unfolding_systematics_outputs[grooming_method]["default"],
            plot_png=plot_png,
            prior_variation_output=_prior_variation_output,
            unfolding_kt_display_range=unfolding_kt_display_range[grooming_method],
        )
        for _outputs in [
            unfolding_closure_outputs if plot_closures else {grooming_method: {}},
            unfolding_systematics_outputs if plot_systematics else {grooming_method: {}},
        ]:
            for name, _unfolding_output in _outputs[grooming_method].items():
                # Skip, since we already plotted above.
                if name == "default":
                    continue
                plot_kt_unfolding(
                    unfolding_output=_unfolding_output,
                    plot_png=plot_png,
                    unfolding_kt_display_range=unfolding_kt_display_range[grooming_method],
                )


if __name__ == "__main__":
    # Setup
    helpers.setup_logging()
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
    collision_system = "PbPb"

    # Enable ticks on all sides
    # Unfortunately, some of this is overriding the pachyderm plotting style.
    # That will have to be updated eventually...
    # matplotlib.rcParams["xtick.top"] = True
    # matplotlib.rcParams["xtick.minor.top"] = True
    # matplotlib.rcParams["ytick.right"] = True
    # matplotlib.rcParams["ytick.minor.right"] = True

    run(collision_system=collision_system)
    # run_delta_R(collision_system=collision_system)
