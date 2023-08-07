# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Substructure w/ ROOT 6.28.04, conda
#     language: python
#     name: substructure_c_28_04
# ---

# # Paper plots for Rg + zg for pp, semi-central, central
#

# +
from __future__ import annotations

# Setup
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pachyderm.plot as pb
from mammoth import helpers as mammoth_helpers

from jet_substructure.analysis import (
    model_calculations,
    plot_paper,
    plot_unfolding,
    unfolding_analysis,
)
from jet_substructure.base import helpers

# %load_ext autoreload
# %autoreload 2

# #%matplotlib inline
# #%config InlineBackend.figure_formats = ["png", "pdf"]
# Don't show mpl images inline. We'll handle displaying them separately.
plt.ioff()
# Ensure the axes are legible on a dark background
mpl.rcParams['figure.facecolor'] = 'w'

mammoth_helpers.setup_logging(level=logging.DEBUG)
# Quiet down the matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
logging.getLogger("boost_histogram").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# General settings
embed_images = False
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
# -

# # Load data

# ## R = 0.2

# +
# General setup
plot = False
substructure_variable = "delta_R"
true_jet_pt_range = helpers.JetPtRange(60, 80)
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
grooming_methods = [
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
    #"soft_drop_z_cut_02",
    #"dynamical_core_z_cut_02",
    #"dynamical_kt_z_cut_02",
    #"dynamical_time_z_cut_02",
    #"soft_drop_z_cut_04",
]
_OG_grooming_methods = [
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
    "soft_drop_z_cut_02",
]
_new_grooming_methods = [
    "dynamical_core_z_cut_02",
    "dynamical_kt_z_cut_02",
    "dynamical_time_z_cut_02",
    "soft_drop_z_cut_04",
]
input_dir_tag = "2023-paper"

###################
# Setup I/O options
###################
_use_qm22_inputs = False
_grooming_methods_using_qm_result_conventions = _OG_grooming_methods if _use_qm22_inputs else []
_grooming_methods_using_new_conventions = _new_grooming_methods if _use_qm22_inputs else grooming_methods

_output_dir = output_dir / "comparison" / "unfolding" / "2023-paper-plots" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)
# -

# ### pp

# + tags=["remove_cell"]
# pp
collision_system = "pp"
event_activity = "pp"
_smeared_var_range: helpers.RgRange | dict[str, helpers.RgRange] = helpers.RgRange(0., 0.2)
# NOTE: Using Laura's jet pt range, which is different from my usual one in pp
#_smeared_jet_pt_range = helpers.JetPtRange(20, 85)
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_truncation_shift = 3
_displaced_extremum = 0.4
#_tag_after_suffix = "2_4_split"

###################
# Setup I/O options
###################
# Input directory location
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
_input_dir_tag = {
    _method: "2023-paper"
    for _method in _grooming_methods_using_qm_result_conventions
}
_input_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
_output_dir_tag: dict[str, str | None] = {
    _method: "2023-paper-plots"
    for _method in _grooming_methods_using_qm_result_conventions
}
_output_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
_tag_after_suffix = {
    grooming_method: "" for grooming_method in grooming_methods
}
#_tag_after_suffix["soft_drop_z_cut_04"] = "merge_3_6"

####################################
# Grooming method dependent settings
####################################
_smeared_untagged_var = {
    "dynamical_core": helpers.RgRange(-0.05, 0),
    "dynamical_kt": helpers.RgRange(-0.05, 0),
    "dynamical_time": helpers.RgRange(-0.05, 0),
    # IDK as of July 2023
    "soft_drop_z_cut_02": helpers.RgRange(-0.05, 0),
    "soft_drop_z_cut_04": helpers.RgRange(-0.05, 0),
    "dynamical_core_z_cut_02": helpers.RgRange(-0.05, 0),
    "dynamical_kt_z_cut_02": helpers.RgRange(-0.05, 0),
    "dynamical_time_z_cut_02": helpers.RgRange(-0.05, 0),
}
_n_iter_compare = {
    "dynamical_core": 3,
    "dynamical_kt": 3,
    "dynamical_time": 3,
    "soft_drop_z_cut_02": 3,
    # Not yet optimized...
    "soft_drop_z_cut_04": 3,
    "dynamical_core_z_cut_02": 3,
    "dynamical_kt_z_cut_02": 3,
    "dynamical_time_z_cut_02": 3,
}
_max_n_iter: dict[str, int | None] = {
    # Need +1 for convenience with range iteration
    "soft_drop_z_cut_04": 30,
}
_max_n_iter.update({
    grooming_method: 20 for grooming_method in grooming_methods if grooming_method != "soft_drop_z_cut_04"
})


# Model dependence.
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
_model_dependence_configuration: dict[str, unfolding_analysis.ModelDependenceConfiguration | None] = {
    _method: unfolding_analysis.ModelDependenceConfiguration(
        nominal="pythia_fastsim",
        variations=["herwig_fastsim"],
    ) for _method in _grooming_methods_using_new_conventions
}
# Non-closure
non_closure_configuration = {
    _method: None
    for _method in _grooming_methods_using_new_conventions
}

# Either take model dependence or reweighted prior
# Model dependence is always preferred, but it may not have been analyzed yet for the a particular configuration
# (or in PbPb, it likely isn't possible since we don't have a reliable MC)
skip_reweighted_prior_in_systematics = True

# + tags=["remove_cell"]
pp_R02_unfolding_closure_outputs, pp_R02_unfolding_closure_pure_matches_outputs, pp_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    max_n_iter=_max_n_iter,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    input_dir_tag=_input_dir_tag,
    output_dir_tag=_output_dir_tag,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
    skip_reweighted_prior_in_systematics=skip_reweighted_prior_in_systematics,
    model_dependence_configuration=_model_dependence_configuration,
)

# Focus down onto just the unfolded distributions
pp_R02_unfolded_with_systematics, pp_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=pp_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=pp_R02_unfolding_closure_outputs,
    true_jet_pt_range=true_jet_pt_range,
    model_dependence_configuration=_model_dependence_configuration,
    non_closure_configuration=non_closure_configuration,
    background_subtraction_configuration=None,
)
# -

pp_R02_unfolded_with_systematics["dynamical_core"]

# + jupyter={"outputs_hidden": true} tags=["remove_cell"]
#plot = True
#if plot:
#    for grooming_method in grooming_methods:
#
#        # Plot the individual relative systematics
#        plot_unfolding.plot_relative_individual_systematics(
#            unfolded=pp_R02_unfolded_with_systematics[grooming_method],
#            plot_config=pb.PlotConfig(
#                name="unfolded_systematic_relative",
#                panels=[
#                    pb.Panel(
#                        axes=[
#                            pb.AxisConfig("x", label=r"$R_{\text{g}}$", range=(-0.1, 0.3)),
#                            pb.AxisConfig(
#                                "y",
#                                label="Relative error",
#                                range=[0.5, 1.5],
#                            ),
#                        ],
#                        legend=pb.LegendConfig(location="upper right", ncol=2),
#                        #text=pb.TextConfig(text, 0.97, 0.97),
#                    ),
#                ],
#            ),
#            output_dir = pp_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
#            plot_png = True,
#        )
#        
#        plot_unfolding.plot_delta_R_unfolding(
#            unfolding_output=pp_R02_unfolding_systematics_outputs[grooming_method]["default"],
#            plot_png=True,
#            #reweighted_prior_output=pp_R02_unfolding_systematics_outputs[grooming_method]["model_dependence"],
#            reweighted_prior_output=pp_R02_unfolding_systematics_outputs[grooming_method]["reweight_prior"],
#            unfolding_Rg_display_range=(-0.05, 0.2),
#        )
#        ##for _outputs in [pp_R02_unfolding_closure_outputs, pp_R02_unfolding_closure_pure_matches_outputs, pp_R02_unfolding_systematics_outputs]:
#        #for _outputs in [pp_R02_unfolding_closure_outputs, pp_R02_unfolding_systematics_outputs]:
#        for _outputs in [pp_R02_unfolding_closure_outputs]:
#            for name, _unfolding_output in _outputs[grooming_method].items():
#                # Skip, since we already plotted above.
#                if name == "default":
#                    continue
#                plot_unfolding.plot_delta_R_unfolding(
#                    unfolding_output=_unfolding_output,
#                    plot_png=True,
#                    #unfolding_kt_display_range=(0.5, 6) if "z_cut" not in grooming_method else (0.25, 6),
#                    unfolding_Rg_display_range=(-0.05, 0.2),
#                )


## Plot full systematics for multiple grooming methods on one plot.
#if plot:
#    plot_unfolding.plot_PbPb_systematics(
#        hists=pp_R02_unfolded_with_systematics,
#        reference=pp_R02_true_reference,
#        grooming_methods=grooming_methods,
#        event_activity=event_activity,
#        kt_range=[2, 15],
#        # Arbitrarily take the first grooming method for the output dir
#        output_dir=pp_R02_unfolding_systematics_outputs[grooming_methods[0]]["default"].output_dir,
#    )

plot_unfolding.steer_plotting_of_substructure_var_unfolding_outputs(
    substructure_variable="delta_R",
    grooming_methods=grooming_methods,
    unfolded_with_systematics=pp_R02_unfolded_with_systematics,
    unfolding_systematics_outputs=pp_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=pp_R02_unfolding_closure_outputs,
    plot=True,
    plot_png=False,
    plot_systematic_breakdown=True,
    plot_systematics=False,
    plot_closures=False,
    # NOTE: For the prior variation, passing the HERwIG model dependence includes both:
    #       - HERWIG vs PYTHIA
    #       - fastsim vs full sim as well as whatever HERWIG
    #       Consequently, the fastsim output may not be the most accurate overall magnitude, but we can
    #       still use it to look at the shape for selecting the iteration. Alternatively, we can switch
    #       back to the reweighted_prior, but that output is less satisfying for pp.
    # NOTE: We can't remove the fastsim vs full sim dependence at the moment because we would need
    #       the full UnfoldingOutput object, which we don't have available since the model dependence here
    #       is constructed by transferring the differences from the fastsim outputs to the default.
    #       We could do this, but it's more tricky (eg. can refolded be treated the same way?
    #       Probably, but would need to be checked), so we just stick with the HERWIG model dependence.
    prior_variation_output_name="model_dependence_herwig_fastsim",
    #prior_variation_output_name="reweight_prior",
    unfolding_display_range={
        grooming_method: (-0.05, 0.2)
        for grooming_method in grooming_methods
    },
    relative_individual_systematic_ratio_range={
        grooming_method: (0.5, 1.5) for grooming_method in grooming_methods
    }
)
# -



# ### Semi-central

# + tags=["remove_cell"]
collision_system = "PbPb"
event_activity = "semi_central"
_tag_after_suffix = "pass3"
#_tag_after_suffix = "pass3_peter_binning"
_double_counting_cut = "min_true_10_pt_hat_3"
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_truncation_shift = 5
_displaced_extremum = 0.4
_smeared_var_range = helpers.RgRange(0.04, 0.2)

# Grooming method dependent settings
_smeared_untagged_var = {
    "dynamical_core": helpers.RgRange(-0.05, 0),
    "dynamical_kt": helpers.RgRange(-0.05, 0),
    "dynamical_time": helpers.RgRange(-0.05, 0),
    "soft_drop_z_cut_02": helpers.RgRange(-0.05, 0),
    "soft_drop_z_cut_04": helpers.RgRange(-0.05, 0),
    "dynamical_core_z_cut_02": helpers.RgRange(-0.05, 0),
    "dynamical_kt_z_cut_02": helpers.RgRange(-0.05, 0),
    "dynamical_time_z_cut_02": helpers.RgRange(-0.05, 0),
}
_n_iter_compare = {
    "dynamical_core": 3,
    "dynamical_kt": 3,
    "dynamical_time": 3,
    "soft_drop_z_cut_02": 3,
    "dynamical_core_z_cut_02": 3,
    "dynamical_kt_z_cut_02": 3,
    "dynamical_time_z_cut_02": 3,
    # TODO: May need to increase on account of how long it takes to converge.
    "soft_drop_z_cut_04": 3,
}
# -

from importlib import reload
reload(plot_unfolding)

# + tags=["remove_cell"]
# Initially load data
semi_central_R02_unfolding_closure_outputs, semi_central_R02_unfolding_closure_pure_matches_outputs, semi_central_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    input_dir_tag=input_dir_tag,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
    double_counting_cut=_double_counting_cut,
)

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    semi_central_R02_unfolding_systematics_outputs[grooming_method]["non_closure"] = \
        semi_central_R02_unfolding_closure_outputs[grooming_method]["reweight_response"]
        #semi_central_R02_unfolding_closure_outputs[grooming_method]["reweight_pseudo_data"]

# Focus down onto just the unfolded distributions
semi_central_R02_unfolded_with_systematics, semi_central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=semi_central_R02_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)
# -

list(semi_central_R02_unfolding_systematics_outputs["dynamical_kt_z_cut_02"].keys())

# + jupyter={"outputs_hidden": true} tags=["remove_cell"]
plot = True
plot_png = False
if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=semi_central_R02_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$R_{\text{g}}$", range=(-0.1, 0.3)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.5, 1.5],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir=semi_central_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png=plot_png,
        )
        for k in semi_central_R02_unfolding_closure_outputs[grooming_method]:
            print(f"Plotting {k} for {grooming_method}")
            plot_unfolding.plot_delta_R_unfolding(
                unfolding_output=semi_central_R02_unfolding_closure_outputs[grooming_method][k],
                plot_png=plot_png,
                reweighted_prior_output=semi_central_R02_unfolding_systematics_outputs[grooming_method]["reweight_prior"] if k == "default" else None,
                unfolding_Rg_display_range=(-0.05, 0.2),
            )
        #for k in semi_central_R02_unfolding_systematics_outputs[grooming_method]:
        #    if "Rmax" not in k:
        #        continue
        #    plot_unfolding.plot_kt_unfolding(
        #        unfolding_output=semi_central_R02_unfolding_systematics_outputs[grooming_method][k],
        #        plot_png=True,
        #        #reweighted_prior_output=semi_central_R02_unfolding_systematics_outputs[grooming_method]["reweight_prior"] if k == "default" else None,
        #        unfolding_kt_display_range=(0.5, 6) if "soft_drop" not in grooming_method else (0.25, 6),
        #    )
# -

list(semi_central_R02_unfolded_with_systematics.keys())

# ### Central

# + tags=["remove_cell"]
collision_system = "PbPb"
event_activity = "central"
_tag_after_suffix = "pass3"
#_tag_after_suffix = "pass3_peter_binning"
_double_counting_cut = "min_true_10_pt_hat_3"
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_truncation_shift = 5
_displaced_extremum = 0.4
_smeared_var_range = helpers.RgRange(0, 0.2)

# Grooming method dependent settings
_smeared_untagged_var = {
    "dynamical_core": helpers.RgRange(-0.05, 0),
    "dynamical_kt": helpers.RgRange(-0.05, 0),
    "dynamical_time": helpers.RgRange(-0.05, 0),
    "soft_drop_z_cut_02": helpers.RgRange(-0.05, 0),
    "soft_drop_z_cut_04": helpers.RgRange(-0.05, 0),
    "dynamical_core_z_cut_02": helpers.RgRange(-0.05, 0),
    "dynamical_kt_z_cut_02": helpers.RgRange(-0.05, 0),
    "dynamical_time_z_cut_02": helpers.RgRange(-0.05, 0),
}
_n_iter_compare = {
    "dynamical_core": 10,
    "dynamical_kt": 10,
    "dynamical_time": 10,
    "soft_drop_z_cut_02": 10,
    "dynamical_core_z_cut_02": 10,
    "dynamical_kt_z_cut_02": 10,
    "dynamical_time_z_cut_02": 10,
    "soft_drop_z_cut_04": 10,
}

# + tags=["remove_cell"]
# Initially load data
central_R02_unfolding_closure_outputs, central_R02_unfolding_closure_pure_matches_outputs, central_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    input_dir_tag=input_dir_tag,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
    double_counting_cut=_double_counting_cut,
)

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    central_R02_unfolding_systematics_outputs[grooming_method]["non_closure"] = \
        central_R02_unfolding_closure_outputs[grooming_method]["reweight_response"]

# Focus down onto just the unfolded distributions
central_R02_unfolded_with_systematics, central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=central_R02_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)
# -

plot = True
plot_png = False
if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=central_R02_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$R_{\text{g}}$", range=(-0.1, 0.3)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.5, 1.5],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir=central_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png=plot_png,
        )
        
        for k in central_R02_unfolding_closure_outputs[grooming_method]:
            print(f"Plotting {k} for {grooming_method}")
            plot_unfolding.plot_delta_R_unfolding(
                unfolding_output=central_R02_unfolding_closure_outputs[grooming_method][k],
                plot_png=plot_png,
                reweighted_prior_output=central_R02_unfolding_systematics_outputs[grooming_method]["reweight_prior"] if k == "default" else None,
                unfolding_Rg_display_range=(-0.05, 0.2),
            )

print(list(central_R02_unfolded_with_systematics.keys()))
print(list(central_R02_unfolding_closure_outputs["dynamical_core"].keys()))
print(list(central_R02_unfolding_systematics_outputs["dynamical_core"].keys()))

# ## R = 0.4

# +
# General setup
plot = False
substructure_variable = "kt"
true_jet_pt_range = helpers.JetPtRange(60, 80)
jet_R = 0.4
jet_R_str = f"R{int(jet_R*10):02}"
grooming_methods = ["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"]

_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)
# -

# ### pp

# + tags=["remove_cell"]
# pp
collision_system = "pp"
event_activity = "pp"
_smeared_var_range = helpers.KtRange(0.25, 8)
_smeared_jet_pt_range = helpers.JetPtRange(20, 85)
#_tag_after_suffix = "2_4_split"
_truncation_shift = 3
_displaced_extremum = 20

# Grooming method dependent settings
_smeared_untagged_var = {
    "dynamical_core": helpers.KtRange(0.25, 0.25),
    "dynamical_kt": helpers.KtRange(0.25, 0.25),
    "dynamical_time": helpers.KtRange(0.25, 0.25),
    "soft_drop_z_cut_02": helpers.KtRange(0, 0.25),
}
# TODO: To be checked!
_n_iter_compare = {
    "dynamical_core": 5,
    "dynamical_kt": 5,
    "dynamical_time": 5,
    "soft_drop_z_cut_02": 5,
}

# TODO: Should be changed to True once we add the model dependency...
skip_reweighted_prior_in_systematics = True

# + tags=["remove_cell"]
# Initially load data
pp_R04_unfolding_closure_outputs, pp_R04_unfolding_closure_pure_matches_outputs, pp_R04_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    output_dir=output_dir,
    #tag_after_suffix=_tag_after_suffix,
    skip_reweighted_prior_in_systematics=skip_reweighted_prior_in_systematics,
)
#for grooming_method in grooming_methods:
#    print(f"running {grooming_method}")

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    pp_R04_unfolding_systematics_outputs[grooming_method]["non_closure"] = pp_R04_unfolding_closure_outputs[grooming_method]["reweight_response"]

# Focus down onto just the unfolded distributions
pp_R04_unfolded_with_systematics, pp_R04_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=pp_R04_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)
    
if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=pp_R04_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0.5, 6)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.5, 1.5],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir = pp_R04_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png = True,
        )
        
        #plot_unfolding.plot_kt_unfolding(
        #    unfolding_output=semi_central_R02_unfolding_systematics_outputs[grooming_method]["default"],
        #    plot_png=True,
        #    reweighted_prior_output=semi_central_R02_unfolding_systematics_outputs[grooming_method]["reweight_prior"],
        #    unfolding_kt_display_range=(0.5, 6),
        #)
#
## Plot full systematics for multiple grooming methods on one plot.
#if plot:
#    plot_unfolding.plot_PbPb_systematics(
#        hists=pp_R04_unfolded_with_systematics,
#        reference=pp_R04_true_reference,
#        grooming_methods=grooming_methods,
#        event_activity=event_activity,
#        kt_range=[2, 15],
#        # Arbitrarily take the first grooming method for the output dir
#        output_dir=pp_R04_unfolding_systematics_outputs[grooming_methods[0]]["default"].output_dir,
#    )
# -

list(pp_R04_unfolded_with_systematics.keys())

# ## Models

# ### Pythia
#
# Already loaded in the "true_reference" variables

# ### Jetscape
#
# Includes both R = 0.2 and R = 0.4

jetscape_pp = plot_unfolding.load_jetscape_data(
    output_dir / "comparison" / "models" / "jetscape" / "5020_PP_Colorless" / "AnalysisResultsFinal.root"
)
jetscape_central = plot_unfolding.load_jetscape_data(
    output_dir / "comparison" / "models" / "jetscape" / "5020_PbPb_0-10_0.30_2.0_1" / "AnalysisResultsFinal.root"
)

# ### Sherpa

sherpa_ahadic = plot_unfolding.load_sherpa_predictions(
    filename = output_dir / "comparison" / "models" / "sherpa" / "SherpaHistograms_Ahadic_R04_merged12.root", jet_R_values=[0.2, 0.4]
)
sherpa_lund = plot_unfolding.load_sherpa_predictions(
    filename = output_dir / "comparison" / "models" / "sherpa" / "SherpaHistograms_Lund_R04_merged12.root", jet_R_values=[0.2, 0.4]
)

# ### Analytical Calculations

_bin_edges = {
    # NOTE: Equivalent to [0.5, 1, 2, 4, 6, 8]
    "R04": pp_R04_unfolded_with_systematics["dynamical_kt"].data.axes[0].bin_edges[2:-1]
}
analytical_pp = plot_unfolding.load_analytical_calculations(
    path_to_calculations= output_dir / "comparison" / "models" / "analytical", bin_edges=_bin_edges,
) 

# + [markdown] toc-hr-collapsed=true
# ### Hybrid model
# -

from importlib import reload
reload(plot_unfolding)

_bin_edges = {
    #"R02": {
    #    grooming_method: semi_central_R02_unfolded_with_systematics[grooming_method].data.axes[0].bin_edges[:]
    #    for grooming_method in grooming_methods
    #}
    "R02": {
        "soft_drop_z_cut_02": [0.25, 0.5, 1, 1.5, 2, 3, 4, 6],
        "dynamical_core": [2, 3, 4, 6],
        "dynamical_kt": [2, 3, 4, 6],
        "dynamical_time": [2, 3, 4, 6],
    }
}
semi_central_hybrid_model_ratio = plot_unfolding.load_hybrid_model(
    base_dir=output_dir / "comparison" / "models" / "hybrid" / "ForRaymond_kT" / "kT_raymond",
    bin_edges=_bin_edges,
    jet_R=jet_R_str,
    jet_pt_bin=true_jet_pt_range,
) 

_d05 = semi_central_hybrid_model_ratio["hybrid_moliere"]["R02"]["dynamical_core"].values, semi_central_hybrid_model_ratio["hybrid_moliere"]["R02"]["dynamical_core"].errors
_d1 = semi_central_hybrid_model_ratio["hybrid_moliere"]["R02"]["dynamical_kt"].values, semi_central_hybrid_model_ratio["hybrid_moliere"]["R02"]["dynamical_kt"].errors
print(_d05, _d1)

# # Plots

grooming_methods = ["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"]

# ## pp grooming methods comparison

# ### R = 0.2

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_unfolding.plot_grooming_comparisons_for_single_system(
    hists=pp_R02_unfolded_with_systematics,
    grooming_methods=grooming_methods,
    reference_grooming_method="dynamical_core",
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 6),
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
)
# -

# ### R = 0.4

# + jupyter={"outputs_hidden": true}
jet_R = 0.4
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_unfolding.plot_grooming_comparisons_for_single_system(
    hists=pp_R04_unfolded_with_systematics,
    grooming_methods=grooming_methods,
    reference_grooming_method="dynamical_core",
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 8),
    figure_kt_range=helpers.KtRange(0, 8.25),
    jet_R_str=jet_R_str,
)
# -

# ## pp comparisons to models

# ### R = 0.2

reload(plot_unfolding)

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_unfolding.plot_grooming_model_comparisons_for_single_system(
    hists=pp_R02_unfolded_with_systematics,
    models={
        # TODO: Need to update the jetscape binning. Whoops...
        #"jetscape": jetscape_pp["R02"],
        "pythia": pp_R02_true_reference,
    },
    grooming_methods=grooming_methods,
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 6),
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
)
# -

# ### R = 0.4

# + jupyter={"outputs_hidden": true}
jet_R = 0.4
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_unfolding.plot_grooming_model_comparisons_for_single_system(
    hists=pp_R04_unfolded_with_systematics,
    models={
        # TODO: Need to update the jetscape binning. Whoops...
        #"jetscape": jetscape_pp["R04"],
        "pythia": pp_R04_true_reference,
    },
    grooming_methods=grooming_methods,
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 8),
    figure_kt_range=helpers.KtRange(0, 8.25),
    jet_R_str=jet_R_str,
)
# -

# ## PbPb grooming methods comparison

reload(plot_unfolding)

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / input_dir_tag / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_unfolding.plot_grooming_comparisons_for_single_system(
    hists=semi_central_R02_unfolded_with_systematics,
    grooming_methods=grooming_methods,
    reference_grooming_method="dynamical_kt_z_cut_02",
    collision_system="semi_central",
    collision_system_key="PbPb",
    output_dir=_output_dir,
    kt_range={
        "dynamical_core": helpers.KtRange(2, 6),
        "dynamical_kt": helpers.KtRange(2, 6),
        "dynamical_time": helpers.KtRange(2, 6),
        "soft_drop_z_cut_02": helpers.KtRange(0.25, 6),
        "soft_drop_z_cut_04": helpers.KtRange(0.25, 6),
        "dynamical_core_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_kt_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_time_z_cut_02": helpers.KtRange(0.25, 6),
    },
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
)

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)
_temp_grooming_methods = grooming_methods
grooming_methods = ["soft_drop_z_cut_02", "dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"]

plot_unfolding.plot_grooming_comparisons_for_single_system(
    hists=central_R02_unfolded_with_systematics,
    grooming_methods=grooming_methods,
    reference_grooming_method="dynamical_core",
    collision_system="central",
    collision_system_key="PbPb",
    output_dir=_output_dir,
    kt_range={
        "dynamical_core": helpers.KtRange(3, 6),
        "dynamical_kt": helpers.KtRange(3, 6),
        # TODO: These will need to be revised
        "dynamical_time": helpers.KtRange(2, 6),
        "soft_drop_z_cut_02": helpers.KtRange(0.25, 6),
    },
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
)
# Undoes the temporary restriction of grooming methods due to the data which is available
grooming_methods = _temp_grooming_methods
# -

# ## PbPb comparisons to models

# NOTE: Semi-central doesn't have a meaningful model comparison yet. Still waiting on JS and analytical calculations

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_unfolding.plot_grooming_model_comparisons_for_single_system(
    hists=semi_central_R02_unfolded_with_systematics,
    models={
        # NOTE: Jetscape isn't yet available for semi-central
        "jetscape": jetscape_semi_central["R02"],
    },
    grooming_methods=grooming_methods,
    collision_system="semi_central",
    collision_system_key="PbPb",
    output_dir=_output_dir,
    kt_range={
        "dynamical_core": helpers.KtRange(2, 6),
        "dynamical_kt": helpers.KtRange(2, 6),
        # TODO: This may need to be revised
        "dynamical_time": helpers.KtRange(2, 6),
        "soft_drop_z_cut_02": helpers.KtRange(0.25, 6),
    },
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
)

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)
_temp_grooming_methods = grooming_methods
grooming_methods = ["dynamical_core", "dynamical_kt"]

plot_unfolding.plot_grooming_model_comparisons_for_single_system(
    hists=central_R02_unfolded_with_systematics,
    models={
        # NOTE: Jetscape isn't yet available for semi-central
        "jetscape": jetscape_central["R02"],
    },
    grooming_methods=grooming_methods,
    collision_system="central",
    collision_system_key="PbPb",
    output_dir=_output_dir,
    kt_range={
        "dynamical_core": helpers.KtRange(3, 6),
        "dynamical_kt": helpers.KtRange(3, 6),
        # TODO: These will need to be revised
        "dynamical_time": helpers.KtRange(3, 6),
        "soft_drop_z_cut_02": helpers.KtRange(0.25, 6),
    },
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
)
# Undoes the temporary restriction of grooming methods due to the data which is available
grooming_methods = _temp_grooming_methods
# -

# ## Compare collision systems

from importlib import reload
reload(plot_unfolding)

# TODO: Need to make this more configurable...
#for grooming_method in ["soft_drop_z_cut_02"]:
for grooming_method in grooming_methods:
    plot_unfolding.plot_pp_PbPb_comparison(
        hists={
            "pp": pp_R02_unfolded_with_systematics[grooming_method],
            "semi_central": semi_central_R02_unfolded_with_systematics[grooming_method],
            #"central": central_R02_unfolded_with_systematics[grooming_method],
        },
        grooming_method=grooming_method,
        output_dir=_output_dir,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": helpers.KtRange(2, 6) if "soft_drop" not in grooming_method else helpers.KtRange(0.25, 6),
        },
        kt_display_range=(0.0, 6.25),
        jet_R_str=jet_R_str,
    )





# ## Paper plots

# ### pp grooming method comparison with models

# #### R = 0.2

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_paper.plot_pp_grooming_comparison_with_models(
    hists=pp_R02_unfolded_with_systematics,
    grooming_methods=grooming_methods,
    reference_grooming_method="soft_drop_z_cut_02",
    models={
        "jetscape": jetscape_pp["R02"],
        "pythia": pp_R02_true_reference,
        #"sherpa_ahadic": sherpa_ahadic["R02"],
        #"sherpa_lund": sherpa_lund["R02"],
    },
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 6),
    kt_ranges_for_models={
        # TODO: Update jetscape range when binning is fixed.
        #       For now, overlap between 2 and 6
        "jetscape": helpers.KtRange(2, 6),
        "pythia": helpers.KtRange(0.25, 6),
        "sherpa_ahadic": helpers.KtRange(0.25, 6),
        "sherpa_lund": helpers.KtRange(0.25, 6),
    },
    models_to_normalize=["jetscape"],
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
)
# -

# #### R = 0.4

analytical_pp["R04"]["dynamical_kt"].axes[0].bin_centers, pp_R04_unfolded_with_systematics["dynamical_kt"].data.axes[0].bin_edges

import numpy as np

unfolding_base.select_hist_range(analytical_pp["R04"]["dynamical_kt"], helpers.KtRange(0.5, 8))
#x_range = helpers.KtRange(0.5, 8)
#temp_bin_centers = analytical_pp["R04"]["dynamical_kt"].axes[0].bin_centers
#bin_center_mask = (temp_bin_centers >= x_range.min) & (temp_bin_centers <= x_range.max)
#bin_center_mask
#np.where(bin_center_mask)[0][0]
#np.where(bin_center_mask[::-1])

# +
jet_R = 0.4
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_paper.plot_pp_grooming_comparison_with_models(
    hists=pp_R04_unfolded_with_systematics,
    grooming_methods=grooming_methods,
    reference_grooming_method="soft_drop_z_cut_02",
    models={
        ##"jetscape": jetscape_pp["R04"],
        "analytical": analytical_pp["R04"],
        "sherpa_ahadic": sherpa_ahadic["R04"],
        "sherpa_lund": sherpa_lund["R04"],
        "pythia": pp_R04_true_reference,
    },
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 8),
    kt_ranges_for_models={
        # TODO: Update jetscape range when binning is fixed.
        #       For now, there is basically no overlap :-(
        ##"jetscape": helpers.KtRange(2, 8),
        "analytical": helpers.KtRange(0.5, 8),
        "sherpa_ahadic": helpers.KtRange(0.25, 8),
        "sherpa_lund": helpers.KtRange(0.25, 8),
        "pythia": helpers.KtRange(0.25, 8),
    },
    models_to_normalize=["jetscape", "sherpa_lund", "sherpa_ahadic"],
    figure_kt_range=helpers.KtRange(0, 8.25),
    jet_R_str=jet_R_str,
)
# -

#sns.color_palette("Accent", n_colors=12)
sns.color_palette(plot_unfolding._model_palette)

import matplotlib.colors

t = matplotlib.colors.LinearSegmentedColormap.from_list("test", N=6, colors=[(1, 1, 1), sns.color_palette("colorblind")[0]])
t

t(2)

# # DyG kt

# +
# General setup
plot = False
substructure_variable = "kt"
true_jet_pt_range = helpers.JetPtRange(60, 80)
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
grooming_methods = ["dynamical_kt"]

_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)
# -

# ## pp

# + tags=["remove_cell"]
# pp
collision_system = "pp"
event_activity = "pp"
_smeared_var_range = helpers.KtRange(0.25, 6)
_smeared_untagged_var = helpers.KtRange(0.25, 0.25)
_smeared_jet_pt_range = helpers.JetPtRange(20, 85)
# _tag_after_suffix = "2_4_split"
_n_iter_compare = 3
_truncation_shift = 3
_displaced_extremum = 10

# + tags=["remove_cell"]
# Initially load data
pp_R02_unfolding_closure_outputs, pp_R02_unfolding_closure_pure_matches_outputs, pp_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    output_dir=output_dir,
    # tag_after_suffix=_tag_after_suffix,
)
#for grooming_method in grooming_methods:
#    print(f"running {grooming_method}")

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    pp_R02_unfolding_systematics_outputs[grooming_method]["non_closure"] = pp_R02_unfolding_closure_outputs[grooming_method]["reweight_response"]

# Focus down onto just the unfolded distributions
pp_R02_unfolded_with_systematics, pp_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=pp_R02_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)

if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=pp_R02_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0.5, 6)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.5, 1.5],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir = pp_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png = True,
        )
    
## Fully process systematics
#for grooming_method in grooming_methods:
#    pp_R04_unfolded[grooming_method] = plot_unfolding.unfolded_substructure_results(
#        unfolding_outputs=pp_R04_unfolding_systematics_outputs[grooming_method],
#        true_jet_pt_range=true_jet_pt_range,
#    )
#
#    pp_R04_unfolded_with_systematics[grooming_method] = plot_unfolding.calculate_systematics(
#        unfolded=pp_R04_unfolded[grooming_method],
#        unfolding_outputs=pp_R04_unfolding_systematics_outputs[grooming_method],
#        true_jet_pt_range=true_jet_pt_range,
#    )
#
#    pp_R04_true_reference[grooming_method] = pp_R04_unfolding_systematics_outputs[grooming_method]["default"].true_substructure(
#        pp_R04_unfolding_systematics_outputs[grooming_method]["default"].true_hist_name, true_jet_pt_range=true_jet_pt_range
#    )
#
#    # Plot the individual relative systematics
#    if plot:
#        plot_unfolding.plot_relative_individual_systematics(
#            unfolded=pp_R04_unfolded_with_systematics[grooming_method],
#            plot_config=pb.PlotConfig(
#                name="unfolded_systematic_relative",
#                panels=[
#                    pb.Panel(
#                        axes=[
#                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(1.5, 15)),
#                            pb.AxisConfig(
#                                "y",
#                                label="Relative error",
#                                range=[0, 1],
#                            ),
#                        ],
#                        legend=pb.LegendConfig(location="upper right"),
#                        #text=pb.TextConfig(text, 0.97, 0.97),
#                    ),
#                ],
#            ),
#            output_dir = pp_R04_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
#            plot_png = True,
#        )
#
## Plot full systematics for multiple grooming methods on one plot.
#if plot:
#    plot_unfolding.plot_PbPb_systematics(
#        hists=pp_R04_unfolded_with_systematics,
#        reference=pp_R04_true_reference,
#        grooming_methods=grooming_methods,
#        event_activity=event_activity,
#        kt_range=[2, 15],
#        # Arbitrarily take the first grooming method for the output dir
#        output_dir=pp_R04_unfolding_systematics_outputs[grooming_methods[0]]["default"].output_dir,
#    )
# -

# ## Semi-central

# + tags=["remove_cell"]
collision_system = "PbPb"
event_activity = "semi_central"
_tag_after_suffix = "pass3"
_smeared_var_range = helpers.KtRange(1, 8)
_smeared_untagged_var = helpers.KtRange(0, 1)
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_n_iter_compare = 3
_truncation_shift = 5
_displaced_extremum = 10

# + tags=["remove_cell"]
# Initially load data
semi_central_R02_unfolding_closure_outputs, semi_central_R02_unfolding_closure_pure_matches_outputs, semi_central_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
)

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    semi_central_R02_unfolding_systematics_outputs[grooming_method]["non_closure"] = \
        semi_central_R02_unfolding_closure_outputs[grooming_method]["reweight_response"]
        #semi_central_R02_unfolding_closure_outputs[grooming_method]["reweight_pseudo_data"]

# Focus down onto just the unfolded distributions
semi_central_R02_unfolded_with_systematics, semi_central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=semi_central_R02_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)

if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=semi_central_R02_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0.5, 6)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.25, 1.75],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir = semi_central_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png = True,
        )
# -

# ## Central

# + tags=["remove_cell"]
collision_system = "PbPb"
event_activity = "central"
_tag_after_suffix = "pass3"
_smeared_var_range = helpers.KtRange(1.5, 8)
_smeared_untagged_var = helpers.KtRange(0, 1.5)
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_n_iter_compare = 10
_truncation_shift = 5
_displaced_extremum = 10

# + tags=["remove_cell"]
# Initially load data
central_R02_unfolding_closure_outputs, central_R02_unfolding_closure_pure_matches_outputs, central_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
)

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    central_R02_unfolding_systematics_outputs[grooming_method]["non_closure"] = central_R02_unfolding_closure_outputs[grooming_method]["reweight_response"]

# Focus down onto just the unfolded distributions
central_R02_unfolded_with_systematics, central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=central_R02_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)

if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=central_R02_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0.5, 6)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.25, 1.75],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir = central_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png = True,
        )
# -

for grooming_method in grooming_methods:
    plot_unfolding.plot_pp_PbPb_comparison(
        hists={
            "pp": pp_R02_unfolded_with_systematics[grooming_method],
            "semi_central": semi_central_R02_unfolded_with_systematics[grooming_method],
            "central": central_R02_unfolded_with_systematics[grooming_method],
        },
        grooming_method=grooming_method,
        output_dir=_output_dir,
        kt_range=(0.25, 6.25),
        jet_R_str=jet_R_str,
    )



# # Soft Drop z > 0.2

# +
# General setup
plot = False
substructure_variable = "kt"
true_jet_pt_range = helpers.JetPtRange(60, 80)
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
grooming_methods = ["soft_drop_z_cut_02"]

_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)
# -

# ## pp

# + tags=["remove_cell"]
# pp
collision_system = "pp"
event_activity = "pp"
_smeared_var_range = helpers.KtRange(0.25, 6)
_smeared_untagged_var = helpers.KtRange(0, 0.25)
_smeared_jet_pt_range = helpers.JetPtRange(20, 85)
_tag_after_suffix = "z_cut_02_kt_0.25"
_n_iter_compare = 3
_truncation_shift = 3
_displaced_extremum = 10

# + jupyter={"outputs_hidden": true} tags=["remove_cell"]
# Initially load data
pp_R02_unfolding_closure_outputs, pp_R02_unfolding_closure_pure_matches_outputs, pp_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
)
#for grooming_method in grooming_methods:
#    print(f"running {grooming_method}")

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    pp_R02_unfolding_systematics_outputs[grooming_method]["non_closure"] = pp_R02_unfolding_closure_outputs[grooming_method]["reweight_response"]

# Focus down onto just the unfolded distributions
pp_R02_unfolded_with_systematics, pp_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=pp_R02_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)

if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=pp_R02_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0.5, 6)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.5, 1.5],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir = pp_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png = True,
        )
    
## Fully process systematics
#for grooming_method in grooming_methods:
#    pp_R04_unfolded[grooming_method] = plot_unfolding.unfolded_substructure_results(
#        unfolding_outputs=pp_R04_unfolding_systematics_outputs[grooming_method],
#        true_jet_pt_range=true_jet_pt_range,
#    )
#
#    pp_R04_unfolded_with_systematics[grooming_method] = plot_unfolding.calculate_systematics(
#        unfolded=pp_R04_unfolded[grooming_method],
#        unfolding_outputs=pp_R04_unfolding_systematics_outputs[grooming_method],
#        true_jet_pt_range=true_jet_pt_range,
#    )
#
#    pp_R04_true_reference[grooming_method] = pp_R04_unfolding_systematics_outputs[grooming_method]["default"].true_substructure(
#        pp_R04_unfolding_systematics_outputs[grooming_method]["default"].true_hist_name, true_jet_pt_range=true_jet_pt_range
#    )
#
#    # Plot the individual relative systematics
#    if plot:
#        plot_unfolding.plot_relative_individual_systematics(
#            unfolded=pp_R04_unfolded_with_systematics[grooming_method],
#            plot_config=pb.PlotConfig(
#                name="unfolded_systematic_relative",
#                panels=[
#                    pb.Panel(
#                        axes=[
#                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(1.5, 15)),
#                            pb.AxisConfig(
#                                "y",
#                                label="Relative error",
#                                range=[0, 1],
#                            ),
#                        ],
#                        legend=pb.LegendConfig(location="upper right"),
#                        #text=pb.TextConfig(text, 0.97, 0.97),
#                    ),
#                ],
#            ),
#            output_dir = pp_R04_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
#            plot_png = True,
#        )
#
## Plot full systematics for multiple grooming methods on one plot.
#if plot:
#    plot_unfolding.plot_PbPb_systematics(
#        hists=pp_R04_unfolded_with_systematics,
#        reference=pp_R04_true_reference,
#        grooming_methods=grooming_methods,
#        event_activity=event_activity,
#        kt_range=[2, 15],
#        # Arbitrarily take the first grooming method for the output dir
#        output_dir=pp_R04_unfolding_systematics_outputs[grooming_methods[0]]["default"].output_dir,
#    )
# -

# ## Semi-central

# + tags=["remove_cell"]
collision_system = "PbPb"
event_activity = "semi_central"
_tag_after_suffix = "pass3_z_cut_02_kt_0.25"
_smeared_var_range = helpers.KtRange(0.25, 8)
_smeared_untagged_var = helpers.KtRange(0, 0.25)
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_n_iter_compare = 3
_truncation_shift = 5
_displaced_extremum = 10

# + tags=["remove_cell"]
# Initially load data
semi_central_R02_unfolding_closure_outputs, semi_central_R02_unfolding_closure_pure_matches_outputs, semi_central_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
)

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    semi_central_R02_unfolding_systematics_outputs[grooming_method]["non_closure"] = \
        semi_central_R02_unfolding_closure_outputs[grooming_method]["reweight_response"]
        #semi_central_R02_unfolding_closure_outputs[grooming_method]["reweight_pseudo_data"]

# Focus down onto just the unfolded distributions
semi_central_R02_unfolded_with_systematics, semi_central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=semi_central_R02_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)

if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=semi_central_R02_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0.5, 6)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.25, 1.75],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir = semi_central_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png = True,
        )
# -

# ## Central

# + tags=["remove_cell"]
collision_system = "PbPb"
event_activity = "central"
_tag_after_suffix = "pass3"
_smeared_var_range = helpers.KtRange(1.5, 8)
_smeared_untagged_var = helpers.KtRange(0, 1.5)
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_n_iter_compare = 10
_truncation_shift = 5
_displaced_extremum = 10

# + tags=["remove_cell"]
# Initially load data
central_R02_unfolding_closure_outputs, central_R02_unfolding_closure_pure_matches_outputs, central_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
)

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    central_R02_unfolding_systematics_outputs[grooming_method]["non_closure"] = central_R02_unfolding_closure_outputs[grooming_method]["reweight_response"]

# Focus down onto just the unfolded distributions
central_R02_unfolded_with_systematics, central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=central_R02_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)

if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=central_R02_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0.5, 6)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.25, 1.75],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir = central_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png = True,
        )
# -

for grooming_method in grooming_methods:
    plot_unfolding.plot_pp_PbPb_comparison(
        hists={
            "pp": pp_R02_unfolded_with_systematics[grooming_method],
            "semi_central": semi_central_R02_unfolded_with_systematics[grooming_method],
            #"central": central_R02_unfolded_with_systematics[grooming_method],
        },
        grooming_method=grooming_method,
        output_dir=_output_dir,
        kt_range=(0.25, 6.25),
        jet_R_str=jet_R_str,
    )



# # Compare Grooming Methods for pp

# +
# General setup
plot = True
substructure_variable = "kt"
true_jet_pt_range = helpers.JetPtRange(60, 80)
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
grooming_methods = ["dynamical_core", "dynamical_kt"]

_output_dir = output_dir / "comparison" / "unfolding" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)
# -

# ## pp

# + tags=["remove_cell"]
# pp
collision_system = "pp"
event_activity = "pp"
_smeared_var_range = helpers.KtRange(0.25, 6)
_smeared_untagged_var = helpers.KtRange(0.25, 0.25)
_smeared_jet_pt_range = helpers.JetPtRange(20, 85)
#_tag_after_suffix = "2_4_split"
_n_iter_compare = {
    "dynamical_core": 3,
    "dynamical_kt": 3,
}
_truncation_shift = 3
_displaced_extremum = 10

# + jupyter={"outputs_hidden": true} tags=["remove_cell"]
# Initially load data
pp_R02_unfolding_closure_outputs, pp_R02_unfolding_closure_pure_matches_outputs, pp_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    output_dir=output_dir,
    #tag_after_suffix=_tag_after_suffix,
)
#for grooming_method in grooming_methods:
#    print(f"running {grooming_method}")

# Add in the closure test to provide the non-closure uncertainty
for grooming_method in grooming_methods:
    pp_R02_unfolding_systematics_outputs[grooming_method]["non_closure"] = pp_R02_unfolding_closure_outputs[grooming_method]["reweight_response"]

# Focus down onto just the unfolded distributions
pp_R02_unfolded_with_systematics, pp_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=pp_R02_unfolding_systematics_outputs,
    true_jet_pt_range=true_jet_pt_range,
)
    
if plot:
    for grooming_method in grooming_methods:
        # Plot the individual relative systematics
        plot_unfolding.plot_relative_individual_systematics(
            unfolded=pp_R02_unfolded_with_systematics[grooming_method],
            plot_config=pb.PlotConfig(
                name="unfolded_systematic_relative",
                panels=[
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(0.5, 6)),
                            pb.AxisConfig(
                                "y",
                                label="Relative error",
                                range=[0.5, 1.5],
                            ),
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2),
                        #text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                ],
            ),
            output_dir = pp_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
            plot_png = True,
        )
# -

list(pp_R02_unfolded_with_systematics.keys())

plot_unfolding.plot_grooming_comparisons_for_single_system(
    hists=pp_R02_unfolded_with_systematics,
    grooming_methods=grooming_methods,
    reference_grooming_method="dynamical_core",
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=(0.25, 6.25),
    jet_R_str=jet_R_str,
)





# # QM 2022 Plots

alice_status = "work_in_progress"

# ## pp grooming methods comparison

# ### R = 0.2

from importlib import reload
reload(plot_unfolding)

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / input_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_unfolding.plot_grooming_comparisons_for_single_system(
    hists=pp_R02_unfolded_with_systematics,
    grooming_methods=grooming_methods,
    reference_grooming_method="dynamical_kt_z_cut_02",
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 6),
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
)
# -

# ## pp comparison to models
#
# ### R = 0.2

from importlib import reload
reload(plot_unfolding)

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / input_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_unfolding.plot_grooming_model_comparisons_for_single_system(
    hists=pp_R02_unfolded_with_systematics,
    models={
        # TODO: Need to update the jetscape binning. Whoops...
        #"jetscape": jetscape_pp["R02"],
        "pythia": pp_R02_true_reference,
    },
    grooming_methods=grooming_methods,
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 6),
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
)
# -

# ## PbPb grooming methods comparison
#
# ### R = 0.2

from importlib import reload
reload(plot_unfolding)

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / input_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for _collision_system, _hists in [
    ("semi_central", semi_central_R02_unfolded_with_systematics),
    ("central", central_R02_unfolded_with_systematics),
]:
    for _temp_grooming_methods, _reference_method, _label in [
        #(["soft_drop_z_cut_02", "dynamical_core", "dynamical_kt", "dynamical_time"], "soft_drop_z_cut_02", "1"),
        (["soft_drop_z_cut_04", "dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "soft_drop_z_cut_02", "2"),
        #(["soft_drop_z_cut_02", "soft_drop_z_cut_04", "dynamical_kt", "dynamical_kt_z_cut_02"], "soft_drop_z_cut_02", "3"),
        (["soft_drop_z_cut_02", "dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "soft_drop_z_cut_02", "4"),
    ]:
        plot_unfolding.plot_Rg_grooming_comparisons_for_single_system(
            hists=_hists,
            grooming_methods=_temp_grooming_methods,
            reference_grooming_method=_reference_method,
            collision_system=_collision_system,
            collision_system_key="PbPb",
            output_dir=_output_dir,
            kt_range={
                "dynamical_core": helpers.KtRange(0.0, 0.2),
                "dynamical_kt": helpers.KtRange(0.0, 0.),
                "dynamical_time": helpers.KtRange(0.0, 0.2),
                "soft_drop_z_cut_02": helpers.KtRange(0.0, 0.2),
                "soft_drop_z_cut_04": helpers.KtRange(0.0, 0.2),
                "dynamical_core_z_cut_02": helpers.KtRange(0.0, 0.2),
                "dynamical_kt_z_cut_02": helpers.KtRange(0.0, 0.2),
                "dynamical_time_z_cut_02": helpers.KtRange(0.0, 0.2),
            },
            figure_kt_range=helpers.KtRange(0, 0.2),
            jet_R_str=jet_R_str,
            alice_status=alice_status,
            label=_label,
        )
# -

# ## PbPb-pp comparison by each grooming methods
#
# ### R = 0.2

from importlib import reload
reload(plot_unfolding)

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / input_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for grooming_method in grooming_methods:
    plot_unfolding.plot_pp_PbPb_comparison(
        hists={
            "pp": pp_R02_unfolded_with_systematics[grooming_method],
            "semi_central": semi_central_R02_unfolded_with_systematics[grooming_method],
            "central": central_R02_unfolded_with_systematics[grooming_method],
        },
        grooming_method=grooming_method,
        output_dir=_output_dir,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": helpers.KtRange(0.25, 6) if "z_cut" in grooming_method else helpers.KtRange(2, 6),
            "central": helpers.KtRange(0.25, 6) if "z_cut" in grooming_method else helpers.KtRange(3, 6),
            # Peter's binning
            #"semi_central": helpers.KtRange(2.25, 6) if "z_cut" not in grooming_method else helpers.KtRange(0.25, 6),
        },
        kt_display_range=(0.0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )
# -





# ## PbPb-pp comparison to models
#
# ### R = 0.2

from importlib import reload
reload(plot_unfolding)

# +
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / input_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for grooming_method in grooming_methods:
    plot_unfolding.plot_pp_PbPb_comparison(
        hists={
            "pp": pp_R02_unfolded_with_systematics[grooming_method],
            "semi_central": semi_central_R02_unfolded_with_systematics[grooming_method],
            #"central": central_R02_unfolded_with_systematics[grooming_method],
        },
        models={
            "hybrid_without_moliere": semi_central_hybrid_model_ratio["hybrid_without_moliere"][jet_R_str],
            "hybrid_moliere": semi_central_hybrid_model_ratio["hybrid_moliere"][jet_R_str],
        },
        grooming_method=grooming_method,
        output_dir=_output_dir,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": helpers.KtRange(0.25, 6) if "z_cut" in grooming_method else helpers.KtRange(2, 6),
        },
        kt_display_range=(0.0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )
# -



# # zg

# ## R = 0.2

# +
# General setup
plot = False
substructure_variable = "z"
true_jet_pt_range = helpers.JetPtRange(60, 80)
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
grooming_methods = [
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
    #"soft_drop_z_cut_02",
    #"dynamical_core_z_cut_02",
    #"dynamical_kt_z_cut_02",
    #"dynamical_time_z_cut_02",
    #"soft_drop_z_cut_04",
]
_OG_grooming_methods = [
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
    "soft_drop_z_cut_02",
]
_new_grooming_methods = [
    "dynamical_core_z_cut_02",
    "dynamical_kt_z_cut_02",
    "dynamical_time_z_cut_02",
    "soft_drop_z_cut_04",
]
input_dir_tag = "2023-paper"

###################
# Setup I/O options
###################
_use_qm22_inputs = False
_grooming_methods_using_qm_result_conventions = _OG_grooming_methods if _use_qm22_inputs else []
_grooming_methods_using_new_conventions = _new_grooming_methods if _use_qm22_inputs else grooming_methods

_output_dir = output_dir / "comparison" / "unfolding" / "2023-paper-plots" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)
# -

# ### pp

# +
# pp
collision_system = "pp"
event_activity = "pp"
_smeared_var_range: helpers.ZgRange | dict[str, helpers.ZgRange] = helpers.ZgRange(0., 0.5)
# NOTE: Using Laura's jet pt range, which is different from my usual one in pp
_smeared_jet_pt_range = helpers.JetPtRange(20, 85)
#_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_truncation_shift = 3
_displaced_extremum = 0.6
#_tag_after_suffix = "2_4_split"

###################
# Setup I/O options
###################
# Input directory location
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
_input_dir_tag = {
    _method: "2023-paper"
    for _method in _grooming_methods_using_qm_result_conventions
}
_input_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
_output_dir_tag: dict[str, str | None] = {
    _method: "2023-paper-plots"
    for _method in _grooming_methods_using_qm_result_conventions
}
_output_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
_tag_after_suffix = {
    grooming_method: "" for grooming_method in grooming_methods
}
#_tag_after_suffix["soft_drop_z_cut_04"] = "merge_3_6"

####################################
# Grooming method dependent settings
####################################
_smeared_untagged_var = {
    "dynamical_core": helpers.ZgRange(-0.05, 0),
    "dynamical_kt": helpers.ZgRange(-0.05, 0),
    "dynamical_time": helpers.ZgRange(-0.05, 0),
    "soft_drop_z_cut_02": helpers.ZgRange(-0.05, 0),
    "soft_drop_z_cut_04": helpers.ZgRange(-0.05, 0),
    "dynamical_core_z_cut_02": helpers.ZgRange(-0.05, 0),
    "dynamical_kt_z_cut_02": helpers.ZgRange(-0.05, 0),
    "dynamical_time_z_cut_02": helpers.ZgRange(-0.05, 0),
}
_n_iter_compare = {
    "dynamical_core": 3,
    "dynamical_kt": 3,
    "dynamical_time": 3,
    "soft_drop_z_cut_02": 3,
    # Not yet optimized...
    "soft_drop_z_cut_04": 3,
    "dynamical_core_z_cut_02": 3,
    "dynamical_kt_z_cut_02": 3,
    "dynamical_time_z_cut_02": 3,
}
_max_n_iter: dict[str, int | None] = {
    # Need +1 for convenience with range iteration
    "soft_drop_z_cut_04": 30,
}
_max_n_iter.update({
    grooming_method: 20 for grooming_method in grooming_methods if grooming_method != "soft_drop_z_cut_04"
})


# Model dependence.
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
_model_dependence_configuration: dict[str, unfolding_analysis.ModelDependenceConfiguration | None] = {
    _method: unfolding_analysis.ModelDependenceConfiguration(
        nominal="pythia_fastsim",
        variations=["herwig_fastsim"],
    ) for _method in _grooming_methods_using_new_conventions
}
# Non-closure
non_closure_configuration = {
    _method: None
    for _method in _grooming_methods_using_new_conventions
}

# Either take model dependence or reweighted prior
# Model dependence is always preferred, but it may not have been analyzed yet for the a particular configuration
# (or in PbPb, it likely isn't possible since we don't have a reliable MC)
skip_reweighted_prior_in_systematics = True

# +
pp_R02_unfolding_closure_outputs, pp_R02_unfolding_closure_pure_matches_outputs, pp_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    max_n_iter=_max_n_iter,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    input_dir_tag=_input_dir_tag,
    output_dir_tag=_output_dir_tag,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
    skip_reweighted_prior_in_systematics=skip_reweighted_prior_in_systematics,
    model_dependence_configuration=_model_dependence_configuration,
)

# Focus down onto just the unfolded distributions
pp_R02_unfolded_with_systematics, pp_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=pp_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=pp_R02_unfolding_closure_outputs,
    true_jet_pt_range=true_jet_pt_range,
    model_dependence_configuration=_model_dependence_configuration,
    non_closure_configuration=non_closure_configuration,
    background_subtraction_configuration=None,
)

# +
#plot = True
#if plot:
#    for grooming_method in grooming_methods:
#
#        # Plot the individual relative systematics
#        plot_unfolding.plot_relative_individual_systematics(
#            unfolded=pp_R02_unfolded_with_systematics[grooming_method],
#            plot_config=pb.PlotConfig(
#                name="unfolded_systematic_relative",
#                panels=[
#                    pb.Panel(
#                        axes=[
#                            pb.AxisConfig("x", label=r"$R_{\text{g}}$", range=(-0.1, 0.3)),
#                            pb.AxisConfig(
#                                "y",
#                                label="Relative error",
#                                range=[0.5, 1.5],
#                            ),
#                        ],
#                        legend=pb.LegendConfig(location="upper right", ncol=2),
#                        #text=pb.TextConfig(text, 0.97, 0.97),
#                    ),
#                ],
#            ),
#            output_dir = pp_R02_unfolding_systematics_outputs[grooming_method]["default"].output_dir,
#            plot_png = True,
#        )
#        
#        plot_unfolding.plot_delta_R_unfolding(
#            unfolding_output=pp_R02_unfolding_systematics_outputs[grooming_method]["default"],
#            plot_png=True,
#            #reweighted_prior_output=pp_R02_unfolding_systematics_outputs[grooming_method]["model_dependence"],
#            reweighted_prior_output=pp_R02_unfolding_systematics_outputs[grooming_method]["reweight_prior"],
#            unfolding_Rg_display_range=(-0.05, 0.2),
#        )
#        ##for _outputs in [pp_R02_unfolding_closure_outputs, pp_R02_unfolding_closure_pure_matches_outputs, pp_R02_unfolding_systematics_outputs]:
#        #for _outputs in [pp_R02_unfolding_closure_outputs, pp_R02_unfolding_systematics_outputs]:
#        for _outputs in [pp_R02_unfolding_closure_outputs]:
#            for name, _unfolding_output in _outputs[grooming_method].items():
#                # Skip, since we already plotted above.
#                if name == "default":
#                    continue
#                plot_unfolding.plot_delta_R_unfolding(
#                    unfolding_output=_unfolding_output,
#                    plot_png=True,
#                    #unfolding_kt_display_range=(0.5, 6) if "z_cut" not in grooming_method else (0.25, 6),
#                    unfolding_Rg_display_range=(-0.05, 0.2),
#                )


## Plot full systematics for multiple grooming methods on one plot.
#if plot:
#    plot_unfolding.plot_PbPb_systematics(
#        hists=pp_R02_unfolded_with_systematics,
#        reference=pp_R02_true_reference,
#        grooming_methods=grooming_methods,
#        event_activity=event_activity,
#        kt_range=[2, 15],
#        # Arbitrarily take the first grooming method for the output dir
#        output_dir=pp_R02_unfolding_systematics_outputs[grooming_methods[0]]["default"].output_dir,
#    )

plot_unfolding.steer_plotting_of_substructure_var_unfolding_outputs(
    substructure_variable="z",
    grooming_methods=grooming_methods,
    unfolded_with_systematics=pp_R02_unfolded_with_systematics,
    unfolding_systematics_outputs=pp_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=pp_R02_unfolding_closure_outputs,
    plot=True,
    plot_png=False,
    plot_systematic_breakdown=True,
    plot_systematics=False,
    plot_closures=False,
    # NOTE: For the prior variation, passing the HERwIG model dependence includes both:
    #       - HERWIG vs PYTHIA
    #       - fastsim vs full sim as well as whatever HERWIG
    #       Consequently, the fastsim output may not be the most accurate overall magnitude, but we can
    #       still use it to look at the shape for selecting the iteration. Alternatively, we can switch
    #       back to the reweighted_prior, but that output is less satisfying for pp.
    # NOTE: We can't remove the fastsim vs full sim dependence at the moment because we would need
    #       the full UnfoldingOutput object, which we don't have available since the model dependence here
    #       is constructed by transferring the differences from the fastsim outputs to the default.
    #       We could do this, but it's more tricky (eg. can refolded be treated the same way?
    #       Probably, but would need to be checked), so we just stick with the HERWIG model dependence.
    prior_variation_output_name="model_dependence_herwig_fastsim",
    #prior_variation_output_name="reweight_prior",
    unfolding_display_range={
        grooming_method: (-0.05, 0.5)
        for grooming_method in grooming_methods
    },
    relative_individual_systematic_ratio_range={
        grooming_method: (0.5, 1.5) for grooming_method in grooming_methods
    }
)
# -

# ### Semi-central

# +
collision_system = "PbPb"
event_activity = "semi_central"
#_double_counting_cut = "min_true_10_pt_hat_3"
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_truncation_shift = 5
_displaced_extremum = 0.6

###################
# Setup I/O options
###################
# Input directory location
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
_input_dir_tag = {
    _method: "2023-paper"
    for _method in _grooming_methods_using_qm_result_conventions
}
_input_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
_output_dir_tag: dict[str, str | None] = {
    _method: "2023-paper-plots"
    for _method in _grooming_methods_using_qm_result_conventions
}
_output_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
_tag_after_suffix_base = "pass3"
_tag_after_suffix = {
    grooming_method: _tag_after_suffix_base for grooming_method in grooming_methods
}
_tag_after_suffix["soft_drop_z_cut_04"] = f"{_tag_after_suffix_base}_merge_3_6"
####################################
# Grooming method dependent settings
####################################
_smaered_var_range = helpers.ZgRange(0, 0.5)
_smeared_untagged_var = {
    "dynamical_core": helpers.ZgRange(-0.05, 0),
    "dynamical_kt": helpers.ZgRange(-0.05, 0),
    "dynamical_time": helpers.ZgRange(-0.05, 0),
    "soft_drop_z_cut_02": helpers.ZgRange(-0.05, 0),
    "soft_drop_z_cut_04": helpers.ZgRange(-0.05, 0),
    "dynamical_core_z_cut_02": helpers.ZgRange(-0.05, 0),
    "dynamical_kt_z_cut_02": helpers.ZgRange(-0.05, 0),
    "dynamical_time_z_cut_02": helpers.ZgRange(-0.05, 0),
}
_n_iter_compare = {
    "dynamical_core": 6,
    "dynamical_kt": 6,
    "dynamical_time": 6,
    "soft_drop_z_cut_02": 9,
    "dynamical_core_z_cut_02": 9,
    "dynamical_kt_z_cut_02": 9,
    "dynamical_time_z_cut_02": 9,
    "soft_drop_z_cut_04": 9,
}
_max_n_iter: dict[str, int | None] = {
    # Need +1 for convenience with range iteration
    "soft_drop_z_cut_04": 30,
}
_max_n_iter.update({
    grooming_method: 20 for grooming_method in grooming_methods if grooming_method != "soft_drop_z_cut_04"
})

# Double counting cut
# It's all the same here, but the QM22 results don't include the label
_double_counting_cut = {
    _method: "min_true_10_pt_hat_3"
    for _method in _grooming_methods_using_new_conventions
}
# Model dependence.
_model_dependence_configuration = None
# Background subtraction configurations
_background_subtraction_configuration = {
    _method: unfolding_analysis.BackgroundSubtractionConfiguration(
        contributors=["Rmax005", "Rmax050"]
    )
    for _method in _grooming_methods_using_new_conventions
}
# Add in the closure test to provide the non-closure uncertainty
_non_closure_configuration = {
    grooming_method: unfolding_analysis.NonClosureConfiguration(
        #contributors=["reweight_response", "reweight_pseudo_data", "thermal_model"],
        # Temporarily disabled thermal model because it wasn't immediately available
        contributors=["reweight_response", "reweight_pseudo_data"],
        approach_to_combining="max",
    )
    for grooming_method in _grooming_methods_using_new_conventions
}

# +
# Initially load data
semi_central_R02_unfolding_closure_outputs, semi_central_R02_unfolding_closure_pure_matches_outputs, semi_central_R02_unfolding_systematics_outputs = plot_unfolding.load_unfolded_outputs(
    grooming_methods=grooming_methods,
    substructure_variable=substructure_variable,
    smeared_var_range=_smeared_var_range,
    smeared_untagged_var=_smeared_untagged_var,
    smeared_jet_pt_range=_smeared_jet_pt_range,
    collision_system=collision_system,
    event_activity=event_activity,
    jet_R_str=jet_R_str,
    n_iter_compare=_n_iter_compare,
    max_n_iter=_max_n_iter,
    truncation_shift=_truncation_shift,
    displaced_extremum=_displaced_extremum,
    input_dir_tag=_input_dir_tag,
    output_dir_tag=_output_dir_tag,
    output_dir=output_dir,
    tag_after_suffix=_tag_after_suffix,
    double_counting_cut=_double_counting_cut,
    model_dependence_configuration=_model_dependence_configuration,
)

# Focus down onto just the unfolded distributions
semi_central_R02_unfolded_with_systematics, semi_central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=semi_central_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=semi_central_R02_unfolding_closure_outputs,
    true_jet_pt_range=true_jet_pt_range,
    model_dependence_configuration=_model_dependence_configuration,
    non_closure_configuration=_non_closure_configuration,
    background_subtraction_configuration=_background_subtraction_configuration,
)
# -

print(list(semi_central_R02_unfolding_systematics_outputs["dynamical_kt"].keys()))
print(list(semi_central_R02_unfolding_closure_outputs["dynamical_kt"].keys()))

plot_unfolding.steer_plotting_of_substructure_var_unfolding_outputs(
    substructure_variable="z",
    grooming_methods=grooming_methods,
    unfolded_with_systematics=semi_central_R02_unfolded_with_systematics,
    unfolding_systematics_outputs=semi_central_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=semi_central_R02_unfolding_closure_outputs,
    plot=True,
    plot_png=False,
    plot_systematic_breakdown=True,
    plot_systematics=False,
    plot_closures=True,
    prior_variation_output_name="reweight_prior",
    unfolding_display_range={
        grooming_method: (-0.05, 0.5)
        for grooming_method in grooming_methods
    },
    relative_individual_systematic_ratio_range={
        grooming_method: (0.25, 1.75)
        for grooming_method in grooming_methods
    }
)

list(semi_central_R02_unfolded_with_systematics.keys())
