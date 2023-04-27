# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: substructure_c_24_06
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Paper plots for hardest kt pp, semi-central, central
#

# %%
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

# %% [markdown]
# # Load data

# %% [markdown]
# ## R = 0.2

# %%
# General setup
plot = False
substructure_variable = "kt"
true_jet_pt_range = helpers.JetPtRange(60, 80)
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
grooming_methods = [
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
    "soft_drop_z_cut_02",
    "dynamical_core_z_cut_02",
    "dynamical_kt_z_cut_02",
    "dynamical_time_z_cut_02",
    "soft_drop_z_cut_04",
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
input_dir_tag = "2023-02-HP"
###################
# Setup I/O options
###################
_use_qm22_inputs = True
_grooming_methods_using_qm_result_conventions = _OG_grooming_methods if _use_qm22_inputs else []
_grooming_methods_using_new_conventions = _new_grooming_methods if _use_qm22_inputs else grooming_methods

_output_dir = output_dir / "comparison" / "unfolding" / "2023-paper-plots" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### pp

# %% tags=["remove_cell"]
# pp
collision_system = "pp"
event_activity = "pp"
_smeared_var_range: helpers.KtRange | dict[str, helpers.KtRange] = helpers.KtRange(0.25, 6)
_smeared_jet_pt_range = helpers.JetPtRange(20, 85)
_truncation_shift = 3
_displaced_extremum = 10

###################
# Setup I/O options
###################
# Input directory location
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
_input_dir_tag = {
    _method: "parsl/2022-03-QM"
    for _method in _grooming_methods_using_qm_result_conventions
}
_input_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
_output_dir_tag: dict[str, str | None] = {
    #_method: input_dir_tag + "-from-QM22-results"
    _method: "2023-paper-plots-from-QM22-results"
    for _method in _grooming_methods_using_qm_result_conventions
}
_output_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
#_tag_after_suffix = "2_4_split"
#_tag_after_suffix = "peter_binning"
#_tag_after_suffix = ""
_tag_after_suffix = {
    grooming_method: "" for grooming_method in grooming_methods
}
_tag_after_suffix["soft_drop_z_cut_04"] = "merge_3_6"
####################################
# Grooming method dependent settings
####################################
_smeared_untagged_var = {
    "dynamical_core": helpers.KtRange(0.25, 0.25),
    "dynamical_kt": helpers.KtRange(0.25, 0.25),
    "dynamical_time": helpers.KtRange(0.25, 0.25),
    "soft_drop_z_cut_02": helpers.KtRange(0, 0.25),
    "soft_drop_z_cut_04": helpers.KtRange(0, 0.25),
    "dynamical_core_z_cut_02": helpers.KtRange(0, 0.25),
    "dynamical_kt_z_cut_02": helpers.KtRange(0, 0.25),
    "dynamical_time_z_cut_02": helpers.KtRange(0, 0.25),
}
_n_iter_compare = {
    "dynamical_core": 6,
    "dynamical_kt": 6,
    "dynamical_time": 6,
    "soft_drop_z_cut_02": 6,
    "dynamical_core_z_cut_02": 6,
    "dynamical_kt_z_cut_02": 6,
    "dynamical_time_z_cut_02": 6,
    "soft_drop_z_cut_04": 11,
}
if _use_qm22_inputs:
    _n_iter_compare.update({
        "dynamical_core": 3,
        "dynamical_kt": 3,
        "dynamical_time": 3,
        "soft_drop_z_cut_02": 3,
    })
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
        # We want to load without a suffix, so the nominal needs to be empty. The actual name only
        # matters for loading the data. Everything else for the legacy production is handled manually.
        nominal="",
        variations=[],
        legacy_production=True,
    ) for _method in _grooming_methods_using_qm_result_conventions
}
_model_dependence_configuration.update({
    _method: unfolding_analysis.ModelDependenceConfiguration(
        nominal="pythia_fastsim",
        variations=["herwig_fastsim"],
    ) for _method in _grooming_methods_using_new_conventions
})
# Non-closure
# Apparently I used this non-closure for QM. I don't think it's necessary now since I better understand
# the uncertainties. Also, the stat clearly covers it.
non_closure_configuration: dict[str, unfolding_analysis.NonClosureConfiguration | None] = {
    _method: unfolding_analysis.NonClosureConfiguration(
        contributors=["reweight_pseudo_data"],
        approach_to_combining="max",
    ) for _method in _grooming_methods_using_qm_result_conventions
}
non_closure_configuration.update({
    _method: None
    for _method in _grooming_methods_using_new_conventions
})

# Either take model dependence or reweighted prior
# Model dependence is always preferred, but it may not have been analyzed yet for the a particular configuration
# (or in PbPb, it likely isn't possible since we don't have a reliable MC)
skip_reweighted_prior_in_systematics = True

# %% tags=["remove_cell"]
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

# %%
print(pp_R02_unfolding_systematics_outputs["dynamical_core"].keys())
print(pp_R02_unfolded_with_systematics["dynamical_core"].data.metadata["y_systematic"].keys())

# %%
plot_unfolding.steer_plotting_of_kt_unfolding_outputs(
    #grooming_methods=grooming_methods,
    grooming_methods=_grooming_methods_using_qm_result_conventions,
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
    unfolding_kt_display_range={
        grooming_method: (0.25, 6)
        for grooming_method in grooming_methods
    },
    relative_individual_systematic_ratio_range={
        grooming_method: (0.5, 1.5) for grooming_method in grooming_methods
    }
)

# %% [markdown]
# ### Semi-central

# %% tags=["remove_cell"]
collision_system = "PbPb"
event_activity = "semi_central"
#_double_counting_cut = "min_true_10_pt_hat_3"
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_truncation_shift = 5
_displaced_extremum = 10

###################
# Setup I/O options
###################
# Input directory location
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
_input_dir_tag = {
    _method: "parsl/2022-03-QM"
    for _method in _grooming_methods_using_qm_result_conventions
}
_input_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
_output_dir_tag = {
    #_method: input_dir_tag + "-from-QM22-results"
    _method: "2023-paper-plots-from-QM22-results"
    for _method in _grooming_methods_using_qm_result_conventions
}
_output_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
#_tag_after_suffix = "pass3"
#_tag_after_suffix = "pass3_peter_binning"
_tag_after_suffix_base = "pass3"
_tag_after_suffix = {
    grooming_method: _tag_after_suffix_base for grooming_method in grooming_methods
}
_tag_after_suffix["soft_drop_z_cut_04"] = f"{_tag_after_suffix_base}_merge_3_6"
####################################
# Grooming method dependent settings
####################################
_smeared_var_range = {
    "dynamical_core": helpers.KtRange(1, 8),
    "dynamical_kt": helpers.KtRange(1, 8),
    "dynamical_time": helpers.KtRange(1, 8),
    "soft_drop_z_cut_02": helpers.KtRange(0.25, 8),
    "soft_drop_z_cut_04": helpers.KtRange(0.25, 8),
    "dynamical_core_z_cut_02": helpers.KtRange(0.25, 8),
    "dynamical_kt_z_cut_02": helpers.KtRange(0.25, 8),
    "dynamical_time_z_cut_02": helpers.KtRange(0.25, 8),
}
_smeared_untagged_var = {
    "dynamical_core": helpers.KtRange(0, 1),
    "dynamical_kt": helpers.KtRange(0, 1),
    "dynamical_time": helpers.KtRange(0, 1),
    "soft_drop_z_cut_02": helpers.KtRange(0, 0.25),
    "soft_drop_z_cut_04": helpers.KtRange(0, 0.25),
    "dynamical_core_z_cut_02": helpers.KtRange(0, 0.25),
    "dynamical_kt_z_cut_02": helpers.KtRange(0, 0.25),
    "dynamical_time_z_cut_02": helpers.KtRange(0, 0.25),
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
if _use_qm22_inputs:
    _n_iter_compare.update({
        "dynamical_core": 3,
        "dynamical_kt": 3,
        "dynamical_time": 3,
        "soft_drop_z_cut_02": 3,
    })
_max_n_iter = {
    # Need +1 for convenience with range iteration
    "soft_drop_z_cut_04": 30,
}
_max_n_iter.update({
    grooming_method: 20 for grooming_method in grooming_methods if grooming_method != "soft_drop_z_cut_04"
})

# Double counting cut
# It's all the same here, but the QM22 results don't include the label
_double_counting_cut = {
    _method: ""
    for _method in _grooming_methods_using_qm_result_conventions
}
_double_counting_cut.update({
    _method: "OG_det_level_track_pt"
    for _method in _grooming_methods_using_new_conventions
})
# Model dependence.
_model_dependence_configuration = None
# Background subtraction configurations
_background_subtraction_configuration = {
    _method: unfolding_analysis.BackgroundSubtractionConfiguration(
        contributors=["Rmax005", "Rmax070"]
    )
    for _method in _grooming_methods_using_qm_result_conventions
}
_background_subtraction_configuration.update({
    _method: unfolding_analysis.BackgroundSubtractionConfiguration(
        contributors=["Rmax005", "Rmax050"]
    )
    for _method in _grooming_methods_using_new_conventions
})
# Add in the closure test to provide the non-closure uncertainty
_non_closure_configuration = {
    grooming_method: unfolding_analysis.NonClosureConfiguration(
        contributors=["reweight_response"],
        approach_to_combining="max",
    )
    for grooming_method in _grooming_methods_using_qm_result_conventions
}
_non_closure_configuration.update({
    grooming_method: unfolding_analysis.NonClosureConfiguration(
        contributors=["reweight_response", "reweight_pseudo_data", "thermal_model"],
        approach_to_combining="max",
    )
    for grooming_method in _grooming_methods_using_new_conventions
})

# %% tags=["remove_cell"]
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

# %%
print(list(semi_central_R02_unfolding_systematics_outputs["dynamical_kt_z_cut_02"].keys()))
print(list(semi_central_R02_unfolding_closure_outputs["dynamical_kt_z_cut_02"].keys()))

# %%
plot_unfolding.steer_plotting_of_kt_unfolding_outputs(
    grooming_methods=grooming_methods,
    unfolded_with_systematics=semi_central_R02_unfolded_with_systematics,
    unfolding_systematics_outputs=semi_central_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=semi_central_R02_unfolding_closure_outputs,
    plot=True,
    plot_png=False,
    plot_systematic_breakdown=True,
    plot_systematics=False,
    plot_closures=False,
    prior_variation_output_name="reweight_prior",
    unfolding_kt_display_range={
        grooming_method: (0.25, 6) if "z_cut" in grooming_method else (1, 6)
        for grooming_method in grooming_methods
    },
    relative_individual_systematic_ratio_range={
        grooming_method: (0.6, 1.4) if "z_cut" in grooming_method else(0.25, 1.75)
        for grooming_method in grooming_methods
    }
)

# %%
list(semi_central_R02_unfolded_with_systematics.keys())

# %% [markdown]
# ### Central

# %% tags=["remove_cell"]
collision_system = "PbPb"
event_activity = "central"
_smeared_jet_pt_range = helpers.JetPtRange(40, 120)
_truncation_shift = 5
_displaced_extremum = 10

###################
# Setup I/O options
###################
# Input directory location
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
_input_dir_tag = {
    _method: input_dir_tag for _method in grooming_methods
}
_output_dir_tag = {
    _method: input_dir_tag for _method in grooming_methods
}
#_tag_after_suffix = "pass3"
#_tag_after_suffix = "pass3_peter_binning"
_tag_after_suffix_base = "pass3"
_tag_after_suffix = {
    grooming_method: _tag_after_suffix_base for grooming_method in grooming_methods
}
_tag_after_suffix["soft_drop_z_cut_04"] = f"{_tag_after_suffix_base}_merge_3_6"
####################################
# Grooming method dependent settings
####################################
_smeared_var_range = {
    "dynamical_core": helpers.KtRange(1.5, 8),
    "dynamical_kt": helpers.KtRange(1.5, 8),
    "dynamical_time": helpers.KtRange(1.5, 8),
    "soft_drop_z_cut_02": helpers.KtRange(0.25, 8),
    "soft_drop_z_cut_04": helpers.KtRange(0.25, 8),
    "dynamical_core_z_cut_02": helpers.KtRange(0.25, 8),
    "dynamical_kt_z_cut_02": helpers.KtRange(0.25, 8),
    "dynamical_time_z_cut_02": helpers.KtRange(0.25, 8),
}
_smeared_untagged_var = {
    "dynamical_core": helpers.KtRange(0, 1.5),
    "dynamical_kt": helpers.KtRange(0, 1.5),
    "dynamical_time": helpers.KtRange(0, 1.5),
    "soft_drop_z_cut_02": helpers.KtRange(0, 0.25),
    "soft_drop_z_cut_04": helpers.KtRange(0, 0.25),
    "dynamical_core_z_cut_02": helpers.KtRange(0, 0.25),
    "dynamical_kt_z_cut_02": helpers.KtRange(0, 0.25),
    "dynamical_time_z_cut_02": helpers.KtRange(0, 0.25),
}
_n_iter_compare = {
    "dynamical_core": 11,
    "dynamical_kt": 9,
    "dynamical_time": 10,
    "soft_drop_z_cut_02": 8,
    "dynamical_core_z_cut_02": 8,
    "dynamical_kt_z_cut_02": 8,
    "dynamical_time_z_cut_02": 8,
    "soft_drop_z_cut_04": 17,
}
_max_n_iter = {
    # Need +1 for convenience with range iteration
    "soft_drop_z_cut_04": 30,
}
_max_n_iter.update({
    grooming_method: 20 for grooming_method in grooming_methods if grooming_method != "soft_drop_z_cut_04"
})

# Double counting cut
_double_counting_cut = {
    _method: "min_true_10_pt_hat_3"
    for _method in grooming_methods
}
# Model dependence.
_model_dependence_configuration = None
# Background subtraction configurations
_background_subtraction_configuration = {
    _method: unfolding_analysis.BackgroundSubtractionConfiguration(
        contributors=["Rmax005", "Rmax050"]
    )
    for _method in grooming_methods
}
# Add in the closure test to provide the non-closure uncertainty
_non_closure_configuration = {
    grooming_method: unfolding_analysis.NonClosureConfiguration(
        contributors=["reweight_pseudo_data", "reweight_response", "thermal_model"],
        approach_to_combining="max",
    )
    for grooming_method in grooming_methods
}

# %% tags=["remove_cell"]
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
central_R02_unfolded_with_systematics, central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=central_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=central_R02_unfolding_closure_outputs,
    true_jet_pt_range=true_jet_pt_range,
    model_dependence_configuration=_model_dependence_configuration,
    non_closure_configuration=_non_closure_configuration,
    background_subtraction_configuration=_background_subtraction_configuration,
)

# %%
plot_unfolding.steer_plotting_of_kt_unfolding_outputs(
    grooming_methods=grooming_methods,
    unfolded_with_systematics=central_R02_unfolded_with_systematics,
    unfolding_systematics_outputs=central_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=central_R02_unfolding_closure_outputs,
    plot=True,
    plot_png=False,
    plot_systematic_breakdown=True,
    plot_systematics=True,
    plot_closures=True,
    prior_variation_output_name="reweight_prior",
    unfolding_kt_display_range={
        grooming_method: (0.25, 6) if "z_cut" in grooming_method else (1.5, 6)
        for grooming_method in grooming_methods
    },
    relative_individual_systematic_ratio_range={
        grooming_method: (0.6, 1.4) if "z_cut" in grooming_method else (0.25, 1.75)
        for grooming_method in grooming_methods
    }
)

# %%
print(list(central_R02_unfolded_with_systematics.keys()))
print(list(central_R02_unfolding_closure_outputs["dynamical_core"].keys()))
print(list(central_R02_unfolding_systematics_outputs["dynamical_core"].keys()))

# %% [markdown]
# ## R = 0.4

# %%
# General setup
plot = False
substructure_variable = "kt"
true_jet_pt_range = helpers.JetPtRange(60, 80)
jet_R = 0.4
jet_R_str = f"R{int(jet_R*10):02}"
grooming_methods = [
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
    "soft_drop_z_cut_02",
    "dynamical_core_z_cut_02",
    "dynamical_kt_z_cut_02",
    "dynamical_time_z_cut_02",
    "soft_drop_z_cut_04",
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
input_dir_tag = "2023-02-HP"
###################
# Setup I/O options
###################
# NOTE: Technically, these are HP2020 results rather than QM2022, but good enough
_use_qm22_inputs = True
_grooming_methods_using_qm_result_conventions = _OG_grooming_methods if _use_qm22_inputs else []
_grooming_methods_using_new_conventions = _new_grooming_methods if _use_qm22_inputs else grooming_methods

_output_dir = output_dir / "comparison" / "unfolding" / "2023-paper-plots" / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### pp

# %% tags=["remove_cell"]
# pp
collision_system = "pp"
event_activity = "pp"
_smeared_var_range: helpers.KtRange | dict[str, helpers.KtRange] = helpers.KtRange(0.25, 8)
_smeared_jet_pt_range = helpers.JetPtRange(20, 85)
_truncation_shift = 3
_displaced_extremum = 10
#_displaced_extremum = 20

###################
# Setup I/O options
###################
# Input directory location
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
# TODO: I don't think this is correct. Just need to re
_input_dir_tag = {
    _method: "parsl"
    for _method in _grooming_methods_using_qm_result_conventions
}
_input_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
_output_dir_tag: dict[str, str | None] = {
    #_method: input_dir_tag + "-from-QM22-results"
    _method: "2023-paper-plots-from-HP20-results"
    for _method in _grooming_methods_using_qm_result_conventions
}
_output_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
#_tag_after_suffix = "2_4_split"
_tag_after_suffix = {
    grooming_method: "" for grooming_method in grooming_methods
}
####################################
# Grooming method dependent settings
####################################
_smeared_untagged_var = {
    "dynamical_core": helpers.KtRange(0.25, 0.25),
    "dynamical_kt": helpers.KtRange(0.25, 0.25),
    "dynamical_time": helpers.KtRange(0.25, 0.25),
    "soft_drop_z_cut_02": helpers.KtRange(0, 0.25),
    "dynamical_core_z_cut_02": helpers.KtRange(0.0, 0.25),
    "dynamical_kt_z_cut_02": helpers.KtRange(0.0, 0.25),
    "dynamical_time_z_cut_02": helpers.KtRange(0.0, 0.25),
    "soft_drop_z_cut_04": helpers.KtRange(0, 0.25),
}
_n_iter_compare = {
    # TODO: To be optimized!
    "dynamical_core": 5,
    "dynamical_kt": 5,
    "dynamical_time": 5,
    "soft_drop_z_cut_02": 5,
    "dynamical_core": 5,
    "dynamical_kt": 5,
    "dynamical_time": 5,
    "soft_drop_z_cut_02": 5,
}
if _use_qm22_inputs:
    _n_iter_compare.update({
        "dynamical_core": 5,
        "dynamical_kt": 5,
        "dynamical_time": 5,
        "soft_drop_z_cut_02": 5,
    })
_max_n_iter: dict[str, int | None] = {
    # Need +1 for convenience with range iteration
    # TODO: Update this once we've rerun it.
    #"soft_drop_z_cut_04": 30,
    "soft_drop_z_cut_04": 20,
}
_max_n_iter.update({
    grooming_method: 20 for grooming_method in grooming_methods if grooming_method != "soft_drop_z_cut_04"
})

# Model dependence.
# Varies here by grooming method because we need to be able to support the QM preliminaries (for now).
_model_dependence_configuration: dict[str, unfolding_analysis.ModelDependenceConfiguration | None] = {
    _method: unfolding_analysis.ModelDependenceConfiguration(
        # We want to load without a suffix, so the nominal needs to be empty. The actual name only
        # matters for loading the data. Everything else for the legacy production is handled manually.
        nominal="",
        variations=[],
        legacy_production=True,
    ) for _method in _grooming_methods_using_qm_result_conventions
}
_model_dependence_configuration.update({
    _method: unfolding_analysis.ModelDependenceConfiguration(
        nominal="pythia_fastsim",
        variations=["herwig_fastsim"],
    ) for _method in _grooming_methods_using_new_conventions
})
# Non-closure
# Apparently I used this non-closure for QM. I don't think it's necessary now since I better understand
# the uncertainties. Also, the stat clearly covers it.
non_closure_configuration: dict[str, unfolding_analysis.NonClosureConfiguration | None] = {
    _method: unfolding_analysis.NonClosureConfiguration(
        contributors=["reweight_pseudo_data"],
        approach_to_combining="max",
    ) for _method in _grooming_methods_using_qm_result_conventions
}
non_closure_configuration.update({
    _method: None
    for _method in _grooming_methods_using_new_conventions
})

# Either take model dependence or reweighted prior
# Model dependence is always preferred, but it may not have been analyzed yet for the a particular configuration
# (or in PbPb, it likely isn't possible since we don't have a reliable MC)
skip_reweighted_prior_in_systematics = True

# %% tags=["remove_cell"]
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
pp_R04_unfolded_with_systematics, pp_R04_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=pp_R04_unfolding_systematics_outputs,
    unfolding_closure_outputs=pp_R04_unfolding_closure_outputs,
    true_jet_pt_range=true_jet_pt_range,
    model_dependence_configuration=_model_dependence_configuration,
    non_closure_configuration=non_closure_configuration,
    background_subtraction_configuration=None,
)

# %%
print(pp_R04_unfolding_systematics_outputs["dynamical_core"].keys())
print(pp_R04_unfolded_with_systematics["dynamical_core"].data.metadata["y_systematic"].keys())

# %%
plot_unfolding.steer_plotting_of_kt_unfolding_outputs(
    #grooming_methods=grooming_methods,
    grooming_methods=_grooming_methods_using_qm_result_conventions,
    unfolded_with_systematics=pp_R04_unfolded_with_systematics,
    unfolding_systematics_outputs=pp_R04_unfolding_systematics_outputs,
    unfolding_closure_outputs=pp_R04_unfolding_closure_outputs,
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
    unfolding_kt_display_range={
        grooming_method: (0.25, 8)
        for grooming_method in grooming_methods
    },
    relative_individual_systematic_ratio_range={
        grooming_method: (0.5, 1.5) for grooming_method in grooming_methods
    }
)

# %% [markdown]
# ## Models

# %% [markdown]
# ### Pythia
#
# Already loaded in the "true_reference" variables. We just need to create the model objects to wrap them up

# %% [markdown]
# #### R = 0.2

# %%
pythia_predictions_R02 = model_calculations.ModelCalculation(
    name="pythia8",
    label_pp="PYTHIA8 Monash 2013",
    label_AA="",
    normalized=True,
    grooming_methods=grooming_methods,
    metadata={"jet_R": 0.2},
    pp=pp_R02_true_reference,
)

# %% [markdown]
# #### R = 0.4

# %%
pythia_predictions_R02 = model_calculations.ModelCalculation(
    name="pythia8",
    label_pp="PYTHIA8 Monash 2013",
    label_AA="",
    normalized=True,
    grooming_methods=grooming_methods,
    metadata={"jet_R": 0.4},
    pp=pp_R04_true_reference,
)

# %% [markdown]
# ### Jetscape
#

# %% [markdown]
# #### R = 0.2
#
# Predictions from Yasuki provided for HP 2023 for R = 0.2

# %%
jetscape_R02 = model_calculations.Jetscape(
    base_dir=Path("output/comparison/models/jetscape/2023-03-23-yasuki"),
    needs_normalization=False,
    metadata={"jet_R": 0.2},
)
jetscape_predictions_R02 = jetscape_R02.load_predictions()

# %% [markdown]
# #### R = 0.4, TBD

# %% [markdown]
# ### Sherpa
#

# %% [markdown]
# #### R = 0.2, TBD

# %% [markdown]
# #### R = 0.4
#
# R = 0.4 predctions performed by Leticia

# %%
sherpa_ahadic = plot_unfolding.load_sherpa_predictions(
    filename = output_dir / "comparison" / "models" / "sherpa" / "SherpaHistograms_Ahadic_R04_merged12.root", jet_R_values=[0.2, 0.4]
)
sherpa_lund = plot_unfolding.load_sherpa_predictions(
    filename = output_dir / "comparison" / "models" / "sherpa" / "SherpaHistograms_Lund_R04_merged12.root", jet_R_values=[0.2, 0.4]
)

# %% [markdown]
# ### Caucal et al. Analytical Calculations

# %% [markdown]
# #### R = 0.2, TBD

# %% [markdown]
# #### R = 0.4
#
# Provided by Alba et al for DyG a=1.0, 2.0

# %%
# NOTE: We only have R=0.4 for DyG kt and DyG time as of April 2023
caucal_analytical_pp_R04 = model_calculations.CaucalAnalyticalCalculations(
    base_dir=output_dir / "comparison" / "models" / "caucal_analytical",
    needs_normalization=False,
    metadata={
        # NOTE: Equivalent to [0.5, 1, 2, 4, 6, 8]
        #"bin_edges": pp_R04_unfolded_with_systematics["dynamical_kt"].data.axes[0].bin_edges[2:-1]
        "bin_edges": [0.5, 1, 2, 4, 6, 8]
    },
)
caucal_analytical_pp_R04_predictions = caucal_analytical_pp_R04.load_predictions()

# %%
caucal_analytical_pp_R04_predictions

# %% [markdown] toc-hr-collapsed=true
# ### Hybrid model

# %% [markdown]
# #### R = 0.2
#
# Provided by Dani for Hard Probes 2023

# %%
# NOTE: We use the narrower bin edges than the data since we're just plotting a ratio.
#       And since we don't the data by the ratio, we can just use whatever binning,
#       including the same for central and semi-central!
bin_edges = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2.0, 3.0, 4., 5., 6., 7., 8.]
hybrid_model_with_wake_with_moliere_R02 = model_calculations.HybridModel(
    base_dir=Path("/Users/REhlers/software/dev/substructure/output/comparison/models/hybrid/HP2023/ForRaymond/results_kt_raymond"),
    metadata={
        "include_elastic": True,
        "include_wake": True,
        "bin_edges": bin_edges,
    },
)
hybrid_model_with_wake_with_moliere_predictions_R02 = hybrid_model_with_wake_with_moliere_R02.load_predictions()
hybrid_model_with_wake_without_moliere_R02 = model_calculations.HybridModel(
    base_dir=Path("/Users/REhlers/software/dev/substructure/output/comparison/models/hybrid/HP2023/ForRaymond/results_kt_raymond"),
    metadata={
        "include_elastic": False,
        "include_wake": True,
        "bin_edges": bin_edges,
    },
)
hybrid_model_with_wake_without_moliere_predictions_R02 = hybrid_model_with_wake_without_moliere_R02.load_predictions()

# %%
hybrid_model_with_wake_without_moliere_predictions_R02.semi_central_ratio["soft_drop_z_cut_04"]

# %% [markdown]
# # Plots

# %% [markdown]
# ### Setup

# %%
alice_status = "final"
plot_output_dir_tag = "2023-paper-plots"

# %% [markdown]
# ## pp grooming methods comparison

# %% [markdown]
# ### R = 0.2

# %%
from importlib import reload

reload(plot_unfolding)

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for _temp_grooming_methods, _reference_method, _label in [
    (["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"], "dynamical_core", "0"),
    (["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"], "soft_drop_z_cut_02", "1"),
    (["soft_drop_z_cut_04", "dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "soft_drop_z_cut_02", "2"),
    (["soft_drop_z_cut_02", "soft_drop_z_cut_04", "dynamical_kt", "dynamical_kt_z_cut_02"], "soft_drop_z_cut_02", "3"),
    (["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "soft_drop_z_cut_02", "4"),
    (["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "dynamical_kt_z_cut_02", "5"),
]:
    plot_unfolding.plot_grooming_comparisons_for_single_system(
        hists=pp_R02_unfolded_with_systematics,
        grooming_methods=_temp_grooming_methods,
        reference_grooming_method=_reference_method,
        collision_system="pp",
        collision_system_key="pp_5TeV",
        output_dir=_output_dir,
        kt_range=helpers.KtRange(0.25, 6),
        figure_kt_range=helpers.KtRange(0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
        label=_label,
    )

# %%

# %% [markdown]
# ## pp comparison to models
#
# ### R = 0.2

# %%
from importlib import reload

reload(plot_unfolding)

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / input_dir_tag / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_unfolding.plot_grooming_model_comparisons_for_single_system(
    hists=pp_R02_unfolded_with_systematics,
    models={
        # TODO: Need to update the jetscape binning. Whoops...
        #"jetscape": jetscape_pp["R02"],
        "pythia": pp_R02_true_reference,
    },
    #grooming_methods=grooming_methods,
    grooming_methods=list(reversed(_grooming_methods_using_new_conventions)),
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 6),
    figure_kt_range=helpers.KtRange(0, 6.25),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
)

# %% [markdown]
# ## PbPb grooming methods comparison
#
# ### R = 0.2

# %%
from importlib import reload

reload(plot_unfolding)

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for _collision_system, _hists in [
    ("semi_central", semi_central_R02_unfolded_with_systematics),
    ("central", central_R02_unfolded_with_systematics),
]:
    for _temp_grooming_methods, _reference_method, _label in [
        (["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"], "dynamical_core", "0"),
        (["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"], "soft_drop_z_cut_02", "1"),
        (["soft_drop_z_cut_04", "dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "soft_drop_z_cut_02", "2"),
        (["soft_drop_z_cut_02", "soft_drop_z_cut_04", "dynamical_kt", "dynamical_kt_z_cut_02"], "soft_drop_z_cut_02", "3"),
        (["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "soft_drop_z_cut_02", "4"),
        (["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "dynamical_kt_z_cut_02", "5"),
    ]:
        plot_unfolding.plot_grooming_comparisons_for_single_system(
            hists=_hists,
            grooming_methods=_temp_grooming_methods,
            reference_grooming_method=_reference_method,
            collision_system=_collision_system,
            collision_system_key="PbPb",
            output_dir=_output_dir,
            kt_range={
                "dynamical_core": helpers.KtRange(2, 6) if _collision_system == "semi_central" else helpers.KtRange(3, 6),
                "dynamical_kt": helpers.KtRange(2, 6) if _collision_system == "semi_central" else helpers.KtRange(3, 6),
                "dynamical_time": helpers.KtRange(2, 6) if _collision_system == "semi_central" else helpers.KtRange(3, 6),
                "soft_drop_z_cut_02": helpers.KtRange(0.25, 6),
                "dynamical_core_z_cut_02": helpers.KtRange(0.25, 6),
                "dynamical_kt_z_cut_02": helpers.KtRange(0.25, 6),
                "dynamical_time_z_cut_02": helpers.KtRange(0.25, 6),
                "soft_drop_z_cut_04": helpers.KtRange(0.25, 6),
            },
            figure_kt_range=helpers.KtRange(0, 6.25),
            jet_R_str=jet_R_str,
            alice_status=alice_status,
            label=_label,
        )

# %% [markdown]
# ## PbPb-pp comparison by each grooming methods
#
# ### R = 0.2

# %%
from importlib import reload

reload(plot_unfolding)

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

_collision_systems_with_stats_tests_results = ["semi_central", "central"]
_ks_test_results = {_k: {} for _k in _collision_systems_with_stats_tests_results}
_chi2_test_results = {_k: {} for _k in _collision_systems_with_stats_tests_results}
for grooming_method in grooming_methods:
    logger.info(f"Processing {grooming_method}")
    _ks_test_result, _chi2_test_result = plot_unfolding.plot_pp_PbPb_comparison(
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
        plausible_stat_test_parameters=[1.0, 2.0],
        calculate_hist_stat_tests_KS_chi2=True,
    )
    if _ks_test_result:
        for _system in _collision_systems_with_stats_tests_results:
            _ks_test_results[_system][grooming_method] = _ks_test_result[_system]
    if _chi2_test_result:
        for _system in _collision_systems_with_stats_tests_results:
            _chi2_test_results[_system][grooming_method] = _chi2_test_result[_system]

# %%
import pprint

pprint.pprint(_ks_test_results["semi_central"].keys())
pprint.pprint([f"{v:.3}" for v in _ks_test_results["semi_central"].values()])
pprint.pprint([f"{v:.3}" for v in _ks_test_results["central"].values()])
pprint.pprint([f"{v:.3}" for v in _chi2_test_results["semi_central"].values()])
pprint.pprint([f"{v:.3}" for v in _chi2_test_results["central"].values()])

# %%

# %% [markdown]
# ## PbPb-pp comparison to models
#
# ### R = 0.2

# %%
from importlib import reload

reload(plot_unfolding)

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

#for event_activity in ["semi_central", "central"]:
for event_activity in ["central"]:
    for grooming_method in grooming_methods:
        _additional_hists = {
            "semi_central": semi_central_R02_unfolded_with_systematics[grooming_method],
            "central": central_R02_unfolded_with_systematics[grooming_method],
        }
        plot_unfolding.plot_pp_PbPb_comparison(
            hists={
                "pp": pp_R02_unfolded_with_systematics[grooming_method],
                event_activity: _additional_hists[event_activity],
                #"semi_central": semi_central_R02_unfolded_with_systematics[grooming_method],
                #"central": central_R02_unfolded_with_systematics[grooming_method],
            },
            models={
                #"hybrid_without_moliere": semi_central_hybrid_model_ratio["hybrid_without_moliere"][jet_R_str],
                #"hybrid_moliere": semi_central_hybrid_model_ratio["hybrid_moliere"][jet_R_str],
                "hybrid_without_moliere": hybrid_model_with_wake_without_moliere_predictions.ratio(event_activity),
                "hybrid_moliere": hybrid_model_with_wake_with_moliere_predictions.ratio(event_activity),
                "jetscape": jetscape_predictions.ratio(event_activity),
            },
            grooming_method=grooming_method,
            output_dir=_output_dir,
            event_activity_to_kt_range={
                "pp": helpers.KtRange(0.25, 6),
                "semi_central": helpers.KtRange(0.25, 6) if "z_cut" in grooming_method else helpers.KtRange(2, 6),
                "central": helpers.KtRange(0.25, 6) if "z_cut" in grooming_method else helpers.KtRange(3, 6),
            },
            kt_display_range=(0.0, 6.25),
            jet_R_str=jet_R_str,
            alice_status=alice_status,
            additional_label=event_activity
        )

# %%
