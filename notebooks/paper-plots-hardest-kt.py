# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv-3.11
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
from jet_substructure.analysis import (
    model_calculations,
    plot_paper,
    plot_unfolding,
    unfolding_analysis,
)
from jet_substructure.base import helpers
from mammoth import helpers as mammoth_helpers

# %load_ext autoreload
# %autoreload 2

mammoth_helpers.setup_logging(level=logging.DEBUG)
# Quiet down loud logging from other libraries
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("boost_histogram").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
logging.getLogger("fsspec").setLevel(logging.INFO)

logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
logging.getLogger("pachyderm.binned_data").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# General settings
embed_images = False
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load data

# %%
# Quick separate setup if I want interactive plots

# #%matplotlib inline
# #%config InlineBackend.figure_formats = ["png", "pdf"]
# Don't show mpl images inline. We'll handle displaying them separately.
plt.ioff()
pb.configure(disable_interactive_backend=True)
# Ensure the axes are legible on a dark background
mpl.rcParams['figure.facecolor'] = 'w'

# %% [markdown]
# ## R = 0.2

# %%
# General setup
plot = False
substructure_variable = "kt"
true_jet_pt_range = helpers.JetPtRange(60, 80)
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
# Uncertainty options
#_unfolding_related_systematic_treatment = "max"
_unfolding_related_systematic_treatment = "std_dev"
#_unfolding_related_systematic_treatment = "all"
calculate_quadrature_assuming_all_are_symmetric = True

grooming_methods = [
    #"dynamical_core",
    "dynamical_kt",
    #"dynamical_time",
    "soft_drop_z_cut_02",
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
# For final thermal model tests
#input_dir_tag = "2024-paper"
# End final thermal model tests
###################
# Setup I/O options
###################
_use_qm22_inputs = False
_grooming_methods_using_qm_result_conventions = _OG_grooming_methods if _use_qm22_inputs else []
_grooming_methods_using_new_conventions = _new_grooming_methods if _use_qm22_inputs else grooming_methods

# Unused...
#_output_dir = output_dir / "comparison" / "unfolding" / "2024-paper-plots" / jet_R_str
#_output_dir.mkdir(parents=True, exist_ok=True)

# %%
# NOTE: This is copied from the "Plots" section. I just need it all over, and easier to put it here.
plot_output_dir_tag = "2024-paper-plots"
# For final thermal model tests
#plot_output_dir_tag = "2024-test-paper-plots"
# End final thermal model tests
grooming_methods_for_letter = ["dynamical_kt", "soft_drop_z_cut_02"]

def PbPb_kt_measured_range_by_grooming_method(event_activity: str) -> dict[str, helpers.KtRange]:
    return {
        "dynamical_core": helpers.KtRange(2, 6) if event_activity == "semi_central" else helpers.KtRange(3, 6),
        "dynamical_kt": helpers.KtRange(2, 6) if event_activity == "semi_central" else helpers.KtRange(3, 6),
        "dynamical_time": helpers.KtRange(2, 6) if event_activity == "semi_central" else helpers.KtRange(3, 6),
        "soft_drop_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_core_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_kt_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_time_z_cut_02": helpers.KtRange(0.25, 6),
        "soft_drop_z_cut_04": helpers.KtRange(0.25, 6),
    }


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
#_model_dependence_configuration: dict[str, unfolding_analysis.ModelDependenceConfiguration | None] = {
#    _method: unfolding_analysis.ModelDependenceConfiguration(
#        # We want to load without a suffix, so the nominal needs to be empty. The actual name only
#        # matters for loading the data. Everything else for the legacy production is handled manually.
#        nominal="",
#        variations=[],
#        legacy_production=True,
#    ) for _method in _grooming_methods_using_qm_result_conventions
#}
_model_dependence_configuration = {
    _method: unfolding_analysis.ModelDependenceConfiguration(
        nominal="pythia_fastsim",
        variations=["herwig_fastsim"],
        approach_to_combining="max",
    ) for _method in _grooming_methods_using_new_conventions
}
# Non-closure
# Apparently I used this non-closure for QM. I don't think it's necessary now since I better understand
# the uncertainties. Also, the stat clearly covers it.
#non_closure_configuration: dict[str, unfolding_analysis.NonClosureConfiguration | None] = {
#    _method: unfolding_analysis.NonClosureConfiguration(
#        contributors=["reweight_pseudo_data"],
#        approach_to_combining="max",
#    ) for _method in _grooming_methods_using_qm_result_conventions
#}
non_closure_configuration = {
    _method: None
    for _method in _grooming_methods_using_new_conventions
}
# Smoothing for systematic uncertainty contributions
_uncertainty_smoothing_configuration = {
    # Reasoning:
    # - Model dependence fluctuates larger and then small at high kt, which seems unphysical.
    # - NOTE: The random binning fluctuates way up at 3-4, but since we combine them together
    #         as the std dev (via _unfolding_related_systematic_treatment), we in practice don't
    #         need to smooth it out due to that bin. (i.e. the unfolding is rarely the
    #         leading uncertainty)
    grooming_method: unfolding_analysis.UncertaintySmoothingConfiguration(
        contributors={"model_dependence": 1, "unfolding": 1},
        # It's pp, so there's no reduced kt range selection (we just select the full range)
        kt_range_to_smooth=helpers.KtRange(0.25, 6),
    )
    for grooming_method in _grooming_methods_using_new_conventions
}
_uncertainty_smoothing_configuration.update({
    "dynamical_kt": unfolding_analysis.UncertaintySmoothingConfiguration(
        # Reasoning:
        # - model dependence: 3-4 jumps quite large, so smooth it out.
        # - It doesn't need the unfolding smoothing because it only jumps 1-2% at 1.5-2
        contributors={"model_dependence": 1},
        #contributors={},
        kt_range_to_smooth=helpers.KtRange(0.25, 6),
    ),
    "soft_drop_z_cut_02": unfolding_analysis.UncertaintySmoothingConfiguration(
        # Reasoning:
        # - model dependence: 3-4 jumps quite large
        # - unfolding: 1.5-2 jumps larger
        contributors={"model_dependence": 1, "unfolding": 1},
        kt_range_to_smooth=helpers.KtRange(0.25, 6),
    ),
})

# We include the prior in the unfolding uncertaintiy.
# The model (generator) dependence is via the modification of the model used for the response matrix.
skip_reweighted_prior_in_systematics = False

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

# We don't need to adjust the n_iter for any of the model dependence, so we just leave that out here...

# Focus down onto just the unfolded distributions
pp_R02_unfolded_with_systematics, pp_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=pp_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=pp_R02_unfolding_closure_outputs,
    true_jet_pt_range=true_jet_pt_range,
    calculate_quadrature_assuming_all_are_symmetric=calculate_quadrature_assuming_all_are_symmetric,
    unfolding_related_systematic_treatment=_unfolding_related_systematic_treatment,
    smooth_systematic_uncertainty_contributions=_uncertainty_smoothing_configuration,
    model_dependence_configuration=_model_dependence_configuration,
    non_closure_configuration=non_closure_configuration,
    background_subtraction_configuration=None,
)

# %%
print(pp_R02_unfolding_systematics_outputs["dynamical_kt"].keys())
print(pp_R02_unfolded_with_systematics["dynamical_kt"].data.metadata["y_systematic"].keys())
print(pp_R02_unfolded_with_systematics["dynamical_kt"].data.metadata["y_systematic"]["quadrature"])

# %%
plot_unfolding.steer_plotting_of_kt_unfolding_outputs(
    grooming_methods=grooming_methods,
    unfolded_with_systematics=pp_R02_unfolded_with_systematics,
    unfolding_systematics_outputs=pp_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=pp_R02_unfolding_closure_outputs,
    plot=True,
    plot_png=False,
    plot_systematic_breakdown=True,
    plot_systematics=False,
    plot_closures=False,
    unfolding_related_systematic_treatment=_unfolding_related_systematic_treatment,
    # NOTE: All of the commentary below is **only** related to plotting of the prior for selecting the
    #       number of iterations. The actual prior systematic is evaluated properly!
    # NOTE: For the prior variation, passing the HERWIG model dependence includes both:
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
    #_method: "2023-paper-plots-from-QM22-results"
    _method: "2023-pre-PF-DyG-untagged-tests"
    for _method in _grooming_methods_using_qm_result_conventions
}
_output_dir_tag.update({
    _method: input_dir_tag for _method in _grooming_methods_using_new_conventions
})
#_tag_after_suffix = "pass3"
#_tag_after_suffix = "pass3_peter_binning"
_tag_after_suffix_base = "pass3"
#_tag_after_suffix_base = "pass3_no_smeared_untagged"
#_tag_after_suffix_base = "pass3_no_untagged"
_tag_after_suffix = {
    grooming_method: _tag_after_suffix_base for grooming_method in grooming_methods
}
#_tag_after_suffix["soft_drop_z_cut_04"] = f"{_tag_after_suffix_base}_merge_3_6"
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
    "dynamical_core": helpers.KtRange(1, 1),
    "dynamical_kt": helpers.KtRange(1, 1),
    "dynamical_time": helpers.KtRange(1, 1),
    "soft_drop_z_cut_02": helpers.KtRange(0, 0.25),
    "soft_drop_z_cut_04": helpers.KtRange(0, 0.25),
    "dynamical_core_z_cut_02": helpers.KtRange(0, 0.25),
    "dynamical_kt_z_cut_02": helpers.KtRange(0, 0.25),
    "dynamical_time_z_cut_02": helpers.KtRange(0, 0.25),
}
_n_iter_compare = {
    "dynamical_core": 7,
    "dynamical_kt": 7,
    "dynamical_time": 7,
    "soft_drop_z_cut_02": 9,
    "dynamical_core_z_cut_02": 9,
    "dynamical_kt_z_cut_02": 9,
    "dynamical_time_z_cut_02": 9,
    "soft_drop_z_cut_04": 10,
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
    # Need larger range to select a higher iter for the model dependence
    "soft_drop_z_cut_02": 30,
    "dynamical_kt": 30,
}
_max_n_iter.update({
    grooming_method: 20 for grooming_method in grooming_methods if grooming_method not in ["dynamical_kt", "soft_drop_z_cut_02","soft_drop_z_cut_04"]
})

# Double counting cut
# It's all the same here, but the QM22 results don't include the label
#_double_counting_cut = {
#    _method: ""
#    for _method in _grooming_methods_using_qm_result_conventions
#}
_double_counting_cut = {
    _method: "min_true_10_pt_hat_3"
    for _method in _grooming_methods_using_new_conventions
}
# Model dependence.
# We use the central model dependence as a proxy since we don't have a semi-central model dependence available.
# Consequently, we've copied over these values from the central configuration
_model_dependence_configuration = {}
_model_dependence_configuration.update({
    _method: unfolding_analysis.ModelDependenceConfiguration(
        # Use the default n_iter for all grooming methods, but optimize below as necessary.
        nominal="embed_pythia_fastsim",
        variations=["embed_jewel_no_recoils_fastsim"],
        approach_to_combining="max",
        skip_double_counting_label=False,
    ) for _method in _grooming_methods_using_new_conventions
})
_model_dependence_update_n_iter = {
    # Selected based on usual convergence criteria...
    "dynamical_kt": {
        "embed_pythia_fastsim": 17,
        # The usual n_iter seems to be reasonable for JEWEL
        # NOTE: We still set it manually here for this semi-central case because it needs to match the **central** n_iter value instead...
        #       (otherwise, it could be None, as in the case when we use it for central)
        "embed_jewel_no_recoils_fastsim": 9,
    },
    "soft_drop_z_cut_02": {
        "embed_pythia_fastsim": 17,
        "embed_jewel_no_recoils_fastsim": 17,
    }
}
# Background subtraction configurations
#_background_subtraction_configuration = {
#    _method: unfolding_analysis.BackgroundSubtractionConfiguration(
#        contributors=["Rmax005", "Rmax070"]
#    )
#    for _method in _grooming_methods_using_qm_result_conventions
#}
_background_subtraction_configuration = {
    _method: unfolding_analysis.BackgroundSubtractionConfiguration(
        contributors=["Rmax005", "Rmax050"]
    )
    for _method in _grooming_methods_using_new_conventions
}
# Add in the closure test to provide the non-closure uncertainty
#_non_closure_configuration = {
#    grooming_method: unfolding_analysis.NonClosureConfiguration(
#        contributors=["reweight_response"],
#        approach_to_combining="max",
#    )
#    for grooming_method in _grooming_methods_using_qm_result_conventions
#}
_non_closure_configuration = {
    grooming_method: unfolding_analysis.NonClosureConfiguration(
        # NOTE: I exclude the reweight_response because I think it's overlapping with the model dependence
        #contributors=["reweight_response", "reweight_pseudo_data", "thermal_model"],
        contributors=["reweight_pseudo_data", "thermal_model"],
        approach_to_combining="max",
    )
    for grooming_method in _grooming_methods_using_new_conventions
}

# Smoothing for systematic uncertainty contributions
_uncertainty_smoothing_configuration = {
    # Reasoning:
    grooming_method: unfolding_analysis.UncertaintySmoothingConfiguration(
        # Default to no smoothing.
        contributors={},
        kt_range_to_smooth=PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central")[grooming_method],
    )
    for grooming_method in _grooming_methods_using_new_conventions
}
_uncertainty_smoothing_configuration.update({
    "dynamical_kt": unfolding_analysis.UncertaintySmoothingConfiguration(
        # Nothing seems to be fluctuating so much, so seems to be okay.
        # (It's entirely dominated by the model dependence...)
        contributors={},
        kt_range_to_smooth=PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central")["dynamical_kt"],
    ),
    "soft_drop_z_cut_02": unfolding_analysis.UncertaintySmoothingConfiguration(
        # Reasoning:
        # - Tracking eff (2-3 bin drops significantly)
        # - Model dependence (unphysical jumps)
        # - Non-closure (2-3 bin drops significantly)
        # We skip the background subtraction because we don't have the pathalogical bin that we had in central.
        contributors={"tracking_efficiency": 1, "model_dependence": 1, "non_closure": 1},
        kt_range_to_smooth=PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central")["soft_drop_z_cut_02"],
    ),
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

# Update n_iter for model dependence
for grooming_method, model_dependence_n_iter_values in _model_dependence_update_n_iter.items():
    for model_dependence_label, n_iter in model_dependence_n_iter_values.items():
        if n_iter is not None:
            logger.info(f"Changing n_iter for {grooming_method}, model_dependnece_{model_dependence_label}: {semi_central_R02_unfolding_systematics_outputs[grooming_method][f'model_dependence_{model_dependence_label}'].n_iter_compare}->{n_iter}")
            semi_central_R02_unfolding_systematics_outputs[grooming_method][f"model_dependence_{model_dependence_label}"].n_iter_compare = n_iter

# Focus down onto just the unfolded distributions
semi_central_R02_unfolded_with_systematics, semi_central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=semi_central_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=semi_central_R02_unfolding_closure_outputs,
    true_jet_pt_range=true_jet_pt_range,
    calculate_quadrature_assuming_all_are_symmetric=calculate_quadrature_assuming_all_are_symmetric,
    unfolding_related_systematic_treatment=_unfolding_related_systematic_treatment,
    smooth_systematic_uncertainty_contributions=_uncertainty_smoothing_configuration,
    model_dependence_configuration=_model_dependence_configuration,
    non_closure_configuration=_non_closure_configuration,
    background_subtraction_configuration=_background_subtraction_configuration,
)

# %%
print(list(semi_central_R02_unfolding_systematics_outputs["dynamical_kt"].keys()))
print(list(semi_central_R02_unfolding_closure_outputs["dynamical_kt"].keys()))

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
    unfolding_related_systematic_treatment=_unfolding_related_systematic_treatment,
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
#_tag_after_suffix["soft_drop_z_cut_04"] = f"{_tag_after_suffix_base}_merge_3_6"
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
    "dynamical_core": helpers.KtRange(1.5, 1.5),
    "dynamical_kt": helpers.KtRange(1.5, 1.5),
    "dynamical_time": helpers.KtRange(1.5, 1.5),
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
    # Need larger range to select a higher iter for the model dependence
    "soft_drop_z_cut_02": 30,
    "dynamical_kt": 30,
}
_max_n_iter.update({
    grooming_method: 20 for grooming_method in grooming_methods if grooming_method not in ["dynamical_kt", "soft_drop_z_cut_02","soft_drop_z_cut_04"]
})

# Double counting cut
_double_counting_cut = {
    _method: "min_true_10_pt_hat_3"
    for _method in grooming_methods
}
# Model dependence.
#_model_dependence_configuration = None
_model_dependence_configuration = {
    _method: unfolding_analysis.ModelDependenceConfiguration(
        # Use the default n_iter for all grooming methods, but optimize below as necessary.
        nominal="embed_pythia_fastsim",
        variations=["embed_jewel_no_recoils_fastsim"],
        approach_to_combining="max",
        skip_double_counting_label=False,
    ) for _method in _grooming_methods_using_new_conventions
}
_model_dependence_update_n_iter = {
    # Selected based on usual convergence criteria...
    "dynamical_kt": {
        "embed_pythia_fastsim": 17,
        # The usual n_iter seems to be reasonable for JEWEL
        "embed_jewel_no_recoils_fastsim": None,
    },
    "soft_drop_z_cut_02": {
        "embed_pythia_fastsim": 17,
        "embed_jewel_no_recoils_fastsim": 17,
    }
}
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
        # NOTE: I exclude the reweight_response because I think it's overlapping with the model dependence
        #contributors=["reweight_pseudo_data", "reweight_response", "thermal_model"],
        contributors=["reweight_pseudo_data", "thermal_model"],
        approach_to_combining="max",
    )
    for grooming_method in grooming_methods
}

# Smoothing for systematic uncertainty contributions
_uncertainty_smoothing_configuration = {
    grooming_method: unfolding_analysis.UncertaintySmoothingConfiguration(
        # By default, no smoothing
        contributors={},
        kt_range_to_smooth=PbPb_kt_measured_range_by_grooming_method(event_activity="central")[grooming_method],
    )
    for grooming_method in _grooming_methods_using_new_conventions
}
_uncertainty_smoothing_configuration.update({
    "dynamical_kt": unfolding_analysis.UncertaintySmoothingConfiguration(
        # Reasoning:
        # Non-closure. It is bouncing up and down a bit arbitrarily
        contributors={"non_closure": 1},
        kt_range_to_smooth=PbPb_kt_measured_range_by_grooming_method(event_activity="central")["dynamical_kt"],
    ),
    "soft_drop_z_cut_02": unfolding_analysis.UncertaintySmoothingConfiguration(
        # Reasoning:
        # - Tracking eff (3-4 bin drops significantly)
        # - Model dependence (unphysical jumps)
        # - Background sub (mid kt, 3-4 bin drops significantly)
        # - Non-closure (low kt, 3-4 bin drops significantly)
        # - Note: tracking eff is not smoothed because it causes it to be artificially flat (e.g. it is effectively
        #   a flat 2%, which appears implausible based on DyG + semi-central SD). So leave it.
        #   It has minimal impact on the sum, but seems a more plausible estimate.
        # So, this suggests smoothing for all uncertainties (unfolding is so small that doesn't matter)
        contributors={"model_dependence": 1, "background_sub": 1, "non_closure": 1},
        kt_range_to_smooth=PbPb_kt_measured_range_by_grooming_method(event_activity="central")["soft_drop_z_cut_02"],
    ),
})

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

# Update n_iter for model dependence
for grooming_method, model_dependence_n_iter_values in _model_dependence_update_n_iter.items():
    for model_dependence_label, n_iter in model_dependence_n_iter_values.items():
        if n_iter is not None:
            logger.info(f"Changing n_iter for {grooming_method}, model_dependnece_{model_dependence_label}: {central_R02_unfolding_systematics_outputs[grooming_method][f'model_dependence_{model_dependence_label}'].n_iter_compare}->{n_iter}")
            central_R02_unfolding_systematics_outputs[grooming_method][f"model_dependence_{model_dependence_label}"].n_iter_compare = n_iter

# Focus down onto just the unfolded distributions
central_R02_unfolded_with_systematics, central_R02_true_reference = plot_unfolding.unfolded_outputs_with_systematics(
    grooming_methods=grooming_methods,
    unfolding_systematics_outputs=central_R02_unfolding_systematics_outputs,
    unfolding_closure_outputs=central_R02_unfolding_closure_outputs,
    true_jet_pt_range=true_jet_pt_range,
    calculate_quadrature_assuming_all_are_symmetric=calculate_quadrature_assuming_all_are_symmetric,
    unfolding_related_systematic_treatment=_unfolding_related_systematic_treatment,
    smooth_systematic_uncertainty_contributions=_uncertainty_smoothing_configuration,
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
    plot_systematics=False,
    plot_closures=False,
    unfolding_related_systematic_treatment=_unfolding_related_systematic_treatment,
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
print(list(central_R02_unfolding_closure_outputs["dynamical_kt"].keys()))
print(list(central_R02_unfolding_systematics_outputs["dynamical_kt"].keys()))

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
input_dir_tag = "2023-paper"
###################
# Setup I/O options
###################
# NOTE: Technically, these are HP2020 results rather than QM2022, but good enough
_use_qm22_inputs = False
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
    "dynamical_core": 7,
    "dynamical_kt": 7,
    "dynamical_time": 7,
    "soft_drop_z_cut_02": 9,
    "dynamical_core_z_cut_02": 9,
    "dynamical_kt_z_cut_02": 9,
    "dynamical_time_z_cut_02": 9,
    "soft_drop_z_cut_04": 10,
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
    unfolding_related_systematic_treatment=_unfolding_related_systematic_treatment,
    model_dependence_configuration=_model_dependence_configuration,
    non_closure_configuration=non_closure_configuration,
    background_subtraction_configuration=None,
)

# %%
print(pp_R04_unfolding_systematics_outputs["dynamical_core"].keys())
print(pp_R04_unfolded_with_systematics["dynamical_core"].data.metadata["y_systematic"].keys())

# %%
plot_unfolding.steer_plotting_of_kt_unfolding_outputs(
    grooming_methods=grooming_methods,
    unfolded_with_systematics=pp_R04_unfolded_with_systematics,
    unfolding_systematics_outputs=pp_R04_unfolding_systematics_outputs,
    unfolding_closure_outputs=pp_R04_unfolding_closure_outputs,
    plot=True,
    plot_png=False,
    plot_systematic_breakdown=True,
    plot_systematics=False,
    plot_closures=False,
    # NOTE: All of the commentary below is **only** related to plotting of the prior for selecting the
    #       number of iterations. The actual prior systematic is evaluated properly!
    # NOTE: For the prior variation, passing the HERWIG model dependence includes both:
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
pythia_predictions_R04 = model_calculations.ModelCalculation(
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
    base_dir=Path("output/comparison/models/jetscape/2023-10-yasuki"),
    needs_normalization=False,
    # NOTE: This range is techincally incorrect, but the jet fiducial acceptance corrects for it,
    #       so in the end, it doesn't matter.
    hadron_rapidity_range=2.0,
    metadata={"jet_R": 0.2},
)
jetscape_predictions_R02 = jetscape_R02.load_predictions()

# %% [markdown]
# #### R = 0.4
#
# Predcitions from Yasuki provided after HP 2023 for R = 0.4, pp only

# %%
jetscape_R04 = model_calculations.Jetscape(
    base_dir=Path("output/comparison/models/jetscape/2023-10-yasuki"),
    needs_normalization=False,
    # NOTE: This range is techincally incorrect, but the jet fiducial acceptance corrects for it,
    #       so in the end, it doesn't matter.
    hadron_rapidity_range=2.0,
    metadata={"jet_R": 0.4, "selected_collision_systems": ["pp"]},
)
jetscape_predictions_R04 = jetscape_R04.load_predictions()

# %% [markdown]
# ### JEWEL

# %% [markdown]
# #### R = 0.2

# %%
jewel_recoils_R02 = model_calculations.JEWEL(
    base_dir=Path("output/comparison/models/jewel"),
    needs_normalization=True,
    metadata={
        "recoils": True,
        "jet_R": 0.2,
    },
)
jewel_recoils_predictions_R02 = jewel_recoils_R02.load_predictions()
jewel_no_recoils_R02 = model_calculations.JEWEL(
    base_dir=Path("output/comparison/models/jewel"),
    needs_normalization=True,
    metadata={
        "recoils": False,
        "jet_R": 0.2,
    },
)
jewel_no_recoils_predictions_R02 = jewel_no_recoils_R02.load_predictions()

# %%
jewel_no_recoils_predictions_R02.spectra("central")["dynamical_core"].axes[0].bin_edges

# %% [markdown]
# ### Sherpa
#

# %% [markdown]
# #### R = 0.2, TBD

# %%
sherpa_ahadic_R02 = model_calculations.SherpaFromLeticia(
    base_dir=Path(output_dir / "comparison" / "models" / "sherpa"),
    needs_normalization=False,
    metadata={
        "jet_R": 0.2,
        "hadronization_method": "ahadic"
    },
)
sherpa_ahadic_predictions_R02 = sherpa_ahadic_R02.load_predictions()
sherpa_lund_R02 = model_calculations.SherpaFromLeticia(
    base_dir=Path(output_dir / "comparison" / "models" / "sherpa"),
    needs_normalization=False,
    metadata={
        "jet_R": 0.2,
        "hadronization_method": "lund"
    },
)
sherpa_lund_predictions_R02 = sherpa_lund_R02.load_predictions()

# %% [markdown]
# #### R = 0.4
#
# R = 0.4 predictions performed by Leticia

# %%
sherpa_ahadic_R04 = model_calculations.SherpaFromLeticia(
    base_dir=Path(output_dir / "comparison" / "models" / "sherpa"),
    needs_normalization=True,
    metadata={
        "jet_R": 0.4,
        "hadronization_method": "ahadic",
        "pre_bin_width_normalization": False,
        "bin_width_normalization": False,
    },
)
sherpa_ahadic_predictions_R04 = sherpa_ahadic_R04.load_predictions()
sherpa_lund_R04 = model_calculations.SherpaFromLeticia(
    base_dir=Path(output_dir / "comparison" / "models" / "sherpa"),
    needs_normalization=True,
    metadata={
        "jet_R": 0.4,
        "hadronization_method": "lund",
        "pre_bin_width_normalization": False,
        "bin_width_normalization": False,
    },
)
sherpa_lund_predictions_R04 = sherpa_lund_R04.load_predictions()

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
caucal_analytical_pp_R04 = model_calculations.Caucal2020AnalyticalCalculations(
    base_dir=output_dir / "comparison" / "models" / "caucal_analytical",
    needs_normalization=False,
    metadata={
        # NOTE: Equivalent to [0.5, 1, 2, 4, 6, 8]
        #"bin_edges": pp_R04_unfolded_with_systematics["dynamical_kt"].data.axes[0].bin_edges[2:-1]
        "bin_edges": [0.5, 1, 2, 4, 6, 8]
    },
)
caucal_analytical_pp_predictions_R04 = caucal_analytical_pp_R04.load_predictions()

# %%
caucal_analytical_pp_predictions_R04.pp["dynamical_kt"]

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
plot_output_dir_tag = "2024-paper-plots"
grooming_methods_for_letter = ["dynamical_kt", "soft_drop_z_cut_02"]

def PbPb_kt_measured_range_by_grooming_method(event_activity: str) -> dict[str, helpers.KtRange]:
    return {
        "dynamical_core": helpers.KtRange(2, 6) if event_activity == "semi_central" else helpers.KtRange(3, 6),
        "dynamical_kt": helpers.KtRange(2, 6) if event_activity == "semi_central" else helpers.KtRange(3, 6),
        "dynamical_time": helpers.KtRange(2, 6) if event_activity == "semi_central" else helpers.KtRange(3, 6),
        "soft_drop_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_core_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_kt_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_time_z_cut_02": helpers.KtRange(0.25, 6),
        "soft_drop_z_cut_04": helpers.KtRange(0.25, 6),
    }


# %% [markdown]
# #### Temp testing code

# %%
# %matplotlib widget

# %%
import seaborn as sns

sns.color_palette(
  [
    # Dark Green
    '#003d31', '#0b5345', '#1d7373',
    # Olive
    '#4d2600', '#6e2c00', '#8f3f00',
    # Rust
    '#44110d', '#b03a2e', '#f27573',
    # Peach
    '#b2a464', '#f7dc6f', '#fff9d9',
    # Navy Blue
    '#050b41', '#0a1172', '#1d2c9b',
    # Sky Blue
    '#5390d9', '#6fb1fc', '#9cc9ff',
    # Salmon
    '#b83621', '#ff5733', '#ff8752',
    # Mustard
    '#b89400', '#f0c72f', '#fbe7a4'
])
#sns.color_palette("rocket")

# %%

_colors = {
    "magenta": {
        # 35% darker, --, 45% darker
        "dark": "#793051",
        "mid": "#b84c7d",
        #"light": "#d89db7",
        "light": "#ff6361",
    },
    "violet": {
        # Previously generated
        "dark": "#7e459e",
        "mid": "#cda9e0",
        "light": "",
    },
    "blue": {
        # Previously generated
        #"dark": "#7385d9",
        # 40% darker than mid
        "dark": "#2b3f9d",
        "mid": "#4bafd0",
        "light": "#abb6e8",
    },
    "blue2": {
        # Previously generated
        #"dark": "#7385d9",
        # 40% darker than mid
        "dark": "#2980b9",
        "mid": "#4bafd0",
        "light": "#8bc1e5",
    },
    "green": {
        # Previously generated
        "dark": "#147736",
        #"mid": "#2ecc71",
        "mid": "#85aa55",
        "light": "#2ecc71",
    },
    "orange": {
        "dark": "#d35400",
        "mid": "#FF8301",
        # 40% lighter
        "light": "#ffb567",
    }
}

method_to_color = dict(zip(
    [
        "dynamical_core",
        "dynamical_core_z_cut_02",
        "dynamical_kt",
        "dynamical_kt_z_cut_02",
        "dynamical_time",
        "dynamical_time_z_cut_02",
        "soft_drop_z_cut_02",
        "soft_drop_z_cut_04",
    ],
    [
        # https://colorkit.co/palette/7e459e-c09cd3-7385d9-4bafd0-517225-85aa55-b84c7d-FF8301/
        #"#7e459e","#c09cd3","#7385d9","#4bafd0","#517225","#85aa55","#b84c7d","#FF8301"
        # 5 looks good
        #"#7e459e","#cda9e0","#7385d9","#4bafd0","#367325","#7fad93","#b84c7d","#FF8301"
        # 5 looks even better here...
        #"#7e459e","#cda9e0","#7385d9","#4bafd0","#147736","#7fad93","#b84c7d","#FF8301"
        #_colors["magenta"]["mid"], _colors["violet"]["mid"],
        _colors["magenta"]["mid"], _colors["magenta"]["light"],
        #*list(_colors["magenta"].values())[:2],
        _colors["blue2"]["dark"], _colors["blue2"]["light"],
        #_colors["blue2"]["dark"], _colors["blue"]["mid"],
        #*list(_colors["blue"].values())[:2],
        _colors["green"]["dark"], _colors["green"]["light"],
        #*list(_colors["green"].values())[:2],
        *list(_colors["orange"].values())[1:],
        #_colors["magenta"]["mid"],
        #_colors["orange"]["mid"],
        #"#7e459e","#cda9e0","#7385d9","#4bafd0","#147736","#27ae60","#b84c7d","#FF8301"
    ]
    # Nice teal: 008585
    , strict=True
))

method_to_color = dict(zip(
    [
        "dynamical_core",
        "dynamical_core_z_cut_02",
        "dynamical_kt",
        "dynamical_kt_z_cut_02",
        "dynamical_time",
        "dynamical_time_z_cut_02",
        "soft_drop_z_cut_02",
        "soft_drop_z_cut_04",
    ],
    [
        # Dark and light
        # Magenta
        "#b84c7d", "#ff6361",
        # Blue
        "#2980b9", "#8bc1e5",
        # Greens
        "#147736", "#2ecc71",
        # Orange
        "#FF8301", "#ffb567",
    ]
    # Nice teal: 008585
    , strict=True
))


# %% [markdown]
# ## pp data: compare between grooming methods

# %% [markdown]
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for _temp_grooming_methods, _reference_method, _label, _ratio_y_range in [
    (["soft_drop_z_cut_02", "dynamical_core", "dynamical_kt", "dynamical_time"], "soft_drop_z_cut_02", "primary", (0.35, 1.65)),
    (["soft_drop_z_cut_04", "dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "soft_drop_z_cut_02", "secondary", (0.2, 1.8)),
    (["soft_drop_z_cut_02", "soft_drop_z_cut_04", "dynamical_kt", "dynamical_kt_z_cut_02"], "soft_drop_z_cut_02", "summary", (0.2, 1.8)),
]:
    plot_paper.plot_comparisons_of_grooming_methods_for_single_system(
        hists=pp_R02_unfolded_with_systematics,
        grooming_methods=_temp_grooming_methods,
        reference_grooming_method=_reference_method,
        collision_system="pp",
        collision_system_key="pp_5TeV",
        output_dir=_output_dir,
        kt_range=helpers.KtRange(0.25, 6),
        figure_kt_range=helpers.KtRange(0, 6.25),
        ratio_y_range=_ratio_y_range,
        jet_R_str=jet_R_str,
        alice_status=alice_status,
        label=_label,
    )

# %% [markdown]
# ### R = 0.4

# %%
jet_R = 0.4
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for _temp_grooming_methods, _reference_method, _label, _ratio_y_range in [
    (["soft_drop_z_cut_02", "dynamical_core", "dynamical_kt", "dynamical_time"], "soft_drop_z_cut_02", "primary", (0.2, 1.8)),
    (["soft_drop_z_cut_04", "dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "soft_drop_z_cut_02", "secondary", (0.1, 1.9)),
    (["soft_drop_z_cut_02", "soft_drop_z_cut_04", "dynamical_kt", "dynamical_kt_z_cut_02"], "soft_drop_z_cut_02", "summary", (0.1, 1.9)),
]:
    plot_paper.plot_comparisons_of_grooming_methods_for_single_system(
        hists=pp_R04_unfolded_with_systematics,
        grooming_methods=_temp_grooming_methods,
        reference_grooming_method=_reference_method,
        collision_system="pp",
        collision_system_key="pp_5TeV",
        output_dir=_output_dir,
        kt_range=helpers.KtRange(0.25, 8),
        figure_kt_range=helpers.KtRange(0, 8.25),
        ratio_y_range=_ratio_y_range,
        jet_R_str=jet_R_str,
        alice_status=alice_status,
        label=_label,
    )

# %% [markdown]
# ## pp data comparison to models

# %% [markdown]
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for _grooming_method in grooming_methods:
    plot_paper.plot_grooming_methods_comparison_with_model_for_single_system(
        hists=pp_R02_unfolded_with_systematics,
        models={
            "jetscape": jetscape_predictions_R02,
            "pythia": pythia_predictions_R02,
            # All of the hybrid loaded predictions have the same pp, so picking any one is fine!
            "hybrid": hybrid_model_with_wake_with_moliere_predictions_R02,
            # All of the JEWEL loaded predictions have the same pp, so picking any one is fine!
            "jewel": jewel_no_recoils_predictions_R02,
        },
        grooming_methods=[_grooming_method],
        collision_system="pp",
        collision_system_key="pp_5TeV",
        output_dir=_output_dir,
        kt_range=helpers.KtRange(0.25, 6),
        figure_kt_range=helpers.KtRange(0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )

# Figure for the paper, saving the space by using one figure and fully shared axes
plot_paper.plot_grooming_methods_comparison_with_model_for_single_system_one_figure(
    hists=pp_R02_unfolded_with_systematics,
    models={
        "jetscape": jetscape_predictions_R02,
        "pythia": pythia_predictions_R02,
        # All of the hybrid loaded predictions have the same pp, so picking any one is fine!
        "hybrid": hybrid_model_with_wake_with_moliere_predictions_R02,
    },
    grooming_methods=grooming_methods,
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 6),
    figure_kt_range=helpers.KtRange(0, 6.25),
    main_panel_y_axis_range=(8e-3, 0.8),
    ratio_y_axis_range=(0.3, 1.7),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
)

# Some other plots that we keep around, but are unlikely to go into the paper
for _method_groups in [
    ["dynamical_core", "dynamical_core_z_cut_02"],
    ["dynamical_kt", "dynamical_kt_z_cut_02"],
    ["dynamical_time", "dynamical_time_z_cut_02"],
    ["soft_drop_z_cut_02", "soft_drop_z_cut_04"],
    ["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"],
    list(reversed(["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"])),
    ["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02", "soft_drop_z_cut_04"],
    list(reversed(["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02", "soft_drop_z_cut_04"])),
    ["dynamical_core_z_cut_02", "soft_drop_z_cut_02", "soft_drop_z_cut_04"],
]:
    # I don't think I will use these in the paper. However, I might, so may as well keep them around
    continue
    plot_paper.plot_grooming_methods_comparison_with_model_for_single_system(
        hists=pp_R02_unfolded_with_systematics,
        models={
            "jetscape": jetscape_predictions_R02,
            "pythia": pythia_predictions_R02,
            # All of the hybrid loaded predictions have the same pp, so picking any one is fine!
            "hybrid": hybrid_model_with_wake_with_moliere_predictions_R02,
        },
        grooming_methods=_method_groups,
        collision_system="pp",
        collision_system_key="pp_5TeV",
        output_dir=_output_dir,
        kt_range=helpers.KtRange(0.25, 6),
        figure_kt_range=helpers.KtRange(0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )
# This plot with all methods isn't really super useful as it's too much information on one panel, but I think
# it's a bit nice as a summary of all of the available data and models
plot_paper.plot_grooming_methods_comparison_with_model_for_single_system(
    hists=pp_R02_unfolded_with_systematics,
    models={
        "jetscape": jetscape_predictions_R02,
        "pythia": pythia_predictions_R02,
        # All of the hybrid loaded predictions have the same pp, so picking any one is fine!
        "hybrid": hybrid_model_with_wake_with_moliere_predictions_R02,
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


# %% [markdown]
# ### R = 0.4

# %%
jet_R = 0.4
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for _grooming_method in grooming_methods:
    plot_paper.plot_grooming_methods_comparison_with_model_for_single_system(
        hists=pp_R04_unfolded_with_systematics,
        models={
            "jetscape": jetscape_predictions_R04,
            "pythia": pythia_predictions_R04,
            # All of the hybrid loaded predictions have the same pp, so any are fine!
            #"hybrid": (hybrid_model_with_wake_with_moliere_predictions_R04, hybrid_model_with_wake_with_moliere_predictions_R04.pp),
            "sherpa_ahadic": sherpa_ahadic_predictions_R04,
            "sherpa_lund": sherpa_lund_predictions_R04,
            "caucal_analytical": caucal_analytical_pp_predictions_R04,
        },
        grooming_methods=[_grooming_method],
        collision_system="pp",
        collision_system_key="pp_5TeV",
        output_dir=_output_dir,
        kt_range=helpers.KtRange(0.25, 8),
        figure_kt_range=helpers.KtRange(0, 8.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )

# Figure for the paper, saving the space by using one figure and fully shared axes
plot_paper.plot_grooming_methods_comparison_with_model_for_single_system_one_figure(
    hists=pp_R04_unfolded_with_systematics,
    models={
        "jetscape": jetscape_predictions_R04,
        "pythia": pythia_predictions_R04,
        # All of the hybrid loaded predictions have the same pp, so any are fine!
        #"hybrid": (hybrid_model_with_wake_with_moliere_predictions_R04, hybrid_model_with_wake_with_moliere_predictions_R04.pp),
        "sherpa_ahadic": sherpa_ahadic_predictions_R04,
        "sherpa_lund": sherpa_lund_predictions_R04,
        "caucal_analytical": caucal_analytical_pp_predictions_R04,
    },
    grooming_methods=grooming_methods,
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 8),
    figure_kt_range=helpers.KtRange(0, 8.25),
    main_panel_y_axis_range=(4e-3, 0.8),
    ratio_y_axis_range=(0.3, 1.7),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
)

# Some other plots that we keep around, but are unlikely to go into the paper
for _method_groups in [
    ["dynamical_core", "dynamical_core_z_cut_02"],
    ["dynamical_kt", "dynamical_kt_z_cut_02"],
    ["dynamical_time", "dynamical_time_z_cut_02"],
    ["soft_drop_z_cut_02", "soft_drop_z_cut_04"],
    ["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"],
    list(reversed(["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"])),
    ["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02", "soft_drop_z_cut_04"],
    list(reversed(["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02", "soft_drop_z_cut_04"])),
    ["dynamical_core_z_cut_02", "soft_drop_z_cut_02", "soft_drop_z_cut_04"],
]:
    # I don't think I will use these in the paper. However, I might, so may as well keep them around
    continue
    plot_paper.plot_grooming_methods_comparison_with_model_for_single_system(
        hists=pp_R04_unfolded_with_systematics,
        models={
            "jetscape": jetscape_predictions_R04,
            "pythia": pythia_predictions_R04,
            # All of the hybrid loaded predictions have the same pp, so any are fine!
            #"hybrid": (hybrid_model_with_wake_with_moliere_predictions_R04, hybrid_model_with_wake_with_moliere_predictions_R04.pp),
            "sherpa_ahadic": sherpa_ahadic_predictions_R04,
            "sherpa_lund": sherpa_lund_predictions_R04,
            "caucal_analytical": caucal_analytical_pp_predictions_R04,
        },
        grooming_methods=_method_groups,
        collision_system="pp",
        collision_system_key="pp_5TeV",
        output_dir=_output_dir,
        kt_range=helpers.KtRange(0.25, 8),
        figure_kt_range=helpers.KtRange(0, 8.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )
# This plot with all methods isn't really super useful as it's too much information on one panel, but I think
# it's a bit nice as a summary of all of the available data and models
plot_paper.plot_grooming_methods_comparison_with_model_for_single_system(
    hists=pp_R04_unfolded_with_systematics,
    models={
        "jetscape": jetscape_predictions_R04,
        "pythia": pythia_predictions_R04,
        # All of the hybrid loaded predictions have the same pp, so any are fine!
        #"hybrid": (hybrid_model_with_wake_with_moliere_predictions_R04, hybrid_model_with_wake_with_moliere_predictions_R04.pp),
        "sherpa_ahadic": sherpa_ahadic_predictions_R04,
        "sherpa_lund": sherpa_lund_predictions_R04,
        "caucal_analytical": caucal_analytical_pp_predictions_R04,
    },
    grooming_methods=grooming_methods,
    collision_system="pp",
    collision_system_key="pp_5TeV",
    output_dir=_output_dir,
    kt_range=helpers.KtRange(0.25, 8),
    figure_kt_range=helpers.KtRange(0, 8.25),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
)

# %% [markdown]
# ## PbPb data: compare between grooming methods

# %% [markdown]
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

for _collision_system, _hists in [
    ("semi_central", semi_central_R02_unfolded_with_systematics),
    ("central", central_R02_unfolded_with_systematics),
]:
    logger.info(f"Plotting {_collision_system}")
    for _temp_grooming_methods, _reference_method, _label, _ratio_y_range in [
        (["soft_drop_z_cut_02", "dynamical_core", "dynamical_kt", "dynamical_time"], "soft_drop_z_cut_02", "primary", (0.45, 1.55)),
        (["soft_drop_z_cut_04", "dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02"], "soft_drop_z_cut_02", "secondary", (0.2, 1.8)),
        (["soft_drop_z_cut_02", "soft_drop_z_cut_04", "dynamical_kt", "dynamical_kt_z_cut_02"], "soft_drop_z_cut_02", "summary", (0.2, 1.8)),
    ]:
        plot_paper.plot_comparisons_of_grooming_methods_for_single_system(
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
# ## PbPb data comparison to models

# %% [markdown]
# ### R = 0.2, semi-central + central

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

models = {
    "hybrid_without_moliere": hybrid_model_with_wake_without_moliere_predictions_R02,
    "hybrid_moliere": hybrid_model_with_wake_with_moliere_predictions_R02,
    "jetscape": jetscape_predictions_R02,
    "jewel_recoils": jewel_recoils_predictions_R02,
    "jewel_no_recoils": jewel_no_recoils_predictions_R02,
}
for _collision_system, _hists in [
    ("semi_central", semi_central_R02_unfolded_with_systematics),
    ("central", central_R02_unfolded_with_systematics),
]:
    _PbPb_kt_range = {
        "dynamical_core": helpers.KtRange(2, 6) if _collision_system == "semi_central" else helpers.KtRange(3, 6),
        "dynamical_kt": helpers.KtRange(2, 6) if _collision_system == "semi_central" else helpers.KtRange(3, 6),
        "dynamical_time": helpers.KtRange(2, 6) if _collision_system == "semi_central" else helpers.KtRange(3, 6),
        "soft_drop_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_core_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_kt_z_cut_02": helpers.KtRange(0.25, 6),
        "dynamical_time_z_cut_02": helpers.KtRange(0.25, 6),
        "soft_drop_z_cut_04": helpers.KtRange(0.25, 6),
    }

    for _grooming_method in grooming_methods:
        plot_paper.plot_grooming_methods_comparison_with_model_for_single_system(
            hists=_hists,
            models=models,
            grooming_methods=[_grooming_method],
            collision_system=_collision_system,
            collision_system_key="PbPb_5TeV",
            output_dir=_output_dir,
            kt_range=_PbPb_kt_range,
            figure_kt_range=helpers.KtRange(0, 6.25),
            jet_R_str=jet_R_str,
            alice_status=alice_status,
        )

    # Figure for the paper, saving the space by using one figure and fully shared axes
    plot_paper.plot_grooming_methods_comparison_with_model_for_single_system_one_figure(
        hists=_hists,
        models=models,
        grooming_methods=grooming_methods,
        collision_system=_collision_system,
        collision_system_key="PbPb_5TeV",
        output_dir=_output_dir,
        kt_range=_PbPb_kt_range,
        figure_kt_range=helpers.KtRange(0, 6.25),
        main_panel_y_axis_range=(8e-3, 0.8),
        ratio_y_axis_range=(0.3, 1.7),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )

    # Some other plots that we keep around, but are unlikely to go into the paper
    for _method_groups in [
        ["dynamical_core", "dynamical_core_z_cut_02"],
        ["dynamical_kt", "dynamical_kt_z_cut_02"],
        ["dynamical_time", "dynamical_time_z_cut_02"],
        ["soft_drop_z_cut_02", "soft_drop_z_cut_04"],
        ["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"],
        list(reversed(["dynamical_core", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02"])),
        ["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02", "soft_drop_z_cut_04"],
        list(reversed(["dynamical_core_z_cut_02", "dynamical_kt_z_cut_02", "dynamical_time_z_cut_02", "soft_drop_z_cut_04"])),
        ["dynamical_core_z_cut_02", "soft_drop_z_cut_02", "soft_drop_z_cut_04"],
    ]:
        # I don't think I will use these in the paper. However, I might, so may as well keep them around
        continue
        plot_paper.plot_grooming_methods_comparison_with_model_for_single_system(
            hists=_hists,
            models=models,
            grooming_methods=_method_groups,
            collision_system=_collision_system,
            collision_system_key="PbPb_5TeV",
            output_dir=_output_dir,
            kt_range=_PbPb_kt_range,
            figure_kt_range=helpers.KtRange(0, 6.25),
            jet_R_str=jet_R_str,
            alice_status=alice_status,
        )
    # This plot with all methods isn't really super useful as it's too much information on one panel, but I think
    # it's a bit nice as a summary of all of the available data and models
    plot_paper.plot_grooming_methods_comparison_with_model_for_single_system(
        hists=_hists,
        models=models,
        grooming_methods=grooming_methods,
        collision_system=_collision_system,
        collision_system_key="PbPb_5TeV",
        output_dir=_output_dir,
        kt_range=_PbPb_kt_range,
        figure_kt_range=helpers.KtRange(0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )

# %% [markdown]
# ## PbPb-pp comparison by each grooming method
#
# ### R = 0.2, semi-central + central

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_paper.plot_pp_PbPb_comparison_single_figure(
    hists={
        "pp": pp_R02_unfolded_with_systematics,
        "semi_central": semi_central_R02_unfolded_with_systematics,
        "central": central_R02_unfolded_with_systematics,
    },
    grooming_methods=grooming_methods,
    output_dir=_output_dir,
    event_activity_to_kt_range={
        "pp": helpers.KtRange(0.25, 6),
        "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
        "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
    },
    kt_display_range=(0.0, 6.25),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
)

for grooming_method in grooming_methods:
    logger.info(f"Processing {grooming_method}")
    plot_paper.plot_pp_PbPb_comparison(
        hists={
            "pp": pp_R02_unfolded_with_systematics,
            "semi_central": semi_central_R02_unfolded_with_systematics,
            "central": central_R02_unfolded_with_systematics,
        },
        grooming_methods=[grooming_method],
        output_dir=_output_dir,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": helpers.KtRange(0.25, 6) if "z_cut" in grooming_method else helpers.KtRange(2, 6),
            "central": helpers.KtRange(0.25, 6) if "z_cut" in grooming_method else helpers.KtRange(3, 6),
        },
        kt_display_range=(0.0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )

# %% [markdown]
# ## PbPb-pp spectra only
#
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

plot_paper.plot_spectra_only_for_letter(
    hists={
        "pp": pp_R02_unfolded_with_systematics,
        "semi_central": semi_central_R02_unfolded_with_systematics,
        "central": central_R02_unfolded_with_systematics,
    },
    grooming_methods=grooming_methods_for_letter,
    output_dir=_output_dir,
    event_activity_to_kt_range={
        "pp": helpers.KtRange(0.25, 6),
        "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
        "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
    },
    # 6.25 would be preferred, but I need space to avoid the 6 and the next 0 over from overlapping
    kt_display_range=(0.0, 6.35),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
    additional_label="_".join(grooming_methods_for_letter),
)

for grooming_method in grooming_methods:
    logger.info(f"Processing {grooming_method}")
    plot_paper.plot_spectra_only(
        hists={
            "pp": pp_R02_unfolded_with_systematics,
            "semi_central": semi_central_R02_unfolded_with_systematics,
            "central": central_R02_unfolded_with_systematics,
        },
        grooming_methods=[grooming_method],
        output_dir=_output_dir,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": helpers.KtRange(0.25, 6) if "z_cut" in grooming_method else helpers.KtRange(2, 6),
            "central": helpers.KtRange(0.25, 6) if "z_cut" in grooming_method else helpers.KtRange(3, 6),
        },
        kt_display_range=(0.0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
    )

# %% [markdown]
# ## PbPb+pp spectra ratios only for model + data

# %% [markdown]
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

models_calculation = {
    "hybrid_without_moliere": hybrid_model_with_wake_without_moliere_predictions_R02,
    "hybrid_moliere": hybrid_model_with_wake_with_moliere_predictions_R02,
    "jetscape": jetscape_predictions_R02,
    "jewel_recoils": jewel_recoils_predictions_R02,
    "jewel_no_recoils": jewel_no_recoils_predictions_R02,
}
# For the smoothed spectra
fit_parameters = {
    "pp": {
        "soft_drop_z_cut_02": {
            "tanh_transition_scale": 0.3,
            "x0": 1.25,
        },
        "dynamical_kt": {
            "tanh_transition_scale": 0.2,
            "x0": 1.25,
        },
    },
    "semi_central": {
        "soft_drop_z_cut_02": {
            "tanh_transition_scale": 0.1,
            "x0": 1.25,
        },
        "dynamical_kt": {
            "tanh_transition_scale": 0.1,
            "x0": 1.25,
        },
    },
    "central": {
        "soft_drop_z_cut_02": {
            "tanh_transition_scale": 0.1,
            "x0": 1.25,
        },
        "dynamical_kt": {
            "tanh_transition_scale": 0.1,
            "x0": 1.25,
        },
    },
}

# Single figure for Letter
plot_paper.plot_pp_PbPb_only_spectra_ratios_for_letter(
    hists={
        "pp": pp_R02_unfolded_with_systematics,
        "semi_central": semi_central_R02_unfolded_with_systematics,
        "central": central_R02_unfolded_with_systematics,
    },
    models_calculation=models_calculation,
    grooming_methods=grooming_methods_for_letter,
    output_dir=_output_dir,
    event_activity_to_kt_range={
        "pp": helpers.KtRange(0.25, 6),
        "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
        "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
    },
    # 6.25 would be preferred, but I need space to avoid the 6 and the next 0 over from overlapping
    kt_display_range=(0.0, 6.35),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
    additional_label="_".join(grooming_methods_for_letter),
    logy=False,
    fit_parameters=fit_parameters,
    fit_QA_plot=True,
    # NOTE: This seems to need to be smaller to fit the header. Still could be tuned...
    text_font_size=24,
)

#for grooming_method in grooming_methods:
for grooming_method in ["dynamical_kt", "soft_drop_z_cut_02"]:
    # TEMP
    continue
    # ENDTEMP
    plot_paper.plot_pp_PbPb_only_spectra_ratios(
        hists={
            "pp": pp_R02_unfolded_with_systematics,
            "semi_central": semi_central_R02_unfolded_with_systematics,
            "central": central_R02_unfolded_with_systematics,
        },
        models_calculation=models_calculation,
        grooming_methods=[grooming_method],
        output_dir=_output_dir,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
            "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
        },
        model_labels_on_axes=[[], ["hybrid_without_moliere", "hybrid_moliere"], ["jetscape"]],
        kt_display_range=(0.0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
        logy=False,
        fit_parameters=fit_parameters,
        fit_QA_plot=True,
    )


# %% [markdown]
# ## PbPb-pp comparison (one centrality) to models
#
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

models_calculation = {
    "hybrid_without_moliere": hybrid_model_with_wake_without_moliere_predictions_R02,
    "hybrid_moliere": hybrid_model_with_wake_with_moliere_predictions_R02,
    "jetscape": jetscape_predictions_R02,
    "jewel_recoils": jewel_recoils_predictions_R02,
    "jewel_no_recoils": jewel_no_recoils_predictions_R02,
}

for event_activity in ["semi_central", "central"]:
    _additional_hists = {
        "semi_central": semi_central_R02_unfolded_with_systematics,
        "central": central_R02_unfolded_with_systematics,
    }
    plot_paper.plot_pp_PbPb_comparison_single_figure(
        hists={
            "pp": pp_R02_unfolded_with_systematics,
            event_activity: _additional_hists[event_activity],
        },
        models_ratio=models_calculation,
        grooming_methods=grooming_methods,
        output_dir=_output_dir,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
            "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
        },
        kt_display_range=(0.0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
        additional_label=event_activity,
    )

    for grooming_method in grooming_methods:
        plot_paper.plot_pp_PbPb_comparison(
            hists={
                "pp": pp_R02_unfolded_with_systematics,
                event_activity: _additional_hists[event_activity],
            },
            models_ratio=models_calculation,
            grooming_methods=[grooming_method],
            output_dir=_output_dir,
            event_activity_to_kt_range={
                "pp": helpers.KtRange(0.25, 6),
                "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
                "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
            },
            kt_display_range=(0.0, 6.25),
            jet_R_str=jet_R_str,
            alice_status=alice_status,
            additional_label=event_activity
        )

# %% [markdown]
# ## PbPb-pp comparison: spectra + multiple ratios (one per centrality) by each grooming method with model
#
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

models_calculation = {
    "hybrid_without_moliere": hybrid_model_with_wake_without_moliere_predictions_R02,
    "hybrid_moliere": hybrid_model_with_wake_with_moliere_predictions_R02,
    "jetscape": jetscape_predictions_R02,
    "jewel_recoils": jewel_recoils_predictions_R02,
    "jewel_no_recoils": jewel_no_recoils_predictions_R02,
}

#plot_paper.plot_pp_PbPb_comparison_single_figure(
#    hists={
#        "pp": pp_R02_unfolded_with_systematics,
#        "semi_central": semi_central_R02_unfolded_with_systematics,
#        "central": central_R02_unfolded_with_systematics,
#    },
#    models_ratio=models_calculation,
#    grooming_methods=grooming_methods,
#    output_dir=_output_dir,
#    event_activity_to_kt_range={
#        "pp": helpers.KtRange(0.25, 6),
#        "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
#        "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
#    },
#    kt_display_range=(0.0, 6.25),
#    jet_R_str=jet_R_str,
#    alice_status=alice_status,
#)

for grooming_method in grooming_methods:
    # TEMP
    continue
    # ENDTEMP
    plot_paper.plot_pp_PbPb_comparison_with_multiple_model_ratios(
        hists={
            "pp": pp_R02_unfolded_with_systematics,
            "semi_central": semi_central_R02_unfolded_with_systematics,
            "central": central_R02_unfolded_with_systematics,
        },
        models_ratio=models_calculation,
        grooming_methods=[grooming_method],
        output_dir=_output_dir,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
            "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
        },
        kt_display_range=(0.0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
        additional_label="multiple_ratios",
    )

# %% [markdown]
# ## PbPb-pp comparison ratios only, including data + models
#
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

models_calculation = {
    "hybrid_without_moliere": hybrid_model_with_wake_without_moliere_predictions_R02,
    "hybrid_moliere": hybrid_model_with_wake_with_moliere_predictions_R02,
    "jetscape": jetscape_predictions_R02,
    "jewel_recoils": jewel_recoils_predictions_R02,
    "jewel_no_recoils": jewel_no_recoils_predictions_R02,
}

plot_paper.plot_pp_PbPb_comparison_only_ratios_for_letter(
    hists={
        "pp": pp_R02_unfolded_with_systematics,
        "semi_central": semi_central_R02_unfolded_with_systematics,
        "central": central_R02_unfolded_with_systematics,
    },
    models_calculation=models_calculation,
    grooming_methods=grooming_methods_for_letter,
    output_dir=_output_dir,
    event_activity_to_kt_range={
        "pp": helpers.KtRange(0.25, 6),
        "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
        "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
    },
    # 6.25 would be preferred, but I need space to avoid the 6 and the next 0 over from overlapping
    kt_display_range=(0.0, 6.35),
    jet_R_str=jet_R_str,
    alice_status=alice_status,
    calculate_n_sigma_stat_from_unity=True,
    additional_label="_".join(grooming_methods_for_letter),
    # NOTE: This seems to need to be smaller to fit the header. Still could be tuned...
    text_font_size=24,
)

for grooming_method in grooming_methods:
    # TEMP
    continue
    # ENDTEMP
    plot_paper.plot_pp_PbPb_comparison_with_multiple_model_ratios(
        hists={
            "pp": pp_R02_unfolded_with_systematics,
            "semi_central": semi_central_R02_unfolded_with_systematics,
            "central": central_R02_unfolded_with_systematics,
        },
        models_ratio=models_calculation,
        grooming_methods=[grooming_method],
        output_dir=_output_dir,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
            "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
        },
        kt_display_range=(0.0, 6.25),
        jet_R_str=jet_R_str,
        alice_status=alice_status,
        additional_label="multiple_ratios",
    )

# %% [markdown]
# ## Subleading subjet purity (supplement)
#
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
# This is effectively encoded in the nb_utils, but we leave it here for clarity (because that's where it's going anyway)
_input_dir = output_dir / "embed_pythia" / "RDF"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)


# %%
from jet_substructure.base import notebook_utils as nb_utils

# Load relevant data, which is the output from the ROOT data frames...
collision_system_to_production_number = {
    "semi_central": "0067",
    "central": "0071",
}
# Example: output/embed_pythia/RDF/LHC20g4_embedded_into_LHC18qr_central_R02_0071_dynamical_kt_prefixes_hybrid_true_det_level_response.root
base_name = "LHC20g4_embedded_into_LHC18qr_{collision_system}_{jet_R_str}_{production_number}_{grooming_method}_prefixes_hybrid_true_det_level_response.root"

# Load the data
rdf_hists = {}
_successfully_loaded_grooming_methods = []
for collision_system, production_number in collision_system_to_production_number.items():
    rdf_hists[collision_system] = {}
    for grooming_method in grooming_methods:
        try:
            rdf_hists[collision_system][grooming_method] = nb_utils.load_histograms(
                filename=base_name.format(
                    collision_system=collision_system,
                    jet_R_str=jet_R_str,
                    production_number=production_number,
                    grooming_method=grooming_method,
                ),
                # This will always be embed pythia. We will determine which PbPb
                # collision system we want via the file that we select
                collision_system="embed_pythia",
                tag="RDF",
                base_path=output_dir,
            )
            _successfully_loaded_grooming_methods.append(grooming_method)
        except FileNotFoundError as e:
            logger.info(f"Skipping grooming method {grooming_method} because the output file isn't available ({e})")

    logger.info(f"{collision_system}: Successfully loaded grooming methods: {_successfully_loaded_grooming_methods}")

# %%
hybrid_min_kt_values = {
    "semi_central": [0, 1.0],
    "central": [0, 1.5],
}
plot_paper.plot_PbPb_subjet_purity_for_letter(
    collision_systems=list(hybrid_min_kt_values.keys()),
    hists=rdf_hists,
    grooming_methods=grooming_methods_for_letter,
    hybrid_min_kt_values=hybrid_min_kt_values,
    output_dir=_output_dir,
    subjet_for_purity="subleading",
    jet_R_str=jet_R_str,
    alice_status="simulation",
    #additional_label="_".join(grooming_methods_for_letter),
    # NOTE: This seems to need to be smaller to fit the header. Still could be tuned...
    #text_font_size=31,
    text_font_size=30,
)

# %%
plt.close("all")

# %% [markdown]
# # Latex tables for paper

# %%
import pandas as pd

# %%
central_R02_unfolded_with_systematics["dynamical_kt"].data.metadata["y_systematic"]["background_sub"]

# %%
#_current_grooming_method = "dynamical_kt"
#df_absolute = pd.DataFrame(
#    {
#        f"{k}_{direction}": getattr(v, direction)
#        for i, (k, v) in enumerate(central_R02_unfolded_with_systematics[_current_grooming_method].data.metadata["y_systematic"].items())
#        for direction in ["low", "high"]
#    }
#)
#df_absolute["stat"] = (
#    central_R02_unfolded_with_systematics[_current_grooming_method].data.errors
#)
#
#df_relative = df_absolute.divide(central_R02_unfolded_with_systematics[_current_grooming_method].data.values, axis=0)

import numpy as np

from jet_substructure.analysis import full_results_helpers


def define_paper_table_dfs(
    hists: dict[str, dict[str, unfolding_analysis.SingleResult]],
    collision_system: str,
    grooming_methods: list[str],
    event_activity_to_kt_range: dict[str, helpers.KtRange] | dict[str, dict[str, helpers.KtRange]],
) -> dict[str, tuple[list[str], pd.DataFrame]]:
    """ Going source-by-source of the uncertainties, we want to extract the minimum and maximum over the measured range.

    """
    # Validation
    for ev, kt_range in event_activity_to_kt_range.items():
        if isinstance(kt_range, helpers.KtRange):
            event_activity_to_kt_range[ev] = {grooming_method: kt_range for grooming_method in grooming_methods}

    # Setup
    collision_system_to_centrality = {
        "central": r"0--10\%",
        "semi_central": r"30--50\%"
    }

    tables = {}
    for grooming_method in grooming_methods:
        uncertainties = {}
        # Retrive hist and select range to consider based on collision system and grooming method.
        h = hists[collision_system][grooming_method].data
        h = full_results_helpers.select_hist_range(h, event_activity_to_kt_range[collision_system][grooming_method])

        # Create dataframes for absolute and relative uncertainties.
        # We want to keep the maximum uncertainties regardless of the low or high direction
        sources = list(h.metadata["y_systematic"].keys())
        # NOTE: The we need to determine the largest and smallest contributors using the relative uncertainty.
        #       Otherwise, we'll always be dominated by the low kt bins - i.e. those which have the highest yield
        #       and therefore the largest absolute error, regardless of whether those are the values of interest.
        # First, we'll symmetrize the uncertainties, since we don't want this to e.g. report zero for the minimum
        # because one side is 0.
        uncertainties = {
            k: v.high / h.values if np.allclose(v.low, v.high)
            else np.maximum(v.low / h.values, v.high / h.values)
            for k, v in h.metadata["y_systematic"].items()
        }
        # Add in the stat uncertainty just to keep track of it
        uncertainties["stat"] = h.errors / h.values
        # Now, calculate the min and max per source
        # NOTE: We wrap it into a list because there's only a single value per source.
        uncertainties = {
            f"{k}_{op}": getattr(np, op)(v)
            for k, v in uncertainties.items()
            for op in ["min", "max"]
        }
        # Add in centrality information if applicable
        if collision_system in collision_system_to_centrality:
            uncertainties["centrality"] = collision_system_to_centrality[collision_system]
        #logger.info(f"{uncertainties=}")

        df_relative = pd.DataFrame({grooming_method: uncertainties}).transpose()

        # NOTE: We can't convert back like this because we've already condensed down to a single minimum and maximum per source.
        #       But I also don't think we need the absolute uncertainties, so we just leave it aside.
        #df_absolute = df_relative.multiply(h.values, axis=0)

        tables[grooming_method] = (sources, df_relative)

    return tables



# %%
tables = {}
for collision_system in ["pp", "semi_central", "central"]:
    tables[collision_system] = define_paper_table_dfs(
        hists={
            "pp": pp_R02_unfolded_with_systematics,
            "semi_central": semi_central_R02_unfolded_with_systematics,
            "central": central_R02_unfolded_with_systematics,
        },
        collision_system=collision_system,
        grooming_methods=grooming_methods_for_letter,
        event_activity_to_kt_range={
            "pp": helpers.KtRange(0.25, 6),
            "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
            "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
        },
    )
tables

# %%
grooming_method_to_display_name = {
    "dynamical_kt": r"DyG. $a = 1.0$",
    "soft_drop_z_cut_02": r"SD $z_{\mathrm{cut}} = 0.2$",
}
uncertainties_display_tables = {}
# pp
uncertainties_display_tables["pp"] = pd.concat(
    [
        tables["pp"][grooming_method][1]
        for grooming_method in grooming_methods_for_letter
    ],
    #keys=grooming_methods_for_letter,
    #names=["grooming_method"],
).rename(index=grooming_method_to_display_name)
uncertainties_display_tables["pp"]
# PbPb
uncertainties_display_tables["PbPb"] = pd.concat(
    [
        tables[collision_system][grooming_method][1]
        for collision_system in ["central", "semi_central"]
        for grooming_method in grooming_methods_for_letter
    ],
    #keys=grooming_methods_for_letter,
    #names=["grooming_method"],
).rename(index=grooming_method_to_display_name)
uncertainties_display_tables["PbPb"]

# %%
# This operation was necessary in the previous iteration of this code, but this new code is simpler, and thus this doesn't seem to be necessary
#uncertainties_display_tables = {}
#for collision_system in tables:
#    uncertainties_display_tables[collision_system] = None
#    min_max_values_per_grooming_method = {}
#    for grooming_method in grooming_methods_for_letter:
#        sources, _, df_relative = tables[collision_system][grooming_method]
#        min_max_values = {}
#        for source in sources:
#            min_max_values[f"{source}_minimum"] = df_relative[f"{source}_maximum"].min()
#            min_max_values[f"{source}_maximum"] = df_relative[f"{source}_maximum"].max()
#        min_max_values_per_grooming_method[grooming_method_to_display_name[grooming_method]] = min_max_values
#    # NOTE: The transpose ensures that the sources columns, while each grooming method is a new row
#    uncertainties_display_tables[collision_system] = pd.DataFrame(min_max_values_per_grooming_method).transpose()

# %%
uncertainties_display_tables


# %%
def format_range(min_val: float, max_val: float, sig_digits: int) -> str:  # noqa: ARG001
    #return fr"{min_val * 100:.{sig_digits}g} -- {max_val * 100:.{sig_digits}g}\%"
    min_value = f"{round(min_val * 100):g}" if not np.isnan(min_val) else "NaN"
    max_value = f"{round(max_val * 100):g}" if not np.isnan(max_val) else "NaN"
    return fr"{min_value} -- {max_value}\%"

output_display_tables = {}
for collision_system, df in uncertainties_display_tables.items():
    # Collect the uncertainty sources to format with the min and max
    # NOTE: We skip the stat uncert and the centrality - we'll bring along the centrality below.
    columns = [k for k in df.columns if k not in ["stat_min", "stat_max", "centrality"]]
    print(f"{columns}")
    columns = set("_".join(k.split("_")[:-1]) for k in columns)  # noqa: C401
    print(f"{columns}")

    output_display_values = {}
    for source in columns:
        print(f"handling source: {source}")
        #df[source] = df.apply(lambda row: format_range(row[f"{source}_minimum"], row[f"{source}_maximum"], 2), axis=1)
        # Add a column for each source, set to the formatted range
        output_display_values[source] = df.apply(lambda row, source=source: format_range(row[f"{source}_min"], row[f"{source}_max"], 1), axis=1)
    # Add back in the centrality:
    if "centrality" in df.columns:
        output_display_values["centrality"] = df["centrality"]
    output_display_tables[collision_system] = pd.DataFrame(output_display_values)

# %%
# How to order and rename the columns for putting into the supplement text
order_and_rename_columns = {
    "centrality": "",
    "tracking_efficiency": "Trk. Eff.",
    "background_sub": "Bkgd. Sub.",
    "unfolding": "Unfolding",
    "model_dependence": "Generator",
    "non_closure": "Non-closure",
    "quadrature": "Total",
}

# %%
# Check how we're doing...
output_display_tables["PbPb"][["centrality", "tracking_efficiency", "model_dependence", "unfolding", "quadrature"]]

# %%
# Rename columns and write LaTeX tables
latex_tables = {}
for collision_system in output_display_tables:
    selected_order_and_rename_columns = {k: v for k, v in order_and_rename_columns.items() if k in output_display_tables[collision_system].columns}
    print(selected_order_and_rename_columns)
    latex_tables[collision_system] = (
        output_display_tables[collision_system][
            list(selected_order_and_rename_columns.keys())
        ].rename(columns=selected_order_and_rename_columns)
        .to_latex(escape=False)
    )

# %%
for collision_system in latex_tables:
    print(f"{collision_system=}")
    print(latex_tables[collision_system])
    print()
#print(latex_tables["pp"])
#output_display_tables["pp"][list(selected_order_and_rename_columns.keys())].rename(columns=selected_order_and_rename_columns)

# %%
central_R02_unfolded_with_systematics["dynamical_kt"].data.axes[0].bin_centers

# %% [markdown]
# # Hepdata

# %% [markdown]
# ## Letter

# %%
import hepdata_lib

from jet_substructure.analysis import full_results_helpers, plot_style  # noqa: F811

# Setup for hep data
# Note:
# - use 3 significant digits for values
# - use 2 significant digits for uncertainties
n_significant_digits_values = 3
n_significant_digits_uncertainty = 2
submission = hepdata_lib.Submission()
submission.comment = r"""The ALICE Collaboration reports measurements of the large relative transverse momentum ($k_{\text{T}}$) component of jet substructure in pp and in central and semicentral Pb$-$Pb collisions at center-of-mass energy per nucleon pair $\sqrt{s_{\text{NN}}}=5.02$ TeV.  Enhancement in the yield of such large-$k_{\text{T}}$ emissions in central Pb$-$Pb collisions is predicted to arise from partonic scattering with quasi-particles of the quark--gluon plasma.  The analysis utilizes charged-particle jets reconstructed by the anti-$k_{\text{T}}$ algorithm with resolution parameter $R=0.2$ in the transverse-momentum interval $60 < p_{\text{T,ch jet}} < 80$\:$\text{GeV}/c$.  The soft drop and dynamical grooming algorithms are used to identify high transverse momentum splittings in the jet shower.  Comparison of measurements in Pb$-$Pb and pp collisions shows medium-induced narrowing, corresponding to yield suppression of high-$k_{\text{T}}$ splittings, in contrast to the expectation of yield enhancement due to quasi-particle scattering.  The measurements are compared to theoretical model calculations incorporating jet quenching, both with and without quasi-particle scattering effects.  These measurements provide new insight into the underlying mechanisms and theoretical modeling of jet quenching.  """
table_index = 0

# Parameters
paper_grooming_styles = plot_style.define_paper_grooming_styles()
_event_activity_full_label_map = {
    "pp": "pp",
    "central": r"0--10% Pb-Pb",
    "semi_central": r"30--50% Pb-Pb",
}
jet_R = 0.2
_jet_pt_bin = helpers.JetPtRange(60, 80)
parameters = {
    "jet_pt_bin": fr"${_jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$",
    "soft_drop_z_cut_02": paper_grooming_styles["soft_drop_z_cut_02"].label + r", $\beta = 0$",
    "dynamical_kt": paper_grooming_styles["dynamical_kt"].label,
}
hists={
    "pp": pp_R02_unfolded_with_systematics,
    "semi_central": semi_central_R02_unfolded_with_systematics,
    "central": central_R02_unfolded_with_systematics,
}
event_activity_to_kt_range={
    "pp": {grooming_method: helpers.KtRange(0.25, 6) for grooming_method in grooming_methods},
    "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
    "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
}


# %%
def reactions_label(collision_system: str, is_ratio: str) -> list[str]:
    if is_ratio:
        return ["P P --> jet+X", "Pb Pb --> jet+X"]
    if collision_system == "pp":
        return ["P P --> jet+X"]
    return ["Pb Pb --> jet+X"]
def centrality_numerical_range(collision_system: str) -> str:
    if collision_system == "central":
        return "[0, 10]"
    if collision_system == "semi_central":
        return "[30, 50]"
    return ""


# %% [markdown]
# ### Figure 1

# %%
# Approach using individual tables...
figure_1_tables = []
table_index = 1

for grooming_method in grooming_methods_for_letter:
    #location = "Figure 1" + ("(left)" if grooming_method == "dynamical_kt" else "(right)")
    #location += ", "
    for collision_system in ["pp", "semi_central", "central"]:
        # Individual tables
        table = hepdata_lib.Table(f'Table {table_index}')

        sys_uncert_other_contributors = ["unfold"]
        if collision_system != "pp":
            sys_uncert_other_contributors.extend(["bkg", "non_closure"])
        sys_uncert_others_minimal_label = ",".join(sys_uncert_other_contributors)
        # We want this to match the label in the table, so we don't format it more nicely.
        #sys_uncert_others_label = ", ".join([f'"{s}"' for s in sys_uncert_other_contributors])
        sys_uncert_others_label = sys_uncert_others_minimal_label

        # Header
        table.keywords["reactions"] = reactions_label(collision_system, is_ratio=False)
        table.keywords["cmenergies"] = ["5020"]
        table.keywords["observables"] = [r"Groomed $k_{\text{T,g}}$ spectra"]
        # Basic description
        table.description = r"Groomed relative transverse momentum, $k_{\text{T,g}}$, spectra measured in " + _event_activity_full_label_map[collision_system] + " collisions."
        # Measurement parameters
        table.description += "\n" + f"{parameters['jet_pt_bin']}, {parameters[grooming_method]}"
        # Uncertainty treatment
        table.description += "\n\n" + fr'For the "trk eff" and "generator" systematic uncertainty sources, the signed systematic uncertainty breakdowns ($\pm$ vs. $\mp$) denote correlation across bins. For the remaining source(s) ("{sys_uncert_others_label}"), no correlation information is specified (i.e. $\pm$ is always used). In the publication, the quadrature sum of all sources of systematic uncertainty is reported, neglecting the sign information reported here.'
        table.location = "Figure 1 " + ("(left)" if grooming_method == "dynamical_kt" else "(right)")
        #x_label = r"$k_{\text{T}}\:(\text{GeV}/c)$"
        #y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$"
        x_label = r"$k_{\text{T}}$"
        y_label = r"$\frac{1}{N_{\text{jets}}}\:\frac{\text{d}N}{\text{d}k_{\text{T,g}}}$"

        # Retrieve data
        h = hists[collision_system][grooming_method].data
        # Select range to display.
        h = full_results_helpers.select_hist_range(h, event_activity_to_kt_range[collision_system][grooming_method])

        # Axes
        x = hepdata_lib.Variable(name=x_label, is_independent=True, is_binned=True, units=r"GeV/c")
        x.digits = n_significant_digits_values
        # Bin edges of the form [(low, high), ...]
        x.values = list(zip(h.axes[0].bin_edges[:-1], h.axes[0].bin_edges[1:], strict=True))
        y = hepdata_lib.Variable(name=y_label, is_independent=False, is_binned=False, units=r"(GeV/c)^{-1}")
        y.digits = n_significant_digits_values
        y.values = h.values

        y.add_qualifier("RE", ", ".join(reactions_label(collision_system, is_ratio=False)))
        cent_range = centrality_numerical_range(collision_system)
        if cent_range:
            y.add_qualifier("CENTRALITY", cent_range)
        y.add_qualifier("SQRT(S)", 5.02, "TeV")
        y.add_qualifier("ETARAP", "|0.9-R|")
        y.add_qualifier("jet radius", str(jet_R))
        y.add_qualifier("jet method", "Anti-$k_{T}$")

        # Define uncertainties
        # Stat
        uncertainties = []
        stat = hepdata_lib.Uncertainty("stat", is_symmetric=True)
        stat.values = [float(f"{dy:.2g}") for dy in h.errors]
        uncertainties.append(stat)

        # Systematics
        systematic_uncertainties = []
        # Break out the signed uncertainties
        signed_uncertainties_to_report = ["tracking_efficiency", "model_dependence"]
        systematic_uncertainty_display_names = {
            "tracking_efficiency": "sys,trk_eff",
            "model_dependence": "sys,gen",
        }
        for k in signed_uncertainties_to_report:
            uncertainty_values = h.metadata["y_systematic"][k]
            # low is an absolute value (below the nominal value), so we should assign that to -1 when it's nonzero and -1 otherwise
            # high is an absolute value (above the nominal value), so we should assign that to 1 when it nonzero and -1 otherwise
            sign_from_low = np.where(uncertainty_values.low != 0, -1, 1)
            sign_from_high = np.where(uncertainty_values.high != 0, 1, -1)
            # Cross check
            assert np.allclose(sign_from_low, sign_from_high)
            sign = sign_from_high
            values = np.where(sign == -1, uncertainty_values.low, uncertainty_values.high)
            # Encode the signed values
            values = sign * values
            sys_uncert = hepdata_lib.Uncertainty(systematic_uncertainty_display_names[k], is_symmetric=True)
            sys_uncert.values = [float(f"{dy:.2g}") for dy in values]
            systematic_uncertainties.append(sys_uncert)

        # Recalculate standard quadrature sum of the rest of the systematics
        keys_to_skip = ["quadrature", *signed_uncertainties_to_report]
        uncertainty_values = full_results_helpers.AsymmetricErrors(
            low=np.sqrt(
                np.sum(
                    [
                        v.low ** 2
                        for k, v in h.metadata["y_systematic"].items()
                        if k not in keys_to_skip
                    ],
                    axis=0,
                )
            ),
            high=np.sqrt(
                np.sum(
                    [
                        v.high ** 2
                        for k, v in h.metadata["y_systematic"].items()
                        if k not in keys_to_skip
                    ],
                    axis=0,
                )
            ),
        )
        sys_uncert = hepdata_lib.Uncertainty(f"sys,{sys_uncert_others_minimal_label}", is_symmetric=True)
        sys_uncert.values = [float(f"{dy:.2g}") for dy in uncertainty_values.high]
        systematic_uncertainties.append(sys_uncert)

        # Add tables to submission
        table.add_variable(x)
        table.add_variable(y)
        y.add_uncertainty(stat)
        for sys_uncert in systematic_uncertainties:
            y.add_uncertainty(sys_uncert)

        figure_1_tables.append(table)
        table_index += 1

# %%
# Testing...
import numpy as np

from_low = np.where(h.metadata["y_systematic"]["tracking_efficiency"].low != 0, -1, 1)
from_high = np.where(h.metadata["y_systematic"]["tracking_efficiency"].high != 0, 1, -1)
# low is an absolute value (below the nominal value), so we should assign that to -1 when it's nonzero and -1 otherwise
# high is an absolute value (above the nominal value), so we should assign that to 1 when it nonzero and -1 otherwise
assert np.allclose(from_low, from_high)
#h.metadata["y_systematic"]["tracking_efficiency"].low

# %%
# Attempted to use a shared table for e.g. Fig 1 left and Fig 1 right.
# However, this doesn't work since the measured range is not consistent for DyG, so we just have to split it out :-(
figure_1_tables = []
table_index = 0
for grooming_method in grooming_methods_for_letter:
    # Shared tables
    table = hepdata_lib.Table(f"Table {table_index}, Fig. 1")

    # Header
    table.keywords["reactions"] = reactions_label(collision_system, is_ratio=True)
    table.keywords["cmenergies"] = ['5020']
    # Basic description
    table.description = r"Groomed $k_{\text{T,g}} spectra measured in"
    table.description += ", ".join([_event_activity_full_label_map[s] for s in ["pp", "central"]])
    table.description += f"and {_event_activity_full_label_map['semi_central']} collisions."
    # Measurement parameters
    table.description += "\n" + f"{parameters['jet_pt_bin']}, {parameters[grooming_method]}"
    table.location = "Figure 1" + ("(left)" if grooming_method == "dynamical_kt" else "(right)")
    #x_label = r"$k_{\text{T}}\:(\text{GeV}/c)$"
    #y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$"
    x_label = r"$k_{\text{T}}$"
    y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}$"

    first_loop = True

    for collision_system in ["pp", "semi_central", "central"]:
        # Individual tables
        #table = hepdata_lib.Table(f'Table {table_index}')

        ## Header
        #table.keywords["reactions"] = reactions_label(collision_system, is_ratio=False)
        #table.keywords["cmenergies"] = ['5020']
        ## Basic description
        #table.description = r"Groomed $k_{\text{T,g}} spectra measured in " + _event_activity_full_label_map[collision_system] + "."
        ## Measurement parameters
        #table.description += "\n" + f"{parameters['jet_pt_bin']}, {parameters[grooming_method]}"
        #table.location = "Figure 1" + ("(left)" if grooming_method == "dynamical_kt" else "(right)")
        ##x_label = r"$k_{\text{T}}\:(\text{GeV}/c)$"
        ##y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$"
        #x_label = r"$k_{\text{T}}$"
        #y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}$"

        # Retrieve data
        h = hists[collision_system][grooming_method].data
        # Select range to display.
        h = full_results_helpers.select_hist_range(h, event_activity_to_kt_range[collision_system][grooming_method])

        # Axes
        if first_loop:
            x = hepdata_lib.Variable(name=x_label, is_independent=True, is_binned=True, units=r"GeV/c")
            x.digits = n_significant_digits_values
            # Bin edges of the form [(low, high), ...]
            x.values = list(zip(h.axes[0].bin_edges[:-1], h.axes[0].bin_edges[1:], strict=True))
            table.add_variable(x)
            first_loop = False

        y = hepdata_lib.Variable(name=y_label, is_independent=False, is_binned=False, units=r"(GeV/c)^{-1}")
        y.digits = n_significant_digits_values
        y.values = h.values

        y.add_qualifier("RE", reactions_label(collision_system, is_ratio=False))
        cent_range = centrality_numerical_range(collision_system)
        if cent_range:
            y.add_qualifier("CENTRALITY", cent_range)
        y.add_qualifier("SQRT(S)", 5.02, "TeV")
        y.add_qualifier("ETARAP", "|0.9-R|")
        y.add_qualifier("jet radius", str(jet_R))
        y.add_qualifier("jet method", "Anti-$k_{T}$")

        # Define uncertainties
        # Stat
        stat = hepdata_lib.Uncertainty("stat", is_symmetric=True)
        stat.values = [float(f"{dy:.2g}") for dy in h.errors]

        # Systematics
        # TODO: ...
        #if self.centrality == [0,10]:
        #    h_sys = hepdata_lib.root_utils.get_hist_1d_points(h_sys)
        #    sys = hepdata_lib.Uncertainty('sys', is_symmetric=True)
        #    sys.values = [float('{:.2g}'.format(dy)) for dy in h_sys['dy']]
        #elif self.centrality == [30,50]:
        #    h_sys = hepdata_lib.root_utils.get_graph_points(h_sys)
        #    sys = hepdata_lib.Uncertainty('sys', is_symmetric=False)
        #    sys.values = [(float('{:.2g}'.format(dy[0])), float('{:.2g}'.format(dy[1]))) for dy in h_sys['dy']]
        #y.add_uncertainty(sys)

        # Add tables to submission
        table.add_variable(y)
        y.add_uncertainty(stat)

    figure_1_tables.append(table)
    table_index += 1

# %% [markdown]
# ### Figure 2
#
# For now, I skip this. The figures are identical to those in Figure 1 - just divided by an arbitrary parametrization, whose parameters are listed in the paper. It's really just a different way to visualize the data (and models), so there's no point in separately reporting it.

# %% [markdown]
# ### Figure 3

# %%
# Unfortunately, I need to recalculate the ratio :-/
figure_3_tables = []
table_index = 7

for grooming_method in grooming_methods_for_letter:
    h_reference_original = hists["pp"][grooming_method].data
    # Select range to display.
    for collision_system in ["semi_central", "central"]:
        # Individual tables
        table = hepdata_lib.Table(f'Table {table_index}')

        sys_uncert_other_contributors = ["unfold"]
        if collision_system != "pp":
            sys_uncert_other_contributors.extend(["bkg", "non_closure"])
        sys_uncert_others_minimal_label = ",".join(sys_uncert_other_contributors)
        # We want this to match the label in the table, so we don't format it more nicely.
        #sys_uncert_others_label = ", ".join([f'"{s}"' for s in sys_uncert_other_contributors])
        sys_uncert_others_label = sys_uncert_others_minimal_label

        # Header
        table.keywords["reactions"] = reactions_label(collision_system, is_ratio=True)
        table.keywords["cmenergies"] = ["5020"]
        table.keywords["observables"] = [r"Groomed relative transverse momentum, $k_{\text{T,g}}$, PbPb/pp ratio"]
        # Basic description
        table.description = r"Groomed $k_{\text{T,g}}$ ratio of " + _event_activity_full_label_map[collision_system] + " to pp collisions."
        # Measurement parameters
        table.description += "\n" + f"{parameters['jet_pt_bin']}, {parameters[grooming_method]}"
        # Uncertainty treatment
        table.description += "\n\n" + fr'For the "trk eff" and "generator" systematic uncertainty sources, the signed systematic uncertainty breakdowns ($\pm$ vs. $\mp$) denote correlation across bins. For the remaining source(s) ("{sys_uncert_others_label}"), no correlation information is specified (i.e. $\pm$ is always used). In the publication, the quadrature sum of all sources of systematic uncertainty is reported, neglecting the sign information reported here.'
        location_collision_system = {
            "semi_central": "upper",
            "central": "lower",
        }
        location_grooming = {
            "soft_drop_z_cut_02": "right",
            "dynamical_kt": "left",
        }
        table.location = f"Figure 3 ({location_collision_system[collision_system]} {location_grooming[grooming_method]})"
        #x_label = r"$k_{\text{T}}\:(\text{GeV}/c)$"
        #y_label = r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T,g}}\:(\text{GeV}/c)^{-1}$"
        x_label = r"$k_{\text{T}}$"
        y_label = r"$\text{Pb--Pb}/\text{pp}$"

        # Retrieve data
        # Reference
        h_reference = full_results_helpers.select_hist_range(h_reference_original, event_activity_to_kt_range[collision_system][grooming_method])
        # PbPb
        h = hists[collision_system][grooming_method].data
        # Select range to display.
        h = full_results_helpers.select_hist_range(h, event_activity_to_kt_range[collision_system][grooming_method])
        ratio = h / h_reference

        # Axes
        x = hepdata_lib.Variable(name=x_label, is_independent=True, is_binned=True, units=r"GeV/c")
        x.digits = n_significant_digits_values
        # Bin edges of the form [(low, high), ...]
        x.values = list(zip(ratio.axes[0].bin_edges[:-1], ratio.axes[0].bin_edges[1:], strict=True))
        y = hepdata_lib.Variable(name=y_label, is_independent=False, is_binned=False, units="")
        y.digits = n_significant_digits_values
        y.values = ratio.values

        y.add_qualifier("RE", ", ".join(reactions_label(collision_system, is_ratio=True)))
        cent_range = centrality_numerical_range(collision_system)
        if cent_range:
            y.add_qualifier("CENTRALITY", cent_range)
        y.add_qualifier("SQRT(S)", 5.02, "TeV")
        y.add_qualifier("ETARAP", "|0.9-R|")
        y.add_qualifier("jet radius", str(jet_R))
        y.add_qualifier("jet method", "Anti-$k_{T}$")

        # Define uncertainties
        # Stat
        uncertainties = []
        stat = hepdata_lib.Uncertainty("stat", is_symmetric=True)
        stat.values = [float(f"{dy:.2g}") for dy in ratio.errors]
        uncertainties.append(stat)

        # Systematics
        systematic_uncertainties = []
        # Break out the signed uncertainties
        signed_uncertainties_to_report = ["tracking_efficiency", "model_dependence"]
        systematic_uncertainty_display_names = {
            "tracking_efficiency": "sys,trk_eff",
            "model_dependence": "sys,gen",
        }
        for k in signed_uncertainties_to_report:
            uncertainty_values = ratio.metadata["y_systematic"][k]
            # low is an absolute value (below the nominal value), so we should assign that to -1 when it's nonzero and -1 otherwise
            # high is an absolute value (above the nominal value), so we should assign that to 1 when it nonzero and -1 otherwise
            sign_from_low = np.where(uncertainty_values.low != 0, -1, 1)
            sign_from_high = np.where(uncertainty_values.high != 0, 1, -1)
            # Cross check
            assert np.allclose(sign_from_low, sign_from_high)
            sign = sign_from_high
            values = np.where(sign == -1, uncertainty_values.low, uncertainty_values.high)
            # Encode the signed values
            values = sign * values
            sys_uncert = hepdata_lib.Uncertainty(systematic_uncertainty_display_names[k], is_symmetric=True)
            sys_uncert.values = [float(f"{dy:.2g}") for dy in values]
            systematic_uncertainties.append(sys_uncert)

        # Recalculate standard quadrature sum of the rest of the systematics
        keys_to_skip = ["quadrature", *signed_uncertainties_to_report]
        uncertainty_values = full_results_helpers.AsymmetricErrors(
            low=np.sqrt(
                np.sum(
                    [
                        v.low ** 2
                        for k, v in ratio.metadata["y_systematic"].items()
                        if k not in keys_to_skip
                    ],
                    axis=0,
                )
            ),
            high=np.sqrt(
                np.sum(
                    [
                        v.high ** 2
                        for k, v in ratio.metadata["y_systematic"].items()
                        if k not in keys_to_skip
                    ],
                    axis=0,
                )
            ),
        )
        sys_uncert = hepdata_lib.Uncertainty(f"sys,{sys_uncert_others_minimal_label}", is_symmetric=True)
        sys_uncert.values = [float(f"{dy:.2g}") for dy in uncertainty_values.high]
        systematic_uncertainties.append(sys_uncert)

        # Add tables to submission
        table.add_variable(x)
        table.add_variable(y)
        y.add_uncertainty(stat)
        for sys_uncert in systematic_uncertainties:
            y.add_uncertainty(sys_uncert)

        figure_3_tables.append(table)
        table_index += 1


# %% [markdown]
# ### Finalize submission

# %%
for t in figure_1_tables:
    submission.add_table(t)
for t in figure_3_tables:
    submission.add_table(t)

# %%
# Write out the outputs
hepdata_output_dir = output_dir / "hepdata" / "letter"
submission.create_files(str(hepdata_output_dir), remove_old=True)

# %% [markdown]
# # Model-only comparison plot (excerpt for checks - not for paper)

# %% [markdown]
# ### R = 0.2

# %%
jet_R = 0.2
jet_R_str = f"R{int(jet_R*10):02}"
_output_dir = output_dir / "comparison" / "unfolding" / plot_output_dir_tag / substructure_variable / jet_R_str
_output_dir.mkdir(parents=True, exist_ok=True)

def create_plots(models: list[str]) -> None:
    _additional_hists = {
        "semi_central": semi_central_R02_unfolded_with_systematics,
        "central": central_R02_unfolded_with_systematics,
    }
    models_calculation = {}
    if "hybrid" in models:
        models_calculation.update({
            "hybrid_without_moliere": hybrid_model_with_wake_without_moliere_predictions_R02,
            "hybrid_moliere": hybrid_model_with_wake_with_moliere_predictions_R02,
        })
    if "JETSCAPE" in models:
        models_calculation.update({
            "jetscape": jetscape_predictions_R02,
        })
    if "JEWEL" in models:
        models_calculation.update({
            "jewel_recoils": jewel_recoils_predictions_R02,
            "jewel_no_recoils": jewel_no_recoils_predictions_R02,
        })
    event_activity_to_kt_range={
        "pp": {gm: helpers.KtRange(0.25, 6) for gm in grooming_methods},
        "semi_central": PbPb_kt_measured_range_by_grooming_method(event_activity="semi_central"),
        "central": PbPb_kt_measured_range_by_grooming_method(event_activity="central"),
    }
    kt_display_range = (0.0, 6.25)
    text_font_size = 24
    _ratio_range = (0.3, 1.7)
    for event_activity in ["central"]:
        from jet_substructure.analysis import full_results_helpers, plot_style

        grooming_styles = plot_style.define_paper_grooming_styles()
        hists={
            "pp": pp_R02_unfolded_with_systematics,
            event_activity: _additional_hists[event_activity],
        }
        for grooming_method in grooming_methods_for_letter:
            style = grooming_styles[grooming_method]

            plot_config = pb.PlotConfig(
                name=f"{'_'.join(models)}_comparison_{grooming_method}_{event_activity}",
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
                            #pb.TextConfig(x=0.98, y=0.98, text=text, font_size=text_font_size),
                            # Add the grooming label in a separate location in the bottom left
                            # Otherwise, it will overlap with the data
                            pb.TextConfig(x=0.02, y=0.02, text=style.label, font_size=text_font_size),
                        ],
                        legend=pb.LegendConfig(location="lower left", font_size=text_font_size, anchor=(0.025, 0.10), marker_label_spacing=0.),
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
                        #legend=pb.LegendConfig(location="lower left", font_size=22, anchor=(0.01, 0.02), ncol=2, marker_label_spacing=0.05, label_spacing=0.1, handle_height=1.3, column_spacing=0.30)
                    ),
                ],
                figure=pb.Figure(edge_padding={"left": 0.1525, "bottom": 0.095, "top": 0.975}),
            )

            fig, (ax, ax_ratio) = plt.subplots(
                2,
                1,
                figsize=(10, 10),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )

            #for collision_system in ["pp", event_activity]:
            for collision_system in ["pp"]:
                # Axes: jet_pt, attr_name
                h_input = hists[collision_system][grooming_method].data

                # Select range to display.
                h = full_results_helpers.select_hist_range(h_input, event_activity_to_kt_range[collision_system][grooming_method])

                # Spectra
                for model_name, model_calculation in models_calculation.items():
                    model = model_calculation.spectra(event_activity=collision_system).get(grooming_method, None)
                    if not model:
                        logger.debug(f"{model_calculation.ratio(event_activity=collision_system)}")
                        logger.debug(
                            f"Skipping model {model_name}, grooming method: {grooming_method}, {collision_system} because predictions aren't available"
                        )
                        continue

                    ## Select the relevant kt range
                    #model = full_results_helpers.select_hist_range(
                    #    model, event_activity_to_kt_range[collision_system][grooming_method]
                    #)

                    ## We want the model ratio binning to match the data ratio binning, so we rebin here as necessary,
                    ## making a note that this is what we've done. This predominately applies to the hybrid model, which
                    ## is binned more finely. Since I said I would rebin, and it's not especially fair to compare models
                    ## with different binning, it's better to just normalize it here.
                    #if (
                    #    len(h.axes[0].bin_edges) != len(model.axes[0].bin_edges) or
                    #    not np.allclose(h.axes[0].bin_centers, model.axes[0].bin_centers)
                    #):
                    #    logger.info(f"Rebinned model '{model_name}' to match data binning for {grooming_method}, {collision_system}")
                    #    model = full_results_helpers.rebin_bin_width_scaled_hist(
                    #        h_to_rebin=model,
                    #        h_target_axis=h.axes[0],
                    #        # This is okay since the data is explicitly constructed without systematic uncertainties.
                    #        okay_for_systematic_not_to_exist=True,
                    #    )

                    # Fill between
                    # This could definitely be coded more elegantly, but good enough to get something out.
                    temp_kwargs = plot_paper.retrieve_model_styles(
                        event_activity="PbPb" if collision_system != "pp" else "pp",
                        model_name=model_name if collision_system != "pp" else "jewel",
                    )
                    temp_kwargs["facecolor"] = temp_kwargs.pop("color")
                    temp_kwargs["label"] = model_calculation.label(collision_system=collision_system)
                    temp_kwargs.pop("marker")
                    # Skip plotting for JEWEL w/ recoils for pp (just so we don't have it twice...)
                    models_to_skip_plotting_for_pp = ["jewel_recoils", "hybrid_without_moliere"]
                    if not (model_name in models_to_skip_plotting_for_pp and collision_system == "pp"):
                        ax.fill_between(
                            model.axes[0].bin_centers,
                            model.values - model.errors,
                            model.values + model.errors,
                            alpha=0.7,
                            **temp_kwargs,
                        )

                    # pp ratios aren't meaningful
                    if collision_system == "pp":
                        continue

                    # Ratio
                    model = model_calculation.ratio(event_activity=collision_system).get(grooming_method, None)
                    if not model:
                        logger.debug(f"{model_calculation.ratio(event_activity=collision_system)}")
                        logger.debug(
                            f"Skipping model {model_name}, grooming method: {grooming_method}, {collision_system} because predictions aren't available"
                        )
                        continue

                    # Select the relevant kt range
                    #model = full_results_helpers.select_hist_range(
                    #    model, event_activity_to_kt_range[collision_system][grooming_method]
                    #)

                    # We want the model ratio binning to match the data ratio binning, so we rebin here as necessary,
                    # making a note that this is what we've done. This predominately applies to the hybrid model, which
                    # is binned more finely. Since I said I would rebin, and it's not especially fair to compare models
                    # with different binning, it's better to just normalize it here.
                    #if (
                    #    len(h.axes[0].bin_edges) != len(model.axes[0].bin_edges) or
                    #    not np.allclose(h.axes[0].bin_centers, model.axes[0].bin_centers)
                    #):
                    #    logger.info(f"Rebinned model '{model_name}' to match data binning for {grooming_method}, {collision_system}")
                    #    model = full_results_helpers.rebin_bin_width_scaled_hist(
                    #        h_to_rebin=model,
                    #        h_target_axis=h.axes[0],
                    #        # This is okay since the data is explicitly constructed without systematic uncertainties.
                    #        okay_for_systematic_not_to_exist=True,
                    #    )

                    # Fill between
                    # NOTE: This is assuming we'll only plot PbPb model colors here, but I think that's a reasonable assumption,
                    #       since that's the only models that could compare to the PbPb/pp ratio
                    temp_kwargs = plot_paper.retrieve_model_styles(event_activity="PbPb", model_name=model_name)
                    temp_kwargs["facecolor"] = temp_kwargs.pop("color")
                    temp_kwargs["label"] = model_calculation.label(collision_system="PbPb")
                    temp_kwargs.pop("marker")
                    ax_ratio.fill_between(
                        model.axes[0].bin_centers,
                        model.values - model.errors,
                        model.values + model.errors,
                        alpha=0.7,
                        **temp_kwargs,
                    )

            # Reference value for ratio
            ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=0.9)

            plot_config.apply(fig=fig, axes=(ax, ax_ratio))

            # Plot, save, and cleanup
            filename = f"{plot_config.name}"
            fig.savefig(_output_dir / f"{filename}.pdf")
            plt.close(fig)

# Just put into a figure so we don't pollute the global namespace
create_plots(models = ["hybrid", "JETSCAPE", "JEWEL"])

# %%
jewel_no_recoils_predictions_R02.spectra(event_activity="central")["dynamical_kt"]

# %%

# %%

# %% [markdown]
# # Debug area

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# %%
fig, ax = plt.subplots(figsize=(8, 6))

# %%
# Trying to erase lines that are underneath open markers, since mpl can't do it itself...
x = np.linspace(0, 10, 20)
x2 = x + 0.01
y = np.sin(x)
y2 = np.sin(x2)

# %%
x1_res = ax.errorbar(x=x, y=y, xerr=np.sqrt(np.abs(y)), yerr=np.sqrt(np.abs(y)), fmt="o", fillstyle="none", label="x")
x2_res = ax.errorbar(x=x2, y=y2, xerr=np.sqrt(np.abs(y2)),  yerr=np.sqrt(np.abs(y2)), fmt="o", fillstyle="none", label="x2")

# %%
fig

# %%
x2_res.lines[2][1].get_segments()

# %%
x2_res.lines[0].get_markeredgewidth()

# %%
# Find the position of the left edge of the marker in display coordinates
# NOTE: This is dependent on the marker being symmetric
marker_center_in_display_coordinates = ax.transData.transform((x[1], y[1]))
marker_left_edge_in_display_coordinates = marker_center_in_display_coordinates - (x2_res.lines[0].get_markersize()/ 2, 0)
marker_left_edge_in_display_coordinates

# %%
# Translate back from display coordinates to data coordinates
data_edge = ax.transData.inverted().transform(marker_left_edge_in_display_coordinates)
data_edge

# %%
ax.plot(*data_edge, marker="o", markersize=5, color="black")

# %%
fig

# %%
plt.close(fig)


# %%

# %% [markdown]
# ## Nov 2023

# %%
# Testing to determine how best to put text above the figure. The best approach seems to be using `Figure.text`,
# but a possible alternative is to keep a blank axis and ten use `set_axis_off`
def test_axis(output_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, 5, 5, 5]}, sharex=True)
    # Plot text on the figure. Note that this isn't accounted for with the subplots_adjust.
    fig.text(0.5, 0.99, "Testing fig\nNew Line", va="center", ha="center", fontsize=31)

    axes[1].plot(np.linspace(0, 10, num=100), np.linspace(0, 10, num=100))

    # Try a blank axis
    axes[0].text(0.1, 0.5, "Testing up here")
    axes[0].set_axis_off()

    fig.tight_layout()
    fig.subplots_adjust(
        hspace=0, wspace=0, left=0.10, bottom=0.105, right=0.98, top=0.7
    )
    fig.savefig(output_dir / "test.pdf")
    plt.close(fig)

test_axis(output_dir=_output_dir)

# %%
