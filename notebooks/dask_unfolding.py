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

# %load_ext rich

# ### Cross check environment

# +
import logging
import os

print(os.getenv("CONDA_BUILD_SYSROOT"))
print(os.getenv("ROOUNFOLD_ROOT"))
print(os.getenv("LD_LIBRARY_PATH"))
print(os.getenv("PATH"))
# -

print(os.getenv("DASK_CONFIG"))

# # Running jobs

# ## General setup

# +
from importlib import reload

from mammoth import job_utils

from jet_substructure.analysis import parsl as job_runner
# -

# ## Reload

#reload(job_utils)
reload(job_runner)

# ## Tasks

# Define the variables separately so we can change the parameters but keep the session alive
job_executor, job_cluster = None, None

# ### Data frame

# +
# Settings
# Base settings
job_framework = job_utils.JobFramework.dask_delayed
#job_framework = job_utils.JobFramework.immediate_execution_debug
facility: job_utils.FACILITIES = "rehlers_mbp_m1pro"
conda_environment_name = "substructure_c_28_04"

# Base analysis settings
#base_dataset_name = "pp_R02"
#collision_system = "pp"
#collision_system = "pythia"
#base_dataset_name = "PbPb_semi_central_R02_pass3"
#collision_system = "PbPb"
#collision_system = "embed_pythia"
base_dataset_name = "PbPb_central_R02_pass3"
#collision_system = "PbPb"
collision_system = "embed_pythia"
dataset_type = "nominal"
# Detailed analysis settings
jobs_to_execute = [
    "root_data_frame",
    "root_data_frame_response",
]
grooming_methods = [
    # "leading_kt",
    # "leading_kt_z_cut_02",
    # "leading_kt_z_cut_04",
    # "dynamical_z",
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
    "soft_drop_z_cut_02",
    "soft_drop_z_cut_04",
    "dynamical_core_z_cut_02",
    "dynamical_kt_z_cut_02",
    "dynamical_time_z_cut_02",
]
# Just need the default object
unfolding_runtime_settings = job_runner.UnfoldingRuntimeSettings()

# Job execution configuration
task_name = "data_frame_hardest_kt"
task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=4)
if task_config.n_cores_per_task > 1:
    facility = "rehlers_mbp_m1pro_multi_core"
#n_cores_to_allocate = 8
# Formerly n_cores_to_allocate, this new variable is == n_cores_to_allocate if n_cores_per_task == 1
target_n_tasks_to_run_simultaneously = 2
walltime = "24:00:00"
log_level = logging.INFO
debug_mode = False

if debug_mode:
    # Usually, we want to run in the short queue
    target_n_tasks_to_run_simultaneously = 2
    walltime = "1:59:00"

# Keep the job executor just to keep it alive
if job_executor is None or job_cluster is None:
    job_executor, job_cluster = job_runner.setup_job_framework(
        job_framework=job_framework,
        jobs_to_execute=jobs_to_execute,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        conda_environment_name=conda_environment_name,
    )
# -



# ### Unfolding

# +
# Settings
# Base settings
job_framework = job_utils.JobFramework.dask_delayed
#job_framework = job_utils.JobFramework.immediate_execution_debug
facility: job_utils.FACILITIES = "rehlers_mbp_m1pro"
conda_environment_name = "substructure_c_28_04"

# Base analysis settings
# pp
#base_dataset_name = "pp_R02"
#collision_system = "pp"
base_dataset_name = "pp_R04"
collision_system = "pp"
# Semi-central
#base_dataset_name = "PbPb_semi_central_R02_pass3"
#collision_system = "PbPb"
# Central
#base_dataset_name = "PbPb_central_R02_pass3"
#collision_system = "PbPb"
dataset_type = "nominal"
# Detailed analysis settings
jobs_to_execute = [
    "unfolding",
]
grooming_methods = [
    # "leading_kt",
    # "leading_kt_z_cut_02",
    # "leading_kt_z_cut_04",
    # "dynamical_z",
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
    #"soft_drop_z_cut_02",
    #"soft_drop_z_cut_04",
    #"dynamical_core_z_cut_02",
    #"dynamical_kt_z_cut_02",
    #"dynamical_time_z_cut_02",
]
unfolding_runtime_settings = job_runner.UnfoldingRuntimeSettings(
    variable_to_unfold="kt",
    #variable_to_unfold="delta_R",
    #variable_to_unfold="z",
    normalize_variable_by_jet_pt=False,
    selected_settings=[
        #########
        # Nominal
        #########
        "default",
        #"default_delta_R",
        #"default_z",
        ## Systematics
        ## Unfolding
        #"truncation_low",
        #"truncation_high",
        #"random_binning",
        ## Tracking efficiency
        #"tracking_efficiency",
        ## Model dependence in pp
        #"model_dependence_herwig",
        #"model_dependence_pythia",
        ## PbPb background
        #"background_low",
        #"background_high",
        ## PbPb unfolding
        #"reweight_prior",
        ## PbPb thermal model
        #"thermal_model",

        ####################
        # Binning variations
        ####################
        ## ------------------------------
        ## Merge 3-6 bin for SD zcut 0.4
        ## ------------------------------
        #"merge_3_6",
        ## Systematics (copied from above and adapted as needed)
        ## Unfolding
        #"merge_3_6_truncation_low",
        #"merge_3_6_truncation_high",
        #"merge_3_6_random_binning",
        ## Tracking efficiency
        #"merge_3_6_tracking_efficiency",
        ### Model dependence in pp
        ##"merge_3_6_model_dependence_herwig",
        ##"merge_3_6_model_dependence_pythia",
        ## PbPb background
        #"merge_3_6_background_low",
        #"merge_3_6_background_high",
        ## PbPb Unfolding
        #"merge_3_6_reweight_prior",
        ## PbPb thermal model
        #"merge_3_6_thermal_model",

        # ------------------------------
        # Peter's binning
        # ------------------------------
        #"peter_binning",
        ## Systematics (copied from above and adapted as needed)
        ## Unfolding
        #"peter_binning_truncation_low",
        #"peter_binning_truncation_high",
        #"peter_binning_random_binning",
        ## Unfolding PbPb
        #"peter_binning_reweight_prior",
        ## Tracking efficiency
        #"peter_binning_tracking_efficiency",
        ## PbPb background
        #"peter_binning_background_low",
        #"peter_binning_background_high",
    ],
    output_dir_tag="2023-paper",
    #output_dir_tag="2023-conda-test",
)

# Job execution configuration
task_name = "unfolding_hardest_kt"
task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
if task_config.n_cores_per_task > 1:
    facility = "rehlers_mbp_m1pro_multi_core"
#n_cores_to_allocate = 8
# Formerly n_cores_to_allocate, this new variable is == n_cores_to_allocate if n_cores_per_task == 1
# I use 8 when I don't need to do other intensive things at the same time.
target_n_tasks_to_run_simultaneously = 8
# If I have other intensive tasks, I can use 6
#target_n_tasks_to_run_simultaneously = 6
walltime = "24:00:00"
log_level = logging.INFO
debug_mode = False

if debug_mode:
    # Usually, we want to run in the short queue
    n_cores_to_allocate = 2
    walltime = "1:59:00"

# Keep the job executor just to keep it alive
if job_executor is None or job_cluster is None:
    job_executor, job_cluster = job_runner.setup_job_framework(
        job_framework=job_framework,
        jobs_to_execute=jobs_to_execute,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        conda_environment_name=conda_environment_name,
    )
# -

# ## Status

job_executor

# ## Submit jobs

from jet_substructure.cpp import data_frame
reload(data_frame)

futures = job_runner.setup_and_submit_tasks(
    job_framework=job_framework,
    task_config=task_config,
    base_dataset_name=base_dataset_name,
    dataset_type=dataset_type,
    collision_system=collision_system,
    jobs_to_execute=jobs_to_execute,
    input_grooming_methods=grooming_methods,
    unfolding_runtime_settings=unfolding_runtime_settings,
    dask_client=job_executor if job_framework == job_utils.JobFramework.dask_delayed else None,  # type: ignore[arg-type]
)
futures

futures

_res = [f.result() for f in futures]

_res

import numpy as np
np.where([f.status == "error" for f in futures])

# ## Cleanup

job_executor.close()
job_cluster.close()

# # Older experiments

results = job_runner.run(job_framework=job_utils.JobFramework.dask_delayed)
