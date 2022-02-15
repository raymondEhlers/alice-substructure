"""Run mammoth skimming and analysis tasks via parsl

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

import IPython
from mammoth import helpers, job_utils
from mammoth.framework import sources
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from rich.progress import Progress


logger = logging.getLogger(__name__)


@python_app  # type: ignore
def _run_data_skim(
    collision_system: str,
    jet_R: float,
    min_jet_pt: float,
    iterative_splittings: bool,
    loading_data_rename_prefix: Mapping[str, str],
    convert_data_format_prefixes: Mapping[str, str],
    event_activity: str,
    scale_factors: Mapping[int, float],
    pt_hat_bin: int,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    import traceback
    from pathlib import Path

    from jet_substructure.analysis import track_skim_adapter

    try:
        result = track_skim_adapter.hardest_kt_data_skim(
            input_filename=Path(inputs[0].filepath),
            collision_system=collision_system,
            event_activity=event_activity,
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            iterative_splittings=iterative_splittings,
            loading_data_rename_prefix=loading_data_rename_prefix,
            convert_data_format_prefixes=convert_data_format_prefixes,
            scale_factors=scale_factors,
            pt_hat_bin=pt_hat_bin,
            output_filename=Path(outputs[0].filepath),
        )
    except Exception:
        result = (
            False,
            f"failure for {collision_system}, R={jet_R}, {inputs[0].filepath} with: \n{traceback.format_exc()}",
        )
    return result


def setup_calculate_data_skim(
    collision_system: str,
    min_jet_pt: Union[float, Mapping[str, float]],
    jet_R_values: Sequence[float],
    iterative_splittings: bool,
    loading_data_rename_prefix: Mapping[str, str],
    convert_data_format_prefixes: Mapping[str, str],
    input_path: Path,
    event_activity: str = "",
    scale_factors_dataset: str = "",
) -> List[AppFuture]:
    """Analyze hardest kt data"""
    input_files = sorted(input_path.glob("*/*/*.root"))

    # TEMP for testing
    # input_files = input_files[:4]
    # ENDTEMP

    scale_factors = None
    if scale_factors_dataset:
        import jet_substructure.analysis.parsl

        scale_factors = jet_substructure.analysis.parsl.read_extracted_scale_factors(
            # TODO: Unclear if this should be hard coded
            collision_system="pythia",
            dataset_name=scale_factors_dataset,
        )

    results = []
    for i, input_filename in enumerate(input_files):
        if i % 500 == 0:
            logger.info(f"Adding {input_filename} for analysis")

        # The input_file is in trains/collision_system/train_number/run_by_run/period/run_number/filename.root
        # So to get the train directory, we need to take the parent 4 times.
        train_directory = input_filename.parent.parent.parent.parent
        run_dir = input_filename.parent.name
        pt_hat_bin = -1
        # However, if we're looking at pythia, we also need to account for the pt hard bin
        if collision_system == "pythia":
            # In this case, the input_file is in trains/collision_system/train_number/run_by_run/period/run_number/pt_hard_bin/filename.root
            # At least for LHC20g4 and LHC18b8
            # Thus, to get the train directory, we need to take the parent 5 times
            train_directory = input_filename.parent.parent.parent.parent.parent
            # The pt hard bin is the parent dir
            pt_hat_bin = int(str(input_filename.parent.name))
            # For the run dir, we want to include the period, run, and pt hat bin
            # The period is 3 parents up and the run number is 2 parents up
            run_dir = str(Path(input_filename.parent.parent.parent.name) / input_filename.parent.parent.name / str(pt_hat_bin))

        # Further setup
        iterative_splittings_label = "iterative" if iterative_splittings else "recursive"

        for jet_R in jet_R_values:
            # Setup file I/O
            output_dir = train_directory / "skim" / f"R{round(jet_R * 10):02}" / run_dir
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = output_dir / f"{input_filename.stem}_{iterative_splittings_label}_splittings.root"
            results.append(
                _run_data_skim(
                    collision_system=collision_system,
                    event_activity=event_activity,
                    jet_R=jet_R,
                    min_jet_pt=min_jet_pt,
                    iterative_splittings=iterative_splittings,
                    loading_data_rename_prefix=loading_data_rename_prefix,
                    convert_data_format_prefixes=convert_data_format_prefixes,
                    inputs=[File(str(input_filename))],
                    outputs=[File(str(output_filename))],
                    pt_hat_bin=pt_hat_bin,
                    scale_factors=scale_factors,
                )
            )

    return results


@python_app  # type: ignore
def run_embedding_skim(
    collision_system: str,
    jet_R: float,
    min_jet_pt: float,
    iterative_splittings: bool,
    thermal_model_parameters: sources.ThermalModelParameters,
    convert_data_format_prefixes: Mapping[str, str],
    scale_factor: float,
    r_max: float,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    import traceback
    from pathlib import Path

    from jet_substructure.analysis import track_skim_adapter

    try:
        result = track_skim_adapter.hardest_kt_embedding_skim(
            input_filename=Path(inputs[0].filepath),
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            iterative_splittings=iterative_splittings,
            thermal_model_parameters=thermal_model_parameters,
            convert_data_format_prefixes=convert_data_format_prefixes,
            scale_factor=scale_factor,
            r_max=r_max,
            output_filename=Path(outputs[0].filepath),
        )
    except Exception:
        result = (
            False,
            f"failure for {collision_system}, R={jet_R}, {inputs[0].filepath} with: \n{traceback.format_exc()}",
        )
    return result


def setup_calculate_thermal_model_skim(
    collision_system: str,
    min_jet_pt: Union[float, Mapping[str, float]],
    jet_R_values: Sequence[float],
    iterative_splittings: bool,
    convert_data_format_prefixes: Mapping[str, str],
    input_path: Path,
    event_activity: str,
    scale_factors_dataset: str,
    r_max: float = 0.25,
    n_repeat_file: int = 1,
) -> List[AppFuture]:
    """Analyze and skim hardest kt embedding"""
    # NOTE: These are pythia input files. They include an extra "*" to account for the pt hard bin...
    input_files = sorted(input_path.glob("*/*/*/*.root"))

    # TEMP for testing
    #input_files = input_files[:1]
    #input_files = input_files[:4]
    #input_files = input_files[:10]
    # ENDTEMP

    # NOTE: Delayed import since this comes with _a lot_ of other things. Should be refactored...
    import jet_substructure.analysis.parsl
    scale_factors = jet_substructure.analysis.parsl.read_extracted_scale_factors(
        # TODO: Unclear if this should be hard coded
        collision_system="embedPythia",
        dataset_name=scale_factors_dataset,
    )

    thermal_model_parameters = sources.THERMAL_MODEL_SETTINGS[event_activity]

    results = []
    for i, input_filename in enumerate(input_files):
        if i % 500 == 0:
            logger.info(f"Adding {input_filename} for analysis")

        # The input_file is in trains/collision_system/train_number/run_by_run/period/run_number/filename.root
        # So to get the train directory, we need to take the parent 4 times.
        train_directory = input_filename.parent.parent.parent.parent
        run_dir = input_filename.parent.name
        pt_hat_bin = -1
        # However, if we're looking at pythia, we also need to account for the pt hard bin
        #if collision_system == "pythia":
        if True:
            # In this case, the input_file is in trains/collision_system/train_number/run_by_run/period/run_number/pt_hard_bin/filename.root
            # At least for LHC20g4 and LHC18b8
            # Thus, to get the train directory, we need to take the parent 5 times
            train_directory = input_filename.parent.parent.parent.parent.parent
            # The pt hard bin is the parent dir
            pt_hat_bin = int(str(input_filename.parent.name))
            # For the run dir, we want to include the period, run, and pt hat bin
            # The period is 3 parents up and the run number is 2 parents up
            run_dir = str(Path(input_filename.parent.parent.parent.name) / input_filename.parent.parent.name / str(pt_hat_bin))

        # Further setup
        iterative_splittings_label = "iterative" if iterative_splittings else "recursive"

        for jet_R in jet_R_values:
            # Setup file I/O
            output_dir = train_directory / "skim" / collision_system / event_activity / f"R{round(jet_R * 10):02}" / run_dir

            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_repeat_file):
                # NOTE: This includes the period, run, and pt hat
                output_filename = output_dir / f"{input_filename.stem}_{iterative_splittings_label}_splittings_{i:02}.root"
                results.append(
                    run_embedding_skim(
                        collision_system=collision_system,
                        jet_R=jet_R,
                        min_jet_pt=min_jet_pt,
                        iterative_splittings=iterative_splittings,
                        thermal_model_parameters=thermal_model_parameters,
                        convert_data_format_prefixes=convert_data_format_prefixes,
                        scale_factor=scale_factors[pt_hat_bin],
                        r_max=r_max,
                        inputs=[File(str(input_filename))],
                        outputs=[File(str(output_filename))],
                    )
                )

    return results


def run() -> None:
    # Basic setup
    iterative_splittings = True
    jet_R_values = [0.2]
    min_jet_pt = {
        "pp": 5,
        "pythia": {"det_level": 5},
        "embedPythia": 20,
        "PbPb": 20,
        "thermal_model": {"hybrid": 20},
    }
    collision_systems_to_process = ["thermal_model"]
    dataset_name = "LHC18qr_central_642"
    event_activity = "central"

    # NOTE: Need to glob in the task
    input_paths = {
        "pythia": Path("trains/pythia/2619/run_by_run/LHC18b8_fast/"),
        "PbPb": Path("trains/PbPb/642/run_by_run/"),
        "thermal_model": Path("trains/pythia/641/run_by_run/"),
    }
    loading_data_rename_prefix = {
        "pp": {"data": "data"},
        # It has a separate function, so this isn't super meaningful.
        # However, it could be if we just wanted to look at one prefix, since we could
        # use the data loading function.
        # "pythia": {"data": "det_level", "true": "part_level"},
        "PbPb": {"data": "data"},
    }
    convert_data_format_prefixes = {
        "pp": {"data": "data"},
        # The loading data rename prefix won't apply any mapping for pythia,
        # so we have to handle it here.
        "pythia": {"det_level": "data", "part_level": "true"},
        "PbPb": {"data": "data"},
        "thermal_model": {"hybrid": "hybrid", "det_level": "det_level", "part_level": "true"},
    }

    # Job execution parameters
    task_name = "hardest_kt_mammoth"
    tasks_to_execute = [
        # "calculate_data_skim"
        # "calculate_embedding_skim",
        "calculate_thermal_model_skim",
    ]

    # Job execution configuration
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    # n_cores_to_allocate = 120
    # walltime = "1:59:00"
    n_cores_to_allocate = 80
    walltime = "24:00:00"
    #n_cores_to_allocate = 2
    #n_cores_to_allocate = 10

    # Validation
    # Collision system
    _possible_collision_systems = [
        "pp",
        "pythia",
        "PbPb",
        "embedPythia",
        "thermal_model",
    ]
    if not set(collision_systems_to_process).issubset(_possible_collision_systems):
        raise ValueError(f"Invalid collisions system(s) to process. Provided: {collision_systems_to_process}")

    # Basic setup: logging and parsl.
    # NOTE: Parsl's logger setup is broken, so we have to set it up before starting logging. Otherwise,
    #       it's super verbose and a huge pain to turn off. Note that by passing on the storage messages,
    #       we don't actually lose any info.
    config, facility_config, stored_messages = job_utils.config(
        facility="ORNL_b587_long",
        # facility="ORNL_b587_short",
        task_config=task_config,
        n_tasks=n_cores_to_allocate,
        walltime=walltime,
        enable_monitoring=True,
    )
    # Keep track of the dfk to keep parsl alive
    dfk = helpers.setup_logging_and_parsl(
        parsl_config=config,
        level=logging.INFO,
        stored_messages=stored_messages,
    )

    all_results = []
    for collision_system in collision_systems_to_process:
        # Collision system dependent
        scale_factors_dataset = "LHC18b8_pythia_R04_2520" if collision_system == "pythia" else ""

        # Setup tasks
        system_results = []
        if "calculate_data_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_data_skim(
                    collision_system=collision_system,
                    event_activity=event_activity,
                    min_jet_pt=min_jet_pt[collision_system],  # type: ignore
                    jet_R_values=jet_R_values,
                    iterative_splittings=iterative_splittings,
                    loading_data_rename_prefix=loading_data_rename_prefix.get(collision_system, {}),
                    convert_data_format_prefixes=convert_data_format_prefixes[collision_system],
                    scale_factors_dataset=scale_factors_dataset,
                    input_path=input_paths[collision_system],
                )
            )
        if "calculate_thermal_model_skim" in tasks_to_execute:
            if event_activity == "central":
                scale_factors_dataset = "LHC20g4_embedded_into_LHC18qr_central_R02_6982_7001"
            elif event_activity == "semi_central":
                scale_factors_dataset = "LHC20g4_embedded_into_LHC18qr_semi_central_R02_6932_6951"
            else:
                raise RuntimeError(f"Dunno what to do with {event_activity} for the scale factors dataset. Check this...")

            system_results.extend(
                setup_calculate_thermal_model_skim(
                    collision_system=collision_system,
                    event_activity=event_activity,
                    min_jet_pt=min_jet_pt[collision_system],  # type: ignore
                    jet_R_values=jet_R_values,
                    iterative_splittings=iterative_splittings,
                    convert_data_format_prefixes=convert_data_format_prefixes[collision_system],
                    scale_factors_dataset=scale_factors_dataset,
                    input_path=input_paths[collision_system],
                    n_repeat_file=5,
                )
            )

        all_results.extend(system_results)
        logger.info(f"Accumulated {len(system_results)} futures for {collision_system}")

    logger.info(f"Accumulated {len(all_results)} total futures")

    # Process the futures, showing processing progress
    # Since it returns the results, we can actually use this to accumulate results.
    gen_results = job_utils.provide_results_as_completed(all_results, running_with_parsl=True)

    # In order to support writing histograms from multiple systems, we need to index the output histograms
    # by the collision system + centrality.
    output_hists: Dict[str, Dict[Any, Any]] = {k: {} for k in collision_systems_to_process}
    with Progress(console=helpers.rich_console, refresh_per_second=1, speed_estimate_period=300) as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        # for a in all_results:
        for result in gen_results:
            # r = a.result()
            logger.info(f"result: {result[:2]}")
            if result[0] and len(result) == 4 and isinstance(result[3], dict):
                k = result[2]
                logger.info(f"Found result for key {k}")
                output_hists[k] = job_utils.merge_results(output_hists[k], result[3])
            logger.info(f"output_hists: {output_hists}")
            progress.update(track_results, advance=1)

    # Save hists to uproot
    for system, hists in output_hists.items():
        if hists:
            import uproot

            split_system_name = system.split("_")
            # Either "pp" or "PbPb"
            collision_system = split_system_name[0]
            # Additional label for centrality when appropriate
            # NOTE: If the list is of length 1, it will be empty
            file_label = "_".join(split_system_name[1:])
            if file_label:
                file_label = f"_{file_label}"

            output_hist_filename = Path("output") / collision_system / f"hardest_kt_{file_label}.root"
            output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing output_hists to {output_hist_filename} for system {system}")
            with uproot.recreate(output_hist_filename) as f:
                helpers.write_hists_to_file(hists=hists, f=f)

    # As far as I can tell, jobs will start executing as soon as they can, regardless of
    # asking for the result. By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns=locals())

    # In case we close IPython early, wait for all apps to complete
    # Also allows for a summary at the end.
    # By taking only the first two, it just tells use the status and a quick message.
    # Otherwise, we can overwhelm with trying to print large objects
    res = [r.result()[:2] for r in all_results]
    logger.info(res)


if __name__ == "__main__":
    run()
