"""Run mammoth skimming and analysis tasks via parsl

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import IPython
from mammoth import helpers, job_utils
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from rich.progress import Progress


logger = logging.getLogger(__name__)


@python_app  # type: ignore
def run_analyze_data(
    collision_system: str,
    jet_R: float,
    iterative_splittings: bool,
    min_jet_pt: float,
    event_activity: str = "",
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    import traceback
    from pathlib import Path

    from jet_substructure.analysis import track_skim_adapter

    try:
        track_skim_adapter.hardest_kt_data(
            collision_system=collision_system,
            event_activity=event_activity,
            input_filename=Path(inputs[0].filepath),
            jet_R=jet_R,
            iterative_splittings=iterative_splittings,
            min_jet_pt=min_jet_pt,
            output_filename=Path(outputs[0].filepath),
        )
        result = True, f"success for {collision_system}, R={jet_R}, {inputs[0].filepath}"
    except Exception:
        result = (
            False,
            f"failure for {collision_system}, R={jet_R}, {inputs[0].filepath} with: \n{traceback.format_exc()}",
        )
    return result


def setup_calculate_data_skim(
    collision_system: str,
    min_jet_pt: float,
    jet_R_values: Sequence[float],
    iterative_splittings: bool,
    input_path: Path,
    event_activity: str = "",
) -> List[AppFuture]:
    """Analyze hardest kt data"""
    input_files = sorted(input_path.glob("*/*/*.root"))

    # TEMP for testing
    input_files = input_files[:2]
    # ENDTEMP

    logger.info(input_files)

    results = []
    for input_filename in input_files:
        logger.info(f"Adding {input_filename} for analysis")

        # The input_file is in trains/collision_system/train_number/run_by_run/period/run_number/filename.root
        # So to get the train directory, we need to take the parent 4 times.
        train_directory = input_filename.parent.parent.parent.parent
        run_dir = input_filename.parent.name
        iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
        for jet_R in jet_R_values:
            # Setup file I/O
            output_dir = train_directory / "skim" / f"R{round(jet_R * 10):02}" / run_dir
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = output_dir / f"{input_filename.stem}_{iterative_splittings_label}_splittings.root"
            results.append(
                run_analyze_data(
                    collision_system=collision_system,
                    event_activity=event_activity,
                    jet_R=jet_R,
                    min_jet_pt=min_jet_pt,
                    iterative_splittings=iterative_splittings,
                    inputs=[File(str(input_filename))],
                    outputs=[File(str(output_filename))],
                )
            )

    return results


def run() -> None:
    # Basic setup
    iterative_splittings = True
    jet_R_values = [0.4]
    min_jet_pt = {
        "pp": 5,
        "pythia": 5,
        "embedPythia": 20,
        "PbPb": 20,
    }
    input_paths = {
        # Need to glob in the task
        "PbPb": Path("trains/PbPb/642/run_by_run/"),
    }
    collision_systems_to_process = ["PbPb"]
    event_activity = "central"

    # Job execution parameters
    task_name = "hardest_kt_mammoth"
    tasks_to_execute = [
        "calculate_data_skim"
        # "calculate_embedding_skim",
        # "calculate_thermal_model_skim",
    ]

    # Job execution configuration
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    # n_cores_to_allocate = 64
    # n_cores_to_allocate = 21
    n_cores_to_allocate = 2
    walltime = "24:00:00"

    # Validation
    # Collision system
    _possible_collision_systems = [
        "pp",
        "pythia",
        "PbPb",
        "embedPythia",
        "thermalModel",
    ]
    if not set(collision_systems_to_process).issubset(_possible_collision_systems):
        raise ValueError(f"Invalid collisions system(s) to process. Provided: {collision_systems_to_process}")

    # Basic setup: logging and parsl.
    # NOTE: Parsl's logger setup is broken, so we have to set it up before starting logging. Otherwise,
    #       it's super verbose and a huge pain to turn off. Note that by passing on the storage messages,
    #       we don't actually lose any info.
    config, facility_config, stored_messages = job_utils.config(
        facility="ORNL_b587_long",
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
        # Setup tasks
        system_results = []
        if "calculate_data_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_data_skim(
                    collision_system=collision_system,
                    event_activity=event_activity,
                    min_jet_pt=min_jet_pt[collision_system],
                    jet_R_values=jet_R_values,
                    iterative_splittings=iterative_splittings,
                    input_path=input_paths[collision_system],
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
    with Progress(console=helpers.rich_console, refresh_per_second=1) as progress:
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
