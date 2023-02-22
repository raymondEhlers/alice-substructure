#!/usr/bin/env python3

"""" Submit analysis using parsl.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import copy
import functools
import logging
import re
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableSequence, Optional, Sequence, Tuple, Union

import attrs
import dask.distributed
import IPython
import numpy as np
import parsl
import uproot
from mammoth.framework.analysis import objects as analysis_objects
from mammoth.framework.analysis import jet_substructure as analysis_jet_substructure
from mammoth.framework import utils as mammoth_utils
from mammoth import helpers
from pachyderm import yaml
from parsl.data_provider.files import File

from mammoth import job_utils
from mammoth.job_utils import python_app
from mammoth.alice import job_utils as alice_job_utils

from jet_substructure.base import helpers as jsub_helpers
from jet_substructure.base import unfolding as unfolding_base


logger = logging.getLogger(__name__)


def read_full_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read full YAML configuration file.

    Args:
        config_path: Path to the configuration file. Default: "config/new_config.yaml".
    Returns:
        Full YAML configuration. Requires some interpretation.
    """
    if config_path is None:
        config_path = Path("config/new_config.yaml")

    y = yaml.yaml()
    with open(config_path, "r") as f:
        full_config: Dict[str, Any] = y.load(f)

    return full_config


def read_dataset_config(base_dataset_name: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read collision system configuration from YAML file.

    The collision system specification is defined in the YAML file.

    Args:
        base_dataset_name: Name of the base dataset.
        collision_system: Name of the collision system.
        config_path: Path to the configuration file.
    Returns:
        The collision system configuration.
    """
    full_config = read_full_config(config_path=config_path)
    config: Dict[str, Any] = full_config["analysis_configurations"][base_dataset_name]

    return config


def read_extracted_scale_factors(
    collision_system: str,
    dataset_name: str,
) -> Dict[int, float]:
    """Read extracted scale factors.

    Args:
        collision_system: Name of the collision system.
        dataset_name: Name of the dataset.

    Returns:
        Normalized scaled factors
    """
    return analysis_objects.read_extracted_scale_factors(
        path=Path(f"trains/{collision_system}/{dataset_name}/scale_factors.yaml")
    )


@python_app
def _repair_root_files(
    tree_name: str, n_cores: int, inputs: Sequence[File] = [], outputs: Sequence[File] = []
) -> Dict[Path, List[Path]]:
    """ Repair ROOT files app. """
    from pathlib import Path

    from jet_substructure.base import helpers as jsub_helpers

    try:
        res = jsub_helpers.split_tree(
            filenames=[Path(inputs[0].filepath)],
            tree_name=tree_name,
            number_of_chunks=1,
            n_cores=n_cores,
        )
    except SystemError as e:
        # ROOT fails this way sometimes. No clear reason, and it's too bad, but so it goes.
        res = {Path(inputs[0].filepath): f"Failed due to ROOT internal issue...: {e}. Basically nothing to be done."}  # type: ignore[dict-item]
        logger.warning(res)
    return res


def setup_repair_root_files(
    n_cores_per_job: int,
    dataset_config: Mapping[str, Any],
    selected_train_numbers: Optional[Sequence[int]] = None,
) -> List[Future[Dict[Path, List[Path]]]]:
    """Repair ROOT files.

    Settings will be taken out of the configuration file.

    Args:
        n_cores_per_job: Number of cores to use per job.
        dataset_config: Dataset configuration.
        selected_train_numbers: Use only a selection of train numbers. Default: None. All train numbers are
            taken from the config.
    Returns:
        List of `AppFuture` created when defining the jobs.
    """
    # Validation
    if selected_train_numbers is None:
        selected_train_numbers = []

    # Setup
    logger.info(f"Repairing files from dataset {dataset_config['name']}")
    tree_name = dataset_config["tree_name"]
    # Determine filenames. This is a bit involved.
    filenames = dataset_config["files"]
    # Filter out already repaired files
    # Specifically, we usually specify the repaired files in the config, but that's
    # not meaningful here. So we remove the "repaired" from the name, and then take those files.
    # NOTE: This is susceptible to issues if "repaired." is in the path, but I think that's unlikely.
    filenames = sorted([Path(str(f).replace("repaired.", "")) for f in filenames])
    # Once we've initially filtered out the repaired filenames, we need to expand them
    filenames = mammoth_utils.expand_wildcards_in_filenames(filenames)
    # After the wild card expansions, we need to do another filter for possible repaired filenames.
    # NOTE: It's important that we take a set because if the dir already has both, we don't
    #       want to try to add files twice.
    # NOTE: This is susceptible to issues if "repaired." is in the path, but I think that's unlikely.
    filenames = sorted(set([Path(str(f).replace("repaired.", "")) for f in filenames]))
    # And then filter by selected trains if necessary
    if selected_train_numbers:
        filenames = [f for f in filenames if int(f.parent.name) in selected_train_numbers]
    logger.debug(f"Repairing filenames: {filenames}")

    results = []
    for filename in filenames:
        # Setup file IO
        parsl_input_file = File(str(filename))
        output_filename = filename.with_name(f"{filename.stem}.repaired.root")
        parsl_output_file = File(str(output_filename))

        results.append(
            _repair_root_files(
                tree_name=tree_name,
                n_cores=n_cores_per_job,
                inputs=[parsl_input_file],
                outputs=[parsl_output_file],
            )
        )

    return results


def _determine_number_of_entries_per_file(filenames: Sequence[Path], tree_name: str) -> Dict[Path, int]:
    """Retrieve the number of tree entries per ROOT file.

    Args:
        filenames: Filenames containing the trees of interest.
        tree_name: Name of the tree.
    Returns:
        Map from the filename to the number of entries.
    """
    number_of_entries_per_file: Dict[Path, int] = {}

    for filename in filenames:
        with uproot.open(filename) as f:
            logger.debug(filename)
            number_of_entries_per_file[filename] = f[tree_name].num_entries

    return number_of_entries_per_file


def _number_of_entries_per_file(
    input_filenames: Sequence[Union[Path, str]],
    tree_name: str,
    collision_system: str,
    dataset_name: str,
    recreate: bool = False,
) -> Dict[Path, int]:
    """Determine number of entries per file.

    Args:
        input_filenames: Filenames to use in the number of entries determination.
        tree_name: Name of the tree.
        collision_system: Collision system.
        dataset_name: Identifier for the dataset. Used to cache the number of entries per file.
        recreate: Force recreation of the number of entries per file for a dataset, skipping
            over the cache. Default: False.
    Returns:
        Mapping between file and number of entries in the file.
    """
    # Validation
    filenames = mammoth_utils.expand_wildcards_in_filenames([Path(f) for f in input_filenames])

    # Setup
    y = yaml.yaml()
    number_of_entries_file = Path(f"trains/{collision_system}/{dataset_name}/entries_per_file.yaml")
    number_of_entries_file.parent.mkdir(parents=True, exist_ok=True)

    # We need the number of entries per file to be able to split up the jobs properly later.
    # If it doesn't exist, create it.
    if not number_of_entries_file.exists() or recreate:
        logger.info("Need to get entries from the input files.")
        number_of_entries_per_file = _determine_number_of_entries_per_file(filenames=filenames, tree_name=tree_name)
        with open(number_of_entries_file, "w") as f:
            # Explicit iteration because we need to convert from Path to str.
            y.dump({str(k): v for k, v in number_of_entries_per_file.items()}, f)

    # Now we know that it exists, we can grab it.
    with open(number_of_entries_file, "r") as f:
        res = y.load(f)
        number_of_entries_per_file = {Path(k): v for k, v in res.items()}

    return number_of_entries_per_file


def _distribute_entries_to_jobs(
    number_of_entries_per_file: Mapping[Path, int], entries_per_job: int
) -> Dict[Path, List[Tuple[int, int]]]:
    """Distribute a specific number of entries to each job.

    We distribute based on the number of entries in a given file, so if a file doesn't contain enough for a full job,
    we assign fewer events. We lose some efficiency, but it's much simpler.

    Args:
        number_of_entries_per_file: Mapping between file and number of entries in the file.
        entries_per_job: Number of entries to be assigned per job.
    Returns:
        Mapping from file to list of entry ranges (low, high).
    """
    job_info = {}

    for filename, number_of_entries in number_of_entries_per_file.items():
        splits = []
        start = 0
        continue_iterating = True
        while continue_iterating:
            end = start + entries_per_job
            # Ensure that we never ask for more entries than are in the file.
            if start + entries_per_job > number_of_entries:
                end = number_of_entries
                continue_iterating = False
            # Store the start and stop for convenience.
            splits.append((start, end))
            # Move up to the next iteration.
            start = end

        job_info[filename] = splits

    return job_info


@python_app
def _number_of_entries_per_file_app(
    tree_name: str,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> int:
    from pathlib import Path

    import uproot

    filename = Path(inputs[0].filepath)
    with uproot.open(filename) as f:
        logger.debug(filename)
        number_of_entries = f[tree_name].num_entries

    return number_of_entries  # type: ignore[no-any-return]


@python_app
def _write_number_of_entries_per_file_cache(
    number_of_entries_per_file: Dict[str, int],
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> bool:
    """Write the number of entries per file to a cache.

    Note:
        Since the values that are inputs to this function are already evaluated,
        this doesn't actually need to be an app. We could just write the values.
        However, since it's already an app, we just leave it.

    """
    from pathlib import Path

    from pachyderm import yaml

    # Setup
    output_filename = Path(outputs[0].filepath)
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    y = yaml.yaml()

    # And then actually write the cache.
    with open(output_filename, "w") as f:
        # Explicit iteration because we need to ask for results.
        # y.dump({k: v for k, v in number_of_entries_per_file.items()}, f)
        # y.dump({k: v.result() for k, v in number_of_entries_per_file.items()}, f)
        y.dump(number_of_entries_per_file, f)

    return True


@python_app
def _entries_to_ranges_for_jobs(
    number_of_entries: int,
    entries_per_job: int,
    job_framework: job_utils.JobFramework,
) -> List[Tuple[int, int]]:
    """Determine the event range for a job given a total number of entries.

    Args:
        number_of_entries: Number of entries in the file.
        entries_per_job: Number of entries to process in a single job.

    Returns:
        List of (start entry, end entry) values.
    """
    splits = []
    start = 0
    continue_iterating = True
    while continue_iterating:
        end = start + entries_per_job
        # Ensure that we never ask for more entries than are in the file.
        if start + entries_per_job > number_of_entries:
            end = number_of_entries
            continue_iterating = False
        # Store the start and stop for convenience.
        splits.append((start, end))
        # Move up to the next iteration.
        start = end

    return splits


@python_app
def _convert_to_parquet(
    tree_name: str,
    prefixes: Sequence[str],
    branches: Sequence[str],
    prefix_branches: Sequence[str],
    job_framework: job_utils.JobFramework,
    event_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> Tuple[bool, Path]:
    """ Convert to parquet app. """
    from pathlib import Path

    from jet_substructure.base import new_methods

    res = new_methods.convert_tree_to_parquet(
        filename=Path(inputs[0].filepath),
        tree_name=tree_name,
        prefixes=prefixes,
        branches=branches,
        prefix_branches=prefix_branches,
        entries=event_range,
        output_filename=Path(outputs[0].filepath),
    )
    logger.debug(outputs)
    return res


def setup_convert_to_parquet(
    collision_system: str,
    entries_per_job: int,
    dataset_config: Mapping[str, Any],
    job_framework: job_utils.JobFramework,
    input_results: Optional[MutableSequence[Future[Any]]] = None,
) -> Tuple[List[Future[Any]], List[Future[Any]]]:
    """Setup convert_to_parquet app for execution with parsl.

    This is a bit more involved than many of the other tasks because it requires some
    intermediate information to fully process. Consequently, it consists of a number
    of apps.

    Args:
        collision_system: Name of collision system.
        entries_per_job: Number of events to process in each job.
        dataset_config: Dataset configuration.
        input_results: AppFuture from a previous step.
    Returns:
        List of `AppFuture` created when defining the jobs.
    """
    # Setup
    results: List[Future[Any]] = []

    # Determine filenames
    if input_results is None or job_framework != job_utils.JobFramework.parsl:
        input_filenames = mammoth_utils.expand_wildcards_in_filenames([Path(f) for f in dataset_config["files"]])
        input_files = [File(str(filename)) for filename in input_filenames]
    else:
        # Only works for parsl, since we explicitly need an app future
        input_files = [r.outputs[0] for r in input_results]  # type: ignore[attr-defined]

    # Attempt to load the number of entries per file cache
    number_of_entries_filename = Path(f"trains/{collision_system}/{dataset_config['name']}/entries_per_file.yaml")
    number_of_entries_per_file = {}
    if number_of_entries_filename.exists():
        logger.info("Loading number of entries per file cache")
        # Now we know that it exists, we can grab it.
        y = yaml.yaml()
        with open(number_of_entries_filename, "r") as f:
            res = y.load(f)
            # number_of_entries_per_file = {Path(k): v for k, v in res.items()}
            number_of_entries_per_file = {k: v for k, v in res.items()}

    # logger.info(f"Input files: {input_files}")
    # logger.info(f"number_of_entries_per_file: {number_of_entries_per_file}")

    # Keeping track of individual job futures.
    job_ranges_results = []
    # NOTE: This will be a proxy for if we have to run jobs to determine the number of
    #       entries per file. If there are entries in this dict, then we need to calculate.
    #       If not, we're using the cache.
    number_of_entries_per_file_results = {}
    for input_file in input_files:
        if not number_of_entries_per_file:
            number_of_entries_per_file_result = _number_of_entries_per_file_app(
                tree_name=dataset_config["tree_name"],
                inputs=[input_file],
            )
            # Store the output in our overall results.
            results.append(number_of_entries_per_file_result)
            # Store this output so we can cache this result.
            number_of_entries_per_file_results[str(input_file.filepath)] = number_of_entries_per_file_result
        else:
            # Utilize the cache.
            number_of_entries_per_file_result = number_of_entries_per_file[input_file.filepath]

        _job_ranges_result = _entries_to_ranges_for_jobs(
            number_of_entries=number_of_entries_per_file_result,  # type: ignore[arg-type]
            entries_per_job=entries_per_job,
            job_framework=job_framework,
        )
        # Store the output in our overall results.
        results.append(_job_ranges_result)
        # And then store just the job ranges results
        job_ranges_results.append((input_file, _job_ranges_result))

    # logger.info(f"Partially done results: {results}")
    # logger.info(f"job_ranges_results: {job_ranges_results}")
    logger.info("About to determine job ranges for conversion to parquet.")
    if number_of_entries_per_file_results:
        logger.warning("This is going to hang for a while during the job range calculations...")

    # Write cache if needed.
    # NOTE: Putting this here has two consequences:
    #       1. It will break the dependency tree that shows up in the monitoring because we ask
    #          for the results here.
    #       2. By just asking for the result, it seems that it won't be considered a dependency
    #          of the previous tasks in the monitoring.
    # NOTE: Both of these consequences cause no issues with the real dependency tree. The apps
    #       will still be executed in the proper order.
    if number_of_entries_per_file_results:
        logger.info("Writing the number of entries per file cache.")
        results.append(
            _write_number_of_entries_per_file_cache(
                # It really, really seems like we should just be able to pass the results here,
                # but for some reason, it crashes with `TypeError: can't pickle _thread.RLock objects`.
                # It's super unclear, because even a simplified example seems to fail. But this
                # works, and since it's often already been calculated, we don't really lose any time with
                # this approach. However, it's frustrating that it doesn't work as expected...
                number_of_entries_per_file={k: v.result() for k, v in number_of_entries_per_file_results.items()},
                # number_of_entries_per_file=number_of_entries_per_file_results,
                job_framework=job_framework,
                outputs=[File(str(number_of_entries_filename))],
            )
        )

    parquet_results = []
    for input_file, job_ranges_result in job_ranges_results:
        # We need the outputs from the job ranges, so we have to evaluate the jobs now.
        # This leads to kind of an awkward hanging if we need to wait to calculate the
        # number of entries per file, but there's nothing else to be done.
        # NOTE: We evaluate in this order instead of trying to evaluate all number of entries
        #       per file because this allows each the processing of each job to be independent.
        # NOTE: Unfortunately, this breaks the dependency chain from job ranges to convert_to_parquet
        #       in the DAG display. However, it seems to internally calculate the dependencies
        #       correctly, so it's fine.
        # logger.info(f"job_ranges_result: {job_ranges_result.result()}")
        for i, event_range in enumerate(job_ranges_result.result()):
            # Setup file IO
            output_filename = Path(input_file.filepath)
            # Path becomes ../parquet/events_per_job_.../filename.00.parquet
            output_filename = (
                output_filename.parent / "parquet" / f"events_per_job_{entries_per_job}" / output_filename.name
            )
            output_filename = output_filename.with_suffix(f".{i:02}.parquet")
            output_file = File(str(output_filename))

            parquet_results.append(
                _convert_to_parquet(
                    tree_name=dataset_config["tree_name"],
                    prefixes=list(dataset_config["prefixes"].values()),
                    branches=dataset_config["branches"],
                    prefix_branches=dataset_config["prefix_branches"],
                    event_range=event_range,
                    job_framework=job_framework,
                    inputs=[input_file],
                    outputs=[output_file],
                )
            )
    # Store with the rest of the results
    results.extend(parquet_results)

    # logger.info(f"Almost done: {results}")
    # logger.info(f"number_of_entries_per_file_results: {number_of_entries_per_file_results}")

    return results, parquet_results


def _determine_pythia_input_files_per_pt_hard_bin(
    dataset_config: Mapping[str, Any],
) -> Dict[int, List[Path]]:
    input_files_per_pt_hard_bin: Dict[int, List[Path]] = {}
    all_filenames = []
    for filename_base in dataset_config["files"]:
        filename_base = Path(filename_base)
        all_filenames.extend([Path(f) for f in mammoth_utils.expand_wildcards_in_filenames([filename_base])])

    # Sort by pt hard bins
    # We're in Run 2, so pretty safe to assume 20 pt hard bins
    input_files_per_pt_hard_bin
    for pt_hard_bin in range(1, 21):
        input_files_per_pt_hard_bin[pt_hard_bin] = [f for f in all_filenames if f"{pt_hard_bin:02g}" in str(f.name)]

    logger.info(f"input_files_per_pt_hard_bin: {input_files_per_pt_hard_bin}")

    return input_files_per_pt_hard_bin


def _determine_embedding_input_files_per_pt_hard_bin(
    dataset_config: Mapping[str, Any],
    selected_train_numbers: Optional[Sequence[int]] = None,
) -> Dict[int, List[Path]]:
    input_files_per_pt_hard_bin = {}
    for filename_base in dataset_config["files"]:
        filename_base = Path(filename_base)

        # Grab the pt hard bin to use as the key.
        y = yaml.yaml()
        with open(filename_base.parent / "config.yaml", "r") as f:
            train_config = y.load(f)
        train_number = train_config["number"]
        pt_hard_bin = train_config["pt_hard_bin"]

        # Validation for the train_number
        assert train_number == int(filename_base.parent.name)
        if selected_train_numbers and train_number not in selected_train_numbers:
            logger.debug(f"Skipping train number {train_number}")
            continue

        # Expand the filenames
        input_files_per_pt_hard_bin[pt_hard_bin] = [
            Path(f) for f in mammoth_utils.expand_wildcards_in_filenames([filename_base])
        ]

    return input_files_per_pt_hard_bin


@python_app
def _extract_scale_factors_from_hists(
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> analysis_objects.ScaleFactor:
    from pathlib import Path

    from mammoth.framework.analysis.objects import ScaleFactor

    from jet_substructure.cpp import scale_factors as sf

    res = ScaleFactor.from_hists(
        *sf.scale_factor_ROOT(filenames=[Path(i.filepath) for i in inputs])
    )
    return res


def setup_extract_scale_factors(
    collision_system: str,
    dataset_config: Mapping[str, Any],
    job_framework: job_utils.JobFramework,
    selected_train_numbers: Optional[Sequence[int]] = None,
) -> Dict[int, Future[analysis_objects.ScaleFactor]]:
    """Extract scale factors from embedding or pythia hists.

    Note:
        This is surprisingly fast.
    """
    # Validation
    if collision_system not in ["pythia", "embedPythia", "embed_pythia", "embed_thermal_model"]:
        raise ValueError(f"Invalid collision system for extracting scale factors: {collision_system}")

    # Setup
    scale_factors = {}
    logger.info("Determining input files for extracting scale factors.")
    if "embed" in collision_system:
        input_files_per_pt_hard_bin = _determine_embedding_input_files_per_pt_hard_bin(
            dataset_config=dataset_config, selected_train_numbers=selected_train_numbers
        )
    elif collision_system == "pythia":
        input_files_per_pt_hard_bin = _determine_pythia_input_files_per_pt_hard_bin(
            dataset_config=dataset_config,
        )
    else:
        raise ValueError(f"Invalid collision system for extracting scale factors: {collision_system}")

    for pt_hard_bin, input_files in input_files_per_pt_hard_bin.items():
        logger.debug(f"pt_hard_bin: {pt_hard_bin}, filenames: {input_files}")
        if input_files:
            scale_factors[pt_hard_bin] = _extract_scale_factors_from_hists(
                job_framework=job_framework,
                inputs=[File(str(fname)) for fname in input_files],
            )

    return scale_factors


@python_app
def _write_scale_factors_to_yaml(
    scale_factors: Mapping[int, analysis_objects.ScaleFactor],
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> bool:
    from pathlib import Path

    from pachyderm import yaml

    from mammoth.framework.analysis import objects as analysis_objects

    # Write them to YAML for later.
    y = yaml.yaml(classes_to_register=[analysis_objects.ScaleFactor])
    output_dir = Path(outputs[0].filepath)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(output_dir, "w") as f:
        y.dump(scale_factors, f)

    return True


@python_app
def _write_cross_check_task_scale_factor_trees(
    scale_factor: float,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> bool:
    from pathlib import Path

    from jet_substructure.cpp import scale_factors as sf

    res = sf.create_scale_factor_tree_for_cross_check_task_output(
        filename=Path(inputs[0].filepath),
        scale_factor=scale_factor,
    )
    return res


def setup_write_scale_factors(
    collision_system: str,
    dataset_config: Mapping[str, Any],
    scale_factors: Mapping[int, Future[analysis_objects.ScaleFactor]],
    job_framework: job_utils.JobFramework,
    selected_train_numbers: Optional[Sequence[int]] = None,
) -> Future[bool]:
    """Write scale factors to YAML and to trees if necessary."""
    # First, we write to YAML.
    # We want to do this regardless of potentially writing the scale factor trees.
    logger.info("Writing scale factors to YAML. Jobs are executing, so this will take a minute...")
    output_filename = Path(f"trains/{collision_system}/{dataset_config['name']}/scale_factors.yaml")
    parsl_output_file = File(str(output_filename))
    # TODO: I'm guessing passing this is a problem because it's a class that's imported in an app, and then
    #       we're trying to pass the result into another app. I think we can go one direction or the other,
    #       but not both. So we just take the result.
    yaml_result = _write_scale_factors_to_yaml(
        scale_factors={k: v.result() for k, v in scale_factors.items()},
        job_framework=job_framework,
        outputs=[parsl_output_file],
    )

    # Setup
    cross_check_task = dataset_config.get("cross_check_task", False)
    if not cross_check_task:
        logger.info(
            f"Dataset {dataset_config['name']} is not a cross check task, so skipping writing the scale factor tree."
        )
        return yaml_result

    return yaml_result


@python_app
def _extract_pt_hard_spectra(
    scale_factors: Mapping[int, float],
    offsets: Mapping[int, int],
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> bool:
    from pathlib import Path

    from jet_substructure.cpp import scale_factors as sf

    # Convert back from parsl inputs
    offsets_values = list(offsets.values())
    filenames = {
        pt_hard_bin: [
            Path(f.filepath) for f in inputs[sum(offsets_values[:i]) : sum(offsets_values[: i + 1])]  # noqa: E203
        ]
        for i, pt_hard_bin in enumerate(offsets)
    }
    # filenames = {
    #    pt_hard_bin: {Path(f.filepath) for pt_hard_bin, f in enumerate(inputs, start=1)}
    # }

    res = sf.pt_hard_spectra_from_hists(
        filenames=filenames,
        scale_factors=scale_factors,
        output_filename=Path(outputs[0].filepath),
    )
    return res


def setup_extract_embedding_pt_hard_spectra(
    collision_system: str,
    dataset_config: Mapping[str, Any],
    job_framework: job_utils.JobFramework,
    input_results: MutableSequence[Future[bool]],
    selected_train_numbers: Optional[Sequence[int]] = None,
) -> Future[bool]:
    # Validation
    if collision_system not in ["pythia", "embedPythia", "embed_pythia", "embed_thermal_model"]:
        raise ValueError(f"Invalid collision system for extracting scale factors: {collision_system}")

    # Input files
    logger.info("Determining input files")
    if "embed" in collision_system:
        input_files_per_pt_hard_bin = _determine_embedding_input_files_per_pt_hard_bin(
            dataset_config=dataset_config, selected_train_numbers=selected_train_numbers
        )
    elif collision_system == "pythia":
        input_files_per_pt_hard_bin = _determine_pythia_input_files_per_pt_hard_bin(
            dataset_config=dataset_config,
        )
    else:
        raise ValueError(f"Invalid collision system for extracting scale factors: {collision_system}")

    # Need a hard dependency on the writing of the yaml output, so we ask for the result here.
    # We don't actually care about the result, but it avoids a race condition.
    _ = input_results[0].result()
    # Must read the scale factors from file to get the properly scaled values.
    scale_factors = read_extracted_scale_factors(collision_system=collision_system, dataset_name=dataset_config["name"])

    # Convert inputs to Parsl files.
    # Needs to be a list, so flatten them, and then unflatten in the App.
    parsl_files = []
    offsets = {}
    for pt_hard_bin, list_of_files in input_files_per_pt_hard_bin.items():
        converted_filenames = [File(str(f)) for f in list_of_files]
        offsets[pt_hard_bin] = len(converted_filenames)
        parsl_files.extend(converted_filenames)
    # Add the dependency. We won't actually open the file in the task, but this will provide explicit dependence.
    if job_framework == job_utils.JobFramework.parsl:
        parsl_files.extend([i.outputs[0] for i in input_results])  # type: ignore[attr-defined]
    else:
        # We lie about the typing here, but it's convenient
        parsl_files.extend(input_results)  # type: ignore[arg-type]

    output_filename = Path(f"trains/{collision_system}/{dataset_config['name']}/pt_hard_spectra.yaml")
    results = _extract_pt_hard_spectra(
        scale_factors=scale_factors,
        offsets=offsets,
        job_framework=job_framework,
        inputs=parsl_files,
        outputs=[File(str(output_filename))],
    )

    return results


@python_app
def _calculate_embedding_skim(
    dataset_config: Mapping[str, Any],
    train_directory: Path,
    iterative_splittings: bool,
    scale_factors: Mapping[int, float],
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> Tuple[bool, Path, str]:
    """ Calculate embedding skim app. """
    import traceback
    from pathlib import Path

    from jet_substructure.analysis import new_skim_to_flat_tree

    try:
        res = new_skim_to_flat_tree.calculate_embedding_skim(
            input_filename=Path(inputs[0].filepath),
            iterative_splittings=iterative_splittings,
            prefixes=dataset_config["prefixes"],
            scale_factors=scale_factors,
            train_directory=train_directory,
            jet_R=dataset_config["jet_R"],
            output_filename=Path(outputs[0].filepath),
        )
    except Exception as e:
        # Skip any problems for now
        logger.warning(e)
        # Match the expected format if the calculation succeeded
        res = (False, Path(inputs[0].filepath), traceback.format_exc())

    logger.debug(outputs)
    return res


def setup_calculate_embedding_skim(
    collision_system: str,
    entries_per_job: int,
    dataset_config: Mapping[str, Any],
    job_framework: job_utils.JobFramework,
    iterative_splittings: bool = True,
    selected_train_numbers: Optional[Sequence[int]] = None,
    input_results: Optional[MutableSequence[Future[Any]]] = None,
) -> List[Future[Tuple[bool, Path, str]]]:
    """Setup to calculate embedding skim.

    Args:
        collision_system: Collision system.
        entries_per_job: Number of entries per job.
        iterative_splittings: True if iterative splittings are selected rather than recursive splittings. Default: True.
        selected_train_numbers: Use only a selection of train numbers. Default: None. All train numbers are
            taken from the config.
        input_results: AppFuture from a previous step.
    Returns:
        List of `AppFuture` created when defining the jobs.
    """
    # Validation
    if selected_train_numbers is None:
        selected_train_numbers = []

    # Setup
    results = []
    scale_factors = read_extracted_scale_factors(collision_system=collision_system, dataset_name=dataset_config["name"])

    # If input files aren't passed, then we need to determine them ourselves.
    input_files = []
    if input_results is None or job_framework != job_utils.JobFramework.parsl:
        logger.info("Determining input files independently.")
        # First, determine the train directories so we can skip over some of them if requested.
        train_directories = set([Path(filename).parent for filename in dataset_config["files"]])
        for train_directory in sorted(train_directories):
            # Select train numbers.
            if selected_train_numbers and int(train_directory.name) not in selected_train_numbers:
                logger.debug(f"Skipping train number {train_directory.name}")
                continue
            logger.info(f"Processing train number {train_directory.name}")

            # Then iterate over the directories.
            for filename in Path(f"{train_directory}/parquet/events_per_job_{entries_per_job}/").glob("*.parquet"):
                input_files.append(File(str(filename)))
    else:
        # Only works for parsl, since we explicitly need an app future
        input_files = [r.outputs[0] for r in input_results]  # type: ignore[attr-defined]

    # Create the Apps.
    for parsl_input_file in input_files:
        # Setup
        # The input_file is in trains/collision_system/train_number/parquet/events_per_job_{entries_per_job}/filename.parquet
        # So to get the train directory, we need to take the parent 3 times.
        train_directory = Path(parsl_input_file.filepath).parent.parent.parent
        iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
        # Setup file I/O
        output_dir = train_directory / "skim"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = (
            output_dir / f"{Path(parsl_input_file.filepath).stem}_{iterative_splittings_label}_splittings.root"
        )
        parsl_output_file = File(str(output_filename))

        results.append(
            _calculate_embedding_skim(
                dataset_config=dataset_config,
                iterative_splittings=iterative_splittings,
                train_directory=train_directory,
                scale_factors=scale_factors,
                job_framework=job_framework,
                inputs=[parsl_input_file],
                outputs=[parsl_output_file],
                # stdout=parsl.AUTO_LOGNAME,
                # stderr=parsl.AUTO_LOGNAME,
            )
        )

    return results


@python_app
def _calculate_data_skim(
    collision_system: str,
    dataset_config: Mapping[str, Any],
    iterative_splittings: bool,
    scale_factors: Mapping[int, float],
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> Tuple[bool, Path, str]:
    """ Calculate data skim app. """
    import traceback
    from pathlib import Path

    from jet_substructure.analysis import new_skim_to_flat_tree

    try:
        res = new_skim_to_flat_tree.calculate_data_skim(
            input_filename=Path(inputs[0].filepath),
            collision_system=collision_system,
            iterative_splittings=iterative_splittings,
            prefixes=dataset_config["prefixes"],
            jet_R=dataset_config["jet_R"],
            output_filename=Path(outputs[0].filepath),
            scale_factors=scale_factors,
        )
    except Exception as e:
        # Skip any problems for now
        logger.warning(e)
        # Match the expected format if the calculation succeeded
        res = (False, Path(inputs[0].filepath), traceback.format_exc())

    logger.debug(outputs)
    return res


def setup_calculate_data_skim(
    collision_system: str,
    entries_per_job: int,
    dataset_config: Mapping[str, Any],
    job_framework: job_utils.JobFramework,
    iterative_splittings: bool = True,
    selected_train_numbers: Optional[Sequence[int]] = None,
    input_results: Optional[MutableSequence[Future[Any]]] = None,
) -> List[Future[Tuple[bool, Path, str]]]:
    """Setup to calculate data skim.

    Args:
        collision_system: Collision system.
        entries_per_job: Number of entries per job.
        iterative_splittings: True if iterative splittings are selected rather than recursive splittings. Default: True.
        selected_train_numbers: Use only a selection of train numbers. Default: None. All train numbers are
            taken from the config.
        input_results: AppFuture from a previous step.
    Returns:
        List of `AppFuture` created when defining the jobs.
    """
    # Validation
    if selected_train_numbers is None:
        selected_train_numbers = []

    # Setup
    results = []
    # Only meaningful for pythia.
    scale_factors = {}
    if collision_system == "pythia":
        scale_factors = read_extracted_scale_factors(
            collision_system=collision_system, dataset_name=dataset_config["name"]
        )

    # If input files aren't passed, then we need to determine them ourselves.
    input_files = []
    if input_results is None or job_framework != job_utils.JobFramework.parsl:
        logger.info("Determining input files independently.")
        # First, determine the train directories so we can skip over some of them if requested.
        train_directories = set([Path(filename).parent for filename in dataset_config["files"]])
        for train_directory in sorted(train_directories):
            # Select train numbers.
            if selected_train_numbers and int(train_directory.name) not in selected_train_numbers:
                logger.debug(f"Skipping train number {train_directory.name}")
                continue
            logger.info(f"Processing train number {train_directory.name}")

            # Then iterate over the directories.
            for filename in Path(f"{train_directory}/parquet/events_per_job_{entries_per_job}/").glob("*.parquet"):
                input_files.append(File(str(filename)))
    else:
        # Only works for parsl, since we explicitly need an app future
        input_files = [r.outputs[0] for r in input_results]  # type: ignore[attr-defined]

    # Create the Apps.
    for parsl_input_file in input_files:
        # Setup
        # The input_file is in trains/collision_system/train_number/parquet/events_per_job_{entries_per_job}/filename.parquet
        # So to get the train directory, we need to take the parent 3 times.
        train_directory = Path(parsl_input_file.filepath).parent.parent.parent
        iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
        # Setup file I/O
        output_dir = train_directory / "skim"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = (
            output_dir / f"{Path(parsl_input_file.filepath).stem}_{iterative_splittings_label}_splittings.root"
        )
        parsl_output_file = File(str(output_filename))

        results.append(
            _calculate_data_skim(
                dataset_config=dataset_config,
                collision_system=collision_system,
                iterative_splittings=iterative_splittings,
                scale_factors=scale_factors,
                job_framework=job_framework,
                inputs=[parsl_input_file],
                outputs=[parsl_output_file],
                # stdout=parsl.AUTO_LOGNAME,
                # stderr=parsl.AUTO_LOGNAME,
            )
        )

    return results


@python_app
def _calculate_cross_check_task_skim(
    dataset_config: Mapping[str, Any],
    grooming_method: str,
    scale_factor: float,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> bool:
    from pathlib import Path

    from jet_substructure.analysis import new_skim_to_flat_tree

    result = new_skim_to_flat_tree.skim_cross_check_task_to_uniform_output(
        input_filename=Path(inputs[0].filepath),
        grooming_method=grooming_method,
        input_tree_name=dataset_config["tree_name"],
        scale_factor=scale_factor,
        prefixes=dataset_config["prefixes"],
        output_filename=Path(outputs[0].filepath),
    )
    logger.debug(result)
    return result


def setup_calculate_cross_check_task_skim(
    collision_system: str,
    dataset_config: Mapping[str, Any],
    grooming_methods: Sequence[str],
    job_framework: job_utils.JobFramework,
    selected_train_numbers: Optional[Sequence[int]] = None,
    input_results: Optional[MutableSequence[Future[Any]]] = None,
    iterative_splittings: bool = True,
) -> List[Future[bool]]:
    # Validation
    if selected_train_numbers is None:
        selected_train_numbers = []

    # Setup
    results = []
    # Retrieve the scale factors.
    # Only possible input is the yaml scale factors, so we block on that result, so we need it immediately.
    if input_results:
        _ = input_results[0].result()
    scale_factors = read_extracted_scale_factors(collision_system=collision_system, dataset_name=dataset_config["name"])

    # File setup.
    logger.info("Determining input files independently.")
    # We'll always have to determine the input files ourselves.
    input_filenames = mammoth_utils.expand_wildcards_in_filenames([Path(f) for f in dataset_config["files"]])
    # Determine the train directories so we can skip over some of them if requested and map the scale factors.
    train_directories = set([Path(filename).parent for filename in dataset_config["files"]])

    # Create map from train directories to scale factors.
    train_number_to_pt_hard_bin = {}
    for train_directory in train_directories:
        y = yaml.yaml()
        with open(train_directory / "config.yaml", "r") as f:
            train_config = y.load(f)
        # Validation
        if train_config["number"] != int(train_directory.name):
            raise ValueError(
                f"Mismatch between train number in config ({train_config['number']}) and directory ({train_directory.name})."
            )
        train_number_to_pt_hard_bin[train_directory.name] = train_config["pt_hard_bin"]

    # Now, setup the Apps based on the input filenames
    for input_filename in input_filenames:
        # NOTE: Although we nominally iterate over grooming methods, cross check tasks should only ever have one.
        #       It's mostly to keep consistency with other tasks.
        for grooming_method in grooming_methods:
            train_directory = input_filename.parent
            # Select train numbers.
            if selected_train_numbers and int(train_directory.name) not in selected_train_numbers:
                logger.debug(f"Skipping input file {input_filename} due to skipped train number {train_directory.name}")
                continue

            # Setup
            # We keep this convention here even though the cross check task always outputs iterative splittings.
            iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
            # Setup file I/O
            parsl_input_file = File(str(input_filename))
            output_dir = train_directory / "skim"
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            # Since we're splitting in the app, we can't predict the splitting here. However,
            # we know that at least file 00 must exist so we require that (and perhaps others
            # will be created). Consequently, this isn't a super reliable dependency based on
            # files, and it would be better to depend on the return value (although it's not
            # meaningful in itself).
            output_filename = (
                output_dir / f"{Path(parsl_input_file.filepath).stem}.00_{iterative_splittings_label}_splittings.root"
            )
            parsl_output_file = File(str(output_filename))

            results.append(
                _calculate_cross_check_task_skim(
                    dataset_config=dataset_config,
                    grooming_method=grooming_method,
                    scale_factor=scale_factors[train_number_to_pt_hard_bin[train_directory.name]],
                    inputs=[parsl_input_file],
                    outputs=[parsl_output_file],
                )
            )

    return results


@python_app
def _root_data_frame(
    collision_system: str,
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,
    main_jet_pt_range: jsub_helpers.JetPtRange,
    n_cores: int,
    cross_check_task: bool,
    double_counting_cut_name: str,
    job_framework: job_utils.JobFramework,
    inputs: MutableSequence[File] = [],
    outputs: MutableSequence[File] = [],
) -> Tuple[bool, str]:
    """ROOT data frame app.

    We keep them separate even though they are quite similar so their app names will be different.
    Since they're so simple, this isn't a big sacrifice.
    """
    import random
    from pathlib import Path

    from jet_substructure.cpp import data_frame

    # Shuffle inputs
    # Randomize the input list so we don't always hit the same files at the same time.
    # Note: It randomizes in place.
    random.shuffle(inputs)

    res = data_frame.run(
        collision_system=collision_system,
        input_filenames=[Path(f.filepath) for f in inputs],
        tree_name=tree_name,
        prefixes=prefixes,
        grooming_method=grooming_method,
        jet_R=jet_R,
        main_jet_pt_range=main_jet_pt_range,
        output_filename=Path(outputs[0].filepath),
        jet_pt_prefix_first=True,
        n_cores=n_cores,
        cross_check_task=cross_check_task,
        double_counting_cut_name=double_counting_cut_name,
    )

    return res


@python_app
def _root_data_frame_response(
    collision_system: str,
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,
    main_jet_pt_range: jsub_helpers.JetPtRange,
    n_cores: int,
    cross_check_task: bool,
    double_counting_cut_name: str,
    job_framework: job_utils.JobFramework,
    inputs: MutableSequence[File] = [],
    outputs: MutableSequence[File] = [],
) -> Tuple[bool, str]:
    """ROOT data frame response app.

    We keep them separate even though they are quite similar so their app names will be different.
    Since they're so simple, this isn't a big sacrifice.
    """
    import random
    from pathlib import Path

    from jet_substructure.cpp import data_frame

    # Shuffle inputs
    # Randomize the input list so we don't always hit the same files at the same time.
    # Note: It randomizes in place.
    random.shuffle(inputs)

    res = data_frame.run_response(
        collision_system=collision_system,
        input_filenames=[Path(f.filepath) for f in inputs],
        tree_name=tree_name,
        prefixes=prefixes,
        grooming_method=grooming_method,
        jet_R=jet_R,
        main_jet_pt_range=main_jet_pt_range,
        output_filename=Path(outputs[0].filepath),
        jet_pt_prefix_first=True,
        n_cores=n_cores,
        cross_check_task=cross_check_task,
        double_counting_cut_name=double_counting_cut_name,
    )

    return res


@python_app
def _root_data_frame_closure(
    collision_system: str,
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,
    main_jet_pt_range: jsub_helpers.JetPtRange,
    n_cores: int,
    cross_check_task: bool,
    base_unfolding_config: Mapping[str, Any],
    unfolding_settings: Mapping[str, Any],
    substructure_variable_name: str,
    double_counting_cut_name: str,
    job_framework: job_utils.JobFramework,
    inputs: MutableSequence[File] = [],
    outputs: MutableSequence[File] = [],
) -> Tuple[bool, str]:
    """ROOT data frame closure app.

    We keep them separate even though they are quite similar so their app names will be different.
    Since they're so simple, this isn't a big sacrifice.
    """
    from pathlib import Path

    from jet_substructure.cpp import data_frame

    res = data_frame.run_create_closure_ratio(
        collision_system=collision_system,
        input_filenames=[Path(f.filepath) for f in inputs],
        tree_name=tree_name,
        prefixes=prefixes,
        grooming_method=grooming_method,
        jet_R=jet_R,
        main_jet_pt_range=main_jet_pt_range,
        output_filename=Path(outputs[0].filepath),
        base_unfolding_config=base_unfolding_config,
        unfolding_settings=unfolding_settings,
        substructure_variable_name=substructure_variable_name,
        jet_pt_prefix_first=True,
        n_cores=n_cores,
        cross_check_task=cross_check_task,
        double_counting_cut_name=double_counting_cut_name,
    )

    return res


@python_app
def _root_data_frame_embedded_pt_hard_scaling(
    collision_system: str,
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,
    main_jet_pt_range: jsub_helpers.JetPtRange,
    n_cores: int,
    cross_check_task: bool,
    double_counting_cut_name: str,
    job_framework: job_utils.JobFramework,
    inputs: MutableSequence[File] = [],
    outputs: MutableSequence[File] = [],
) -> Tuple[bool, str]:
    """ROOT data frame pt hard scaling app.

    We keep them separate even though they are quite similar so their app names will be different.
    Since they're so simple, this isn't a big sacrifice.
    """
    import random
    from pathlib import Path

    from jet_substructure.cpp import data_frame

    # Shuffle inputs
    # Randomize the input list so we don't always hit the same files at the same time.
    # Note: It randomizes in place.
    random.shuffle(inputs)

    res = data_frame.run_embedded_pt_hard_scaling(
        collision_system=collision_system,
        input_filenames=[Path(f.filepath) for f in inputs],
        tree_name=tree_name,
        prefixes=prefixes,
        grooming_method=grooming_method,
        jet_R=jet_R,
        main_jet_pt_range=main_jet_pt_range,
        output_filename=Path(outputs[0].filepath),
        # Workaround for older pythia productions that we can't re-skim so easily.
        # We can remove this eventually when train 2110 is replaced.
        jet_pt_prefix_first=collision_system != "pythia",
        n_cores=n_cores,
        cross_check_task=cross_check_task,
        double_counting_cut_name=double_counting_cut_name,
    )

    return res


@attrs.define
class RootDataFrameProcessingMode:
    name: str
    tag: str
    func: Callable[..., Future[Tuple[bool, str]]]


def setup_root_data_frame(
    processing_mode: str,
    collision_system: str,
    n_cores_per_job: int,
    dataset_config: Mapping[str, Any],
    base_unfolding_config: Mapping[str, Any],
    grooming_methods: Sequence[str],
    job_framework: job_utils.JobFramework,
    selected_train_numbers: Optional[Sequence[int]] = None,
    input_results: Optional[MutableSequence[Future[Any]]] = None,
    unfolding_settings: Optional[Mapping[str, Any]] = None,
    substructure_variable_name: str | None = None,
) -> List[Future[Tuple[bool, str]]]:
    # Validation
    # NOTE: I only compromised on specifying the function here because it loses the typing
    #       information from the `python_app` wrapper anyway.
    _processing_modes = [
        RootDataFrameProcessingMode(
            name="standard",
            tag="",
            func=_root_data_frame,
        ),
        RootDataFrameProcessingMode(
            name="response",
            tag="response",
            func=_root_data_frame_response,
        ),
        RootDataFrameProcessingMode(
            name="closure",
            tag="closure",
            func=_root_data_frame_closure,
        ),
        RootDataFrameProcessingMode(
            name="embedded_pt_hard_scaling",
            tag="embedded_pt_hard_scaling",
            func=_root_data_frame_embedded_pt_hard_scaling,
        ),
    ]
    if processing_mode not in [p.name for p in _processing_modes]:
        raise ValueError(f'Invalid processing mode "{processing_mode}"')
    if processing_mode == "closure" and not unfolding_settings:
        raise ValueError("Must pass unfolding setting with closure")

    # Setup
    mode = [p for p in _processing_modes if processing_mode == p.name][0]
    output_dir = Path("output") / collision_system / "RDF"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = dataset_config["prefixes"]

    # Grab the double counting cut setting
    # We want to protect against accidentally forgetting this in PbPb!
    # NOTE: We use the default double counting cut here by default, but we can configure it by
    #       passing some selected unfolding_settings. We should still be a bit careful with this!
    _config_for_double_counting_cut = unfolding_settings if unfolding_settings is not None else base_unfolding_config["nominal_binning"]["default"]
    _double_counting_cut_name = _config_for_double_counting_cut.get("double_counting_cut", "")
    # Things are treated so different that it's better to be direct about the data collision system.
    _analyzing_pp = (collision_system == "pp" or collision_system == "pythia")
    if not _analyzing_pp and _double_counting_cut_name == "":
        raise ValueError("Must specify a double counting cut setting in PbPb! You can disable it with 'disabled'.")
    # We want this to be disabled in pp. To ensure that's the case, let's set it explicitly here!
    if _analyzing_pp:
        _double_counting_cut_name = "disabled"
    # We'll need the settings below to determine the right range for the true_min_pt
    _double_counting_cut_settings = analysis_jet_substructure.double_counting_cuts[_double_counting_cut_name]

    # If input files aren't passed, then we need to determine them ourselves.
    input_files = []
    if input_results is None:
        logger.info("Determining input files independently.")
        # First, determine the train directories so we can skip over some of them if requested.
        train_directories = set([Path(filename).parent for filename in dataset_config["files"]])
        for train_directory in sorted(train_directories):
            # Select train numbers.
            if selected_train_numbers and int(train_directory.name) not in selected_train_numbers:
                logger.debug(f"Skipping train number {train_directory.name}")
                continue
            logger.info(f"Processing train number {train_directory.name}")

            # Then iterate over the directories.
            for filename in Path(f"{train_directory}/skim/").glob("*.root"):
                # Filter out pt hat bins based on the double counting cut settings
                # NOTE: This won't work if not produced by mammoth...
                if _double_counting_cut_settings.min_pt_hat_bin > 0 and collision_system == "embed_pythia":
                    _extracted_pt_hat_bin = _extract_pt_hat_bin_from_filename_for_embed_pythia_mammoth_production(filename=filename)
                    if _extracted_pt_hat_bin <= _double_counting_cut_settings.min_pt_hat_bin:
                        logger.debug(f"Due to double counting cut of min pt hat bin of {_double_counting_cut_settings.min_pt_hat_bin}, skipping pt hat bin {_extracted_pt_hat_bin} file {filename}")
                        continue
                input_files.append(File(str(filename)))

    # logger.info(f"Input files (len: {len(input_files)}: {input_files}")
    logger.info(f"{processing_mode} RDF, with N cores per job: {n_cores_per_job}")
    # NOTE: Since we now skim the cross check task, we don't need to modify the behavior.
    #       However, in case we need to go back, we avoid wiping out the code entirely.
    #       So for now, we set it always to false.
    cross_check_task = False
    # cross_check_task = dataset_config.get("cross_check_task", False)
    # logger.info(f"Cross check task: {cross_check_task}")

    # Setup optional args
    optional_kwargs = {}
    # We only want to pass the entire configuration for the closure.
    if processing_mode == "closure":
        optional_kwargs = {
            "base_unfolding_config": base_unfolding_config,
            "unfolding_settings": unfolding_settings,
            "substructure_variable_name": substructure_variable_name,
        }

    results = []
    for grooming_method in grooming_methods:
        # Determine the broad smeared jet pt range. We basically want this to cover our
        # entire range of interest. Since this will usually correspond with our unfolding
        # nominal smeared jet pt binning, we just advantage of those values.
        # NOTE: We're okay using the default here because the smeared jet pt bins won't vary
        _nominal_smeared_jet_pt_bins = unfolding_base.get_binning(
            # By passing an empty config here, we should only retrieve from the "default",
            # which is what we want here.
            unfolding_settings={},
            base_unfolding_config=base_unfolding_config,
            binning_type="smeared",
            grooming_method=grooming_method,
            # NOTE: We are hard coding this assumption. It shouldn't vary much better
            #       different substructure variables, so this should be okay
            substructure_variable_to_analyze="kt",
            variable_to_retrieve="jet_pt",
        )
        main_jet_pt_range = jsub_helpers.JetPtRange(_nominal_smeared_jet_pt_bins[0], _nominal_smeared_jet_pt_bins[-1])

        # Setup file IO
        tag = f"_{mode.tag}" if mode.tag else ""
        # If we have a single train, then append the train number to the tag.
        if selected_train_numbers and len(selected_train_numbers) == 1:
            tag += f"_{selected_train_numbers[0]}"
        output_file = (
            output_dir / f"{dataset_config['name']}_{grooming_method}_prefixes_{'_'.join(prefixes.keys())}{tag}.root"
        )
        parsl_output_file = File(str(output_file))
        results.append(
            mode.func(
                collision_system=collision_system,
                tree_name="tree" if not cross_check_task else dataset_config["tree_name"],
                prefixes=list(prefixes.keys()),
                grooming_method=grooming_method,
                jet_R=dataset_config["jet_R"],
                main_jet_pt_range=main_jet_pt_range,
                n_cores=n_cores_per_job,
                cross_check_task=cross_check_task,
                job_framework=job_framework,
                double_counting_cut_name=_double_counting_cut_name,
                # Need to grab the outputs here to ensure that the dependencies are tracked properly.
                inputs=[r.outputs[0] for r in input_results] if input_results and job_framework == job_utils.JobFramework.parsl else input_files,  # type: ignore[attr-defined]
                outputs=[parsl_output_file],
                **optional_kwargs,
            )
        )

    return results


def embedded_pt_hard_scaling_cross_check(
    collision_system: str,
    n_cores_per_job: int,
    dataset_config: Mapping[str, Any],
    base_unfolding_config: Mapping[str, Any],
    grooming_methods: Sequence[str],
    job_framework: job_utils.JobFramework,
) -> List[Future[Tuple[bool, str]]]:
    results = []

    # Assume that we have on train per line. That's usually a pretty good assumption.
    # Each one will be written to separate files.
    train_directories = set([Path(filename).parent for filename in dataset_config["files"]])
    for train_directory in sorted(train_directories):
        results.extend(
            setup_root_data_frame(
                processing_mode="embedded_pt_hard_scaling",
                collision_system=collision_system,
                n_cores_per_job=n_cores_per_job,
                dataset_config=dataset_config,
                base_unfolding_config=base_unfolding_config,
                # Take just the first grooming method - they'll all be the same because we don't care
                # about the grooming method for this check.
                grooming_methods=[grooming_methods[0]],
                job_framework=job_framework,
                selected_train_numbers=[int(train_directory.name)],
            )
        )

    return results


# Matches pt hat values 1-20 in format of __1__ used in mammoth productions
# Based on https://stackoverflow.com/a/39138675/12907985
_match_pt_hat_bin_from_embed_pythia_mammoth_production = re.compile("__(0?[1-9]|1[0-9]|20)__")

def _extract_pt_hat_bin_from_filename_for_embed_pythia_mammoth_production(filename: Path) -> int:
    # Example: `pythia__2640__run_by_run__LHC20g4__295612__1__AnalysisResults_20g4_007`
    possible_pt_hat_bins = _match_pt_hat_bin_from_embed_pythia_mammoth_production.findall(str(filename))
    if len(possible_pt_hat_bins) == 0:
        raise ValueError(f"Extracted no pt hat bins from filename {filename}")
    if len(possible_pt_hat_bins) > 1:
        raise ValueError(f"Extracted multiple pt hat bins: ({possible_pt_hat_bins}) bin from filename {filename}")
    return int(possible_pt_hat_bins[0])


@python_app
def _unfolding_standard(
    settings: unfolding_base.Settings2D,
    unfolding_for_pp: bool,
    reweight_prior: bool,
    reweight_data_dataset_name: str,
    reweight_response_dataset_name: str,
    data_tree_name: str,
    response_tree_name: str,
    debug_cpp_code: bool,
    job_framework: job_utils.JobFramework,
    inputs: Tuple[Sequence[File], Sequence[File]] = ([], []),
    outputs: Sequence[File] = [],
) -> bool:
    import random
    from pathlib import Path

    from jet_substructure.cpp import unfolding_2D

    # Randomize the input list so we don't always hit the same files at the same time.
    # 0 are the data filenames, 1 are the response filenames
    # Note: It randomizes in place.
    data_filenames = [Path(f.filepath) for f in inputs[0]]
    random.shuffle(data_filenames)
    response_filenames = [Path(f.filepath) for f in inputs[1]]
    random.shuffle(response_filenames)

    return unfolding_2D.run_unfolding(
        settings=settings,
        data_filenames=data_filenames,
        data_tree_name=data_tree_name,
        response_filenames=response_filenames,
        response_tree_name=response_tree_name,
        unfolding_for_pp=unfolding_for_pp,
        reweight_prior=reweight_prior,
        reweight_data_dataset_name=reweight_data_dataset_name,
        reweight_response_dataset_name=reweight_response_dataset_name,
        debug_cpp_code=debug_cpp_code,
    )


@python_app
def _unfolding_closure(
    settings: unfolding_base.Settings2D,
    closure_variation: str,
    unfolding_for_pp: bool,
    fraction_for_response: float,
    reweight_data_dataset_name: str,
    reweight_response_dataset_name: str,
    response_tree_name: str,
    debug_cpp_code: bool,
    job_framework: job_utils.JobFramework,
    inputs: Tuple[Sequence[File], Sequence[File]] = ([], []),
    outputs: Sequence[File] = [],
) -> bool:
    import random
    from pathlib import Path

    from jet_substructure.cpp import unfolding_2D

    # Randomize the input list so we don't always hit the same files at the same time.
    # 0 are the data filenames, 1 are the response filenames
    # Note: It randomizes in place.
    response_filenames = [Path(f.filepath) for f in inputs[1]]
    random.shuffle(response_filenames)

    return unfolding_2D.run_unfolding_closure_reweighting(
        settings=settings,
        response_filenames=response_filenames,
        response_tree_name=response_tree_name,
        closure_variation=closure_variation,
        unfolding_for_pp=unfolding_for_pp,
        fraction_for_response=fraction_for_response,
        reweight_data_dataset_name=reweight_data_dataset_name,
        reweight_response_dataset_name=reweight_response_dataset_name,
        debug_cpp_code=debug_cpp_code,
    )


@attrs.define
class UnfoldingRuntimeSettings:
    variable_to_unfold: str = attrs.field(default="kt")
    normalize_variable_by_jet_pt: bool = attrs.field(default=False)
    selected_settings: List[str] = attrs.field(factory=lambda: ["default"])
    _output_dir_tag: str = attrs.field(default="")

    def output_dir(self, data_collision_system: str) -> Path:
        _path = Path("output") / data_collision_system / "unfolding"
        if self._output_dir_tag:
            _path = _path / self._output_dir_tag
        return _path


def setup_all_unfolding(  # noqa: C901
    data_collision_system: str,
    base_dataset_config: Mapping[str, Any],
    grooming_methods: Sequence[str],
    unfolding_runtime_settings: UnfoldingRuntimeSettings,
    n_cores_per_job: int,
    job_framework: job_utils.JobFramework,
    debug_cpp_code: bool = False,
) -> List[Future[Any]]:
    """Setup unfolding jobs.

    Args:
        data_collision_system: Name of the data collision system.
        base_dataset_config: Base dataset configuration.
        grooming_methods: Grooming methods to unfold.
        n_cores_per_job: N cores to make available for ROOT. This is of limited utility for RooUnfold,
            but anecdotally, it seems to still hope a bit with I/O.
    Returns:
        List of `AppFuture` created when defining the jobs.
    """
    # Validation
    if data_collision_system not in ["pp", "PbPb"]:
        raise ValueError(f"Collision must be either pp or PbPb for unfolding. Passed: {data_collision_system}")
    selected_unfolding_settings = unfolding_runtime_settings.selected_settings
    logger.info(f"Unfolding settings: {selected_unfolding_settings}")
    # Setup
    base_unfolding_config = base_dataset_config["unfolding"]
    output_dir = unfolding_runtime_settings.output_dir(data_collision_system=data_collision_system)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Things are treated so different that it's better to be direct about the data collision system.
    unfolding_for_pp = data_collision_system == "pp"
    # Need the response collision system too.
    data_to_response_collision_system_name = {
        "pp": "pythia",
        "PbPb": "embed_pythia",
    }
    response_collision_system = data_to_response_collision_system_name[data_collision_system]

    results: List[Future[Any]] = []
    # Maps from the reweight hist name to the task.
    _reweight_prior_results = {}
    for unfolding_settings_name in selected_unfolding_settings:
        # Setup
        unfolding_settings = base_unfolding_config["settings"][unfolding_settings_name]
        datasets_name = unfolding_settings["datasets"]
        _get_bins = functools.partial(
            unfolding_base.get_binning,
            base_unfolding_config=base_unfolding_config,
            unfolding_settings=unfolding_settings,
        )

        # Grab the double counting cut setting.
        # We want to protect against accidentally forgetting this in PbPb!
        _double_counting_cut_name = unfolding_settings.get("double_counting_cut", "")
        if not unfolding_for_pp and _double_counting_cut_name == "":
            raise ValueError("Must specify a double counting cut setting in PbPb! You can disable it with 'disabled'.")
        # We want this to be disabled in pp. To ensure that's the case, let's set it explicitly here!
        if unfolding_for_pp:
            _double_counting_cut_name = "disabled"
        # We'll need the settings below to determine the right range for the true_min_pt
        _double_counting_cut_settings = analysis_jet_substructure.double_counting_cuts[_double_counting_cut_name]

        # Datasets config and input files.
        data_dataset_config = base_dataset_config["datasets"][datasets_name][data_collision_system]
        data_train_directories = set([Path(filename).parent for filename in data_dataset_config["files"]])
        response_dataset_config = base_dataset_config["datasets"][datasets_name][response_collision_system]
        response_train_directories = set([Path(filename).parent for filename in response_dataset_config["files"]])

        # Determine filenames first since they don't depend on grooming methods
        data_files: List[File] = []
        response_files: List[File] = []
        for label, train_directories, files in [
            (data_collision_system, data_train_directories, data_files),
            (response_collision_system, response_train_directories, response_files),
        ]:
            for train_directory in sorted(train_directories):
                logger.info(f"Processing {label} train number {train_directory.name}")

                # Then iterate over the directories.
                for filename in Path(f"{train_directory}/skim/").glob("*.root"):
                    # Filter out pt hat bins based on the double counting cut settings
                    # NOTE: This won't work if not produced by mammoth...
                    if _double_counting_cut_settings.min_pt_hat_bin > 0 and label == "embed_pythia":
                        _extracted_pt_hat_bin = _extract_pt_hat_bin_from_filename_for_embed_pythia_mammoth_production(filename=filename)
                        if _extracted_pt_hat_bin <= _double_counting_cut_settings.min_pt_hat_bin:
                            logger.debug(f"Due to double counting cut of min pt hat bin of {_double_counting_cut_settings.min_pt_hat_bin}, skipping pt hat bin {_extracted_pt_hat_bin} file {filename}")
                            continue
                    files.append(File(str(filename)))

        # For parsl to keep track of the data and response files
        input_files = [data_files, response_files]

        # Narrow grooming methods if necessary due to cross check task.
        # NOTE: Could add more validation, but this is probably enough for now.
        if data_dataset_config.get("cross_check_task", False):
            grooming_methods = data_dataset_config["grooming_methods"]
        if response_dataset_config.get("cross_check_task", False):
            grooming_methods = response_dataset_config["grooming_methods"]

        # Then we determine the settings.
        for grooming_method in grooming_methods:
            settings = {}

            # First, define the default settings.
            _default_settings = unfolding_base.Settings2D(
                grooming_method=grooming_method,
                jet_pt=unfolding_base.JetPtSettings2D.from_binning(
                    true_bins=_get_bins(
                        grooming_method=grooming_method, substructure_variable_to_analyze=unfolding_runtime_settings.variable_to_unfold,
                        binning_type="true", variable_to_retrieve="jet_pt",
                    ),
                    smeared_bins=_get_bins(
                        grooming_method=grooming_method, substructure_variable_to_analyze=unfolding_runtime_settings.variable_to_unfold,
                        binning_type="smeared", variable_to_retrieve="jet_pt",
                    ),
                    true_min_pt=_double_counting_cut_settings.min_true_pt,
                ),
                substructure_variable=unfolding_base.SubstructureVariableSettings2D.from_binning(
                    true_bins=_get_bins(
                        binning_type="true",
                        grooming_method=grooming_method,
                        substructure_variable_to_analyze=unfolding_runtime_settings.variable_to_unfold,
                        variable_to_retrieve=f"var_over_pt" if unfolding_runtime_settings.normalize_variable_by_jet_pt else None,
                    ),
                    smeared_bins=_get_bins(
                        binning_type="smeared",
                        grooming_method=grooming_method,
                        substructure_variable_to_analyze=unfolding_runtime_settings.variable_to_unfold,
                        variable_to_retrieve=f"var_over_pt" if unfolding_runtime_settings.normalize_variable_by_jet_pt else None,
                    ),
                    name=unfolding_runtime_settings.variable_to_unfold,
                    variable_name=unfolding_runtime_settings.variable_to_unfold,
                    untagged_bin_below_range=True,
                    normalize_by_jet_pt=unfolding_runtime_settings.normalize_variable_by_jet_pt,
                ),
                double_counting_cut_name=_double_counting_cut_name,
                suffix=base_unfolding_config["suffix"],
                label=unfolding_settings.get("label", ""),
                output_dir=output_dir,
                use_pure_matches=False,
            )
            settings[_default_settings.output_tag] = _default_settings

            closure_specific_settings = unfolding_settings.get("closure_settings", {})
            # And then add the variations.
            if unfolding_settings.get("compare_pure_matches", False):
                # Pure matches
                # Disabled by default because it doesn't seem especially important or helpful
                _temp_settings = copy.deepcopy(_default_settings)
                _temp_settings.use_pure_matches = True
                settings[_temp_settings.output_tag] = _temp_settings

            if unfolding_settings["closures"]:
                # Untagged above
                _temp_settings = copy.deepcopy(_default_settings)
                _smeared_bins = _temp_settings.substructure_variable.smeared_bins
                # Replace the untagged value with our new untagged value, and then move it to the upper edge.
                _smeared_bins[0] = closure_specific_settings["max_untagged_bin"]
                _smeared_bins = np.roll(_smeared_bins, -1)
                _temp_settings.substructure_variable = unfolding_base.SubstructureVariableSettings2D.from_binning(
                    true_bins=_temp_settings.substructure_variable.true_bins,
                    smeared_bins=_smeared_bins,
                    name=_temp_settings.substructure_variable.name,
                    variable_name=_temp_settings.substructure_variable.variable_name,
                    untagged_bin_below_range=False,
                )
                settings[_temp_settings.output_tag] = _temp_settings

            # If we're requesting a reweighted prior, we need to ensure that it's created before
            # we attempt to unfold it.
            # NOTE: We could put this above the iteration over grooming methods, but then we would have to match
            #       the grooming method outputs to the unfolding, which could be potentially quite tedious.
            #       Instead, we take a slight hit in efficiency and just call it here, and then easily match them.
            reweight_prior_results = []
            if unfolding_settings["reweight_prior"] or unfolding_settings["closures"]:
                # We enable for both the straightforward reweighting, as well as for the closures where we take
                # advantage of the reweighting.
                for _collision_system in [data_collision_system, response_collision_system]:
                    # Get unfolding name, and check if we already have the jobs setup. If so, grab, just use
                    # the existing task and output files. Otherwise, set up the task, and then store the output.
                    # This avoids duplicate jobs, which could potentially overwriting outputs from each other.
                    _reweight_hist_name = unfolding_base.hist_name_for_ratio_2D(
                        grooming_method=grooming_method,
                        prefix_for_ratio="hybrid" if "embed" in _collision_system else "data",
                        substructure_variable_name=_default_settings.substructure_variable.variable_name,
                        smeared_substructure_variable_bins=_default_settings.substructure_variable.smeared_bins,
                        smeared_jet_pt_bins=_default_settings.jet_pt.smeared_bins,
                        double_counting_cut_name=_default_settings.double_counting_cut_name,
                    )
                    # Add collision system to make it 100% unique (it already is for PbPb due to the prefix_for_ratio,
                    # but it's not for pp).
                    _reweight_hist_name = f"{_collision_system}_{_reweight_hist_name}"
                    if _reweight_hist_name not in _reweight_prior_results:
                        # NOTE: We can use this name directly without adding an additional level of indirection for
                        #       the grooming method because the grooming method (as well as binning) is encoded in
                        #       the hist name.
                        logger.info(f"Adding reweighting {_reweight_hist_name}")
                        _reweight_prior_results[_reweight_hist_name] = setup_root_data_frame(
                            processing_mode="closure",
                            collision_system=_collision_system,
                            n_cores_per_job=n_cores_per_job,
                            # We always want the nominal dataset because we want to reweight by
                            # the nominal data and response. However, we want the binning of the
                            # particular case that we are considering.
                            dataset_config=base_dataset_config["datasets"]["nominal"][_collision_system],
                            grooming_methods=[grooming_method],
                            job_framework=job_framework,
                            base_unfolding_config=base_unfolding_config,
                            unfolding_settings=unfolding_settings,
                            substructure_variable_name=_default_settings.substructure_variable.variable_name,
                        )
                        # And add to final outputs
                        # We only want to add here - otherwise it can show up multiple times, which is
                        # confusing.
                        results.extend(_reweight_prior_results[_reweight_hist_name])

                    # Need to make explicit dependencies for this task, so we add the results into
                    # reweight_prior_results. It will be either from adding the task now, or from
                    # using an existing one stored in the dict.
                    logger.info(f"Using reweighting {_reweight_hist_name}")
                    reweight_prior_results.extend(_reweight_prior_results[_reweight_hist_name])
            # Handle reweighted prior results.
            # Add as job inputs to make them dependencies.
            job_input_files = list(input_files)
            # Apparently if we take the outputs it complete messes up the dependency because one of the
            # outputs is considered to be infinitely pending, despite the fact that it already exists and
            # the task has already returned... So instead, we pass the AppFuture to make the dependency.
            # That means our typing information is a little fudged, but it works fine anyway since we don't
            # actually care about the value of the result - just that the file was created.
            # Note that we have to lie with the typing here...
            job_input_files.extend([r for r in reweight_prior_results])  # type: ignore[misc]

            # Standard unfolding
            for s in settings.values():
                logger.info(f"Adding standard unfolding: {s.output_tag}")
                if s.substructure_variable.disable_untagged_bin:
                    logger.info("NOTE: Disabled untagged bin")
                # NOTE: This is missing some output files (like the trivial closures). But good enough for now...
                parsl_output_file = File(str(s.output_filename))
                results.append(
                    _unfolding_standard(
                        settings=s,
                        unfolding_for_pp=unfolding_for_pp,
                        reweight_prior=unfolding_settings["reweight_prior"],
                        # We always want to reweight with the nominal datasets.
                        reweight_data_dataset_name=base_dataset_config["datasets"]["nominal"][data_collision_system][
                            "name"
                        ],
                        reweight_response_dataset_name=base_dataset_config["datasets"]["nominal"][
                            response_collision_system
                        ]["name"],
                        data_tree_name="tree",
                        # Since we skim everything now, we should have uniform input names here.
                        response_tree_name="tree",
                        # if not response_dataset_config.get("cross_check_task", False)
                        # else response_dataset_config["tree_name"],
                        debug_cpp_code=debug_cpp_code,
                        job_framework=job_framework,
                        inputs=job_input_files,  # type: ignore[arg-type]
                        outputs=[parsl_output_file],
                    )
                )

            if unfolding_settings["closures"]:
                # Skip the untagged bin moved to above the smeared range.
                for s in list(settings.values())[:-1]:
                    for closure_variation in ["split_MC", "reweight_pseudo_data", "reweight_response"]:
                        logger.info(f"Adding unfolding closures: {s.output_tag}, variation: {closure_variation}")
                        parsl_output_file = File(f"{s.output_filename}_closure_{closure_variation}")
                        results.append(
                            _unfolding_closure(
                                settings=s,
                                closure_variation=closure_variation,
                                unfolding_for_pp=unfolding_for_pp,
                                fraction_for_response=closure_specific_settings["split_MC_fraction_for_response"],
                                # We always want to reweight with the nominal datasets.
                                reweight_data_dataset_name=base_dataset_config["datasets"]["nominal"][
                                    data_collision_system
                                ]["name"],
                                reweight_response_dataset_name=base_dataset_config["datasets"]["nominal"][
                                    response_collision_system
                                ]["name"],
                                # Since we skim everything now, we should have uniform input names here.
                                response_tree_name="tree",
                                debug_cpp_code=debug_cpp_code,
                                # if not response_dataset_config.get("cross_check_task", False)
                                # else response_dataset_config["tree_name"],
                                job_framework=job_framework,
                                inputs=job_input_files,  # type: ignore[arg-type]
                                outputs=[parsl_output_file],
                            )
                        )

    return results


def _hours_in_walltime(walltime: str) -> int:
    return int(walltime.split(":")[0])


def setup_job_framework(
    job_framework: job_utils.JobFramework,
    jobs_to_execute: Sequence[str],
    task_config: job_utils.TaskConfig,
    facility: job_utils.FACILITIES,
    walltime: str,
    target_n_tasks_to_run_simultaneously: int,
    log_level: int,
    conda_environment_name: Optional[str] = None,
) -> Tuple[parsl.DataFlowKernel, parsl.Config] | Tuple[dask.distributed.Client, dask.distributed.SpecCluster]:
    # First, need to figure out if we need additional environments such as ROOT
    _additional_worker_init_script = alice_job_utils.determine_additional_worker_init(
        conda_environment_name=conda_environment_name,
        tasks_to_run=[jobs_to_execute],
        tasks_requiring_root=[
            "repair_root_files",
            "extract_scale_factors",
            "root_data_frame_embedded_pt_hard_scaling",
            "root_data_frame",
            "root_data_frame_response",
            "root_data_frame_closure",
            "unfolding",
        ],
        tasks_requiring_aliphysics=["repair_root_files", "extract_scale_factors"],
        tasks_requiring_roounfold=["unfolding"],
    )
    return job_utils.setup_job_framework(
        job_framework=job_framework,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        additional_worker_init_script=_additional_worker_init_script,
    )


def setup_and_submit_tasks(  # noqa: C901
    job_framework: job_utils.JobFramework,
    task_config: job_utils.TaskConfig,
    base_dataset_name: str,
    dataset_type: str,
    collision_system: str,
    jobs_to_execute: Sequence[str],
    input_grooming_methods: Optional[Sequence[str]] = None,
    unfolding_runtime_settings: UnfoldingRuntimeSettings = UnfoldingRuntimeSettings(),
    dask_client: Optional[dask.distributed.Client] = None,
) -> List[Future[Any]]:
    # Validation
    if dask_client is None and job_framework == job_utils.JobFramework.dask_delayed:
        raise ValueError("Must provide dask client if running with dask_delayed job framework!")

    #nodes_to_allocate = 8
    #nodes_to_allocate = 1
    #jobs_per_node = 8

    # Default to all methods. We can restrict if the particular tasks if we see the cross check task.
    if input_grooming_methods is None:
        grooming_methods = [
            # "leading_kt",
            # "leading_kt_z_cut_02",
            # "leading_kt_z_cut_04",
            # "dynamical_z",
            # "dynamical_core",
            "dynamical_kt",
            # "dynamical_time",
            # "soft_drop_z_cut_02",
            # "soft_drop_z_cut_04",
        ]
    else:
        grooming_methods = list(input_grooming_methods)

    # Basic setup for jobs
    _possible_jobs = [
        "repair_root_files",
        "convert_to_parquet",
        "extract_scale_factors",
        "calculate_embedding_skim",
        "calculate_data_skim",
        "root_data_frame_embedded_pt_hard_scaling",
        "root_data_frame",
        "root_data_frame_response",
        "root_data_frame_closure",
        "unfolding",
    ]
    # Validation
    for job_name in jobs_to_execute:
        if job_name not in _possible_jobs:
            raise RuntimeError(
                f"Requested to run job {job_name}, but the name is invalid." f" Possible jobs: {_possible_jobs}"
            )

    # Helpers
    base_dataset_config = read_dataset_config(base_dataset_name=base_dataset_name)
    dataset_config = base_dataset_config["datasets"][dataset_type][collision_system]

    # Determine main settings
    # How many events we split into each parquet file.
    # This size is a balance: we need to be able to run many jobs in parallel, but we need
    # to avoid using too much memory per job. Default: 1e5
    entries_per_job = int(dataset_config.get("entries_per_job", 1e5))
    # If we have a cross check task, it only has the output from a limited set of grooming methods.
    # In that case, we should restrict to only the valid ones.
    cross_check_task = dataset_config.get("cross_check_task", False)
    if cross_check_task:
        grooming_methods = dataset_config["grooming_methods"]

    results: List[Future[Any]] = []
    all_results: List[Future[Any]] = []
    logger.info(f"Jobs to execute: {jobs_to_execute}")
    if "repair_root_files" in jobs_to_execute:
        # NOTE: No input_results here because it's the first step.
        results = setup_repair_root_files(
            n_cores_per_job=task_config.n_cores_per_task,
            dataset_config=dataset_config,
            # selected_train_numbers=list(range(6296, 6297)),
        )
        all_results.extend(results)
    if "convert_to_parquet" in jobs_to_execute:
        # Redefine results so we can use that in the next step.
        _all_results, results = setup_convert_to_parquet(
            collision_system=collision_system,
            entries_per_job=entries_per_job,
            dataset_config=dataset_config,
            job_framework=job_framework,
            input_results=results if results else None,
        )
        all_results.extend(_all_results)
    yaml_result: Optional[Future[bool]] = None
    if "extract_scale_factors" in jobs_to_execute:
        # NOTE: We don't take any input_results because we're super dependent on knowing the
        #       pt hard bins. We would have to reorganize the outputs heavily, so it's
        #       in fact easier for us to determine them independently. Note also that we
        #       we could only take the outputs from repair_root_files because we need to
        #       go back to the original repaired root files.
        scale_factors = setup_extract_scale_factors(
            collision_system=collision_system,
            dataset_config=dataset_config,
            job_framework=job_framework,
            # selected_train_numbers=list(range(6316, 6318)),
        )

        all_results.extend(list(scale_factors.values()))
        yaml_result = setup_write_scale_factors(
            collision_system=collision_system,
            dataset_config=dataset_config,
            scale_factors=scale_factors,
            job_framework=job_framework,
            # selected_train_numbers=list(range(6316, 6318)),
        )
        all_results.append(yaml_result)
        single_results = setup_extract_embedding_pt_hard_spectra(
            collision_system=collision_system,
            dataset_config=dataset_config,
            job_framework=job_framework,
            input_results=[yaml_result],
            # selected_train_numbers=list(range(6316, 6318)),
        )
        all_results.append(single_results)
    if "calculate_embedding_skim" in jobs_to_execute:
        # TODO: Devise an approach to retry.
        if not cross_check_task:
            results = setup_calculate_embedding_skim(
                collision_system=collision_system,
                entries_per_job=entries_per_job,
                dataset_config=dataset_config,
                job_framework=job_framework,
                # selected_train_numbers=list(range(5966, 5967)),
                input_results=results if results else None,
            )
        else:
            results = setup_calculate_cross_check_task_skim(
                collision_system=collision_system,
                dataset_config=dataset_config,
                grooming_methods=grooming_methods,
                job_framework=job_framework,
                # selected_train_numbers=list(range(5966, 5967)),
                input_results=[yaml_result] if yaml_result else None,
            )
        all_results.extend(results)
    if "calculate_data_skim" in jobs_to_execute:
        # TODO: Devise an approach to retry.
        results = setup_calculate_data_skim(
            collision_system=collision_system,
            entries_per_job=entries_per_job,
            dataset_config=dataset_config,
            job_framework=job_framework,
            # selected_train_numbers=list(range(5977, 5978)),
            input_results=results if results else None,
        )
        all_results.extend(results)
    if "root_data_frame" in jobs_to_execute:
        rdf_results = setup_root_data_frame(
            processing_mode="standard",
            collision_system=collision_system,
            n_cores_per_job=task_config.n_cores_per_task,
            dataset_config=dataset_config,
            base_unfolding_config=base_dataset_config["unfolding"],
            grooming_methods=grooming_methods,
            job_framework=job_framework,
            # selected_train_numbers=list(range(5977, 5978)),
            input_results=results if results else None,
        )
        all_results.extend(rdf_results)
    if "root_data_frame_response" in jobs_to_execute:
        rdf_results = setup_root_data_frame(
            processing_mode="response",
            collision_system=collision_system,
            n_cores_per_job=task_config.n_cores_per_task,
            dataset_config=dataset_config,
            base_unfolding_config=base_dataset_config["unfolding"],
            grooming_methods=grooming_methods,
            job_framework=job_framework,
            # selected_train_numbers=list(range(6338, 6339)),
            input_results=results if results else None,
        )
        all_results.extend(rdf_results)
    if "root_data_frame_embedded_pt_hard_scaling" in jobs_to_execute:
        # We're doing something special here, so we don't want previous job inputs.
        all_results.extend(
            embedded_pt_hard_scaling_cross_check(
                collision_system=collision_system,
                n_cores_per_job=task_config.n_cores_per_task,
                dataset_config=dataset_config,
                base_unfolding_config=base_dataset_config["unfolding"],
                grooming_methods=grooming_methods,
                job_framework=job_framework,
            )
        )
    # if "root_data_frame_closure" in jobs_to_execute:
    #     # We'll always want both, so let's just do both.
    #     temp_results = []
    #     for _collision_system in ["PbPb", "embedPythia"]:
    #         temp_results.extend(
    #             setup_root_data_frame(
    #                 processing_mode="closure",
    #                 collision_system=_collision_system,
    #                 n_cores_per_job=task_config.n_cores_per_task,
    #                 dataset_config=dataset_config,
    #                 base_unfolding_config=base_dataset_config["unfolding"],
    #                 grooming_methods=grooming_methods,
    #                 # selected_train_numbers=list(range(5977, 5978)),
    #                 input_results=results if results else None,
    #             )
    #         )
    #     results.extend(temp_results)
    if "unfolding" in jobs_to_execute:
        results = setup_all_unfolding(
            data_collision_system=collision_system,
            base_dataset_config=base_dataset_config,
            grooming_methods=grooming_methods,
            unfolding_runtime_settings=unfolding_runtime_settings,
            n_cores_per_job=task_config.n_cores_per_task,
            job_framework=job_framework,
        )
        # results = setup_unfolding(
        #     grooming_methods=grooming_methods,
        #     datasets_name="nominal",
        #     tag="test_original_parsl_method",
        #     run_closures=False,
        # )
        all_results.extend(results)

    logger.info(f"Accumulated {len(all_results)} results")

    if job_framework == job_utils.JobFramework.dask_delayed:
        assert dask_client is not None
        return dask_client.compute(  # type: ignore[no-any-return,no-untyped-call]
            all_results,
            # Distributed assumes functions are pure, but usually mine are not (ie. they create files)
            pure=False,
            resources={"n_cores": task_config.n_cores_per_task}
        )
    return all_results


def follow_progress_of_futures(futures: List[Future[Any]]) -> None:
    # Process the futures, showing processing progress
    # Since it returns the results, we can actually use this to accumulate results.
    gen_results = job_utils.provide_results_as_completed(futures, running_with_parsl=True)

    # In order to support writing histograms from multiple systems, we need to index the output histograms
    # by the collision system + centrality.
    with helpers.progress_bar() as progress:
        track_results = progress.add_task(total=len(futures), description="Processing results...")
        # for a in all_results:
        for result in gen_results:
            # r = a.result()
            logger.info(f"result: {result}")
            progress.update(track_results, advance=1)

    # As far as I can tell, jobs will start executing as soon as they can, regardless of
    # asking for the result. By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns={**locals(), **globals()})

    # In case we close IPython early, wait on results
    # print each job status, initially all are running
    # print ("Job Status: {}".format([r.done() for r in results]))
    # Wait for all apps to complete
    # res = [r.result() for r in results]
    res = [r.result() for r in futures]
    logger.info(res)

    logger.info("Done")


def run(job_framework: job_utils.JobFramework) -> List[Future[Any]]:
    # Settings
    # Base settings
    facility: job_utils.FACILITIES = "rehlers_mbp_m1pro"
    conda_environment_name = "substructure_c_24_06"
    # base_dataset_name = "PbPb_central_R02_pass1"
    # base_dataset_name = "PbPb_central_R02_pass3"
    base_dataset_name = "PbPb_semi_central_R02_pass3"
    # base_dataset_name = "pp_R02"
    # base_dataset_name = "pp_R04_validation"
    # base_dataset_name = "PbPb_semi_central_R04_validation"
    # dataset_type = "rmax_070"
    dataset_type = "nominal"
    collision_system = "PbPb"

    # Job settings
    jobs_to_execute = [
        # "repair_root_files",
        # "convert_to_parquet",
        # "extract_scale_factors",
        # "calculate_embedding_skim",
        # "calculate_data_skim",
        # "root_data_frame",
        # "root_data_frame_embedded_pt_hard_scaling",
        # "root_data_frame_response",
        "unfolding",
    ]
    grooming_methods = [
        # "leading_kt",
        # "leading_kt_z_cut_02",
        # "leading_kt_z_cut_04",
        # "dynamical_z",
        # "dynamical_core",
        "dynamical_kt",
        # "dynamical_time",
        # "soft_drop_z_cut_02",
        # "soft_drop_z_cut_04",
    ]
    unfolding_runtime_settings = UnfoldingRuntimeSettings(
        selected_settings=[
            # TODO: Re-run pp with a different split MC fraction...
            "default",
            # Soft Drop z > 0.2 tests for min kt
            # "default_z_cut_02_kt_0.5",
            # "default_z_cut_02_kt_0.5_reweight_prior",
            # "default_z_cut_02_kt_0.25",
            # "default_z_cut_02_kt_0.25_reweight_prior",
            # "default_z_cut_02_kt_0",
            # "default_z_cut_02_kt_0_reweight_prior",
            # "default_2_4_split",
            # "default_2_4_split_reweight_prior",
            # "default_kt_1.5",
            # "default_kt_1.5_reweight_prior",
            # "default_kt_1",
            # "default_kt_1_reweight_prior",
            # Semi-central
            # "default_kt_1_var2",
            # "default_kt_1_var2_reweight_prior",
            # "default_kt_1_var3",
            # "default_kt_1_var3_reweight_prior",
            # "default_kt_1_6",
            # "default_kt_1_6_reweight_prior",
            # "default_kt_1_6_var4",
            # "default_kt_1_6_var4_reweight_prior",
            # Central
            # "default_kt_1.5_6",
            # "default_kt_1.5_6_reweight_prior",
            # "default_kt_1.5_6_var2",
            # "default_kt_1.5_6_var2_reweight_prior",
            # *[f"var_{i}" for i in range(1, 17) if i < 7 or i > 10] + ["default"],
            # Systematics
            # Unfolding
            #"truncation_low",
            #"truncation_high",
            #"random_binning",
            # Unfolding PbPb
            #"reweight_prior",
            # Tracking efficiency
            #"tracking_efficiency",
            # PbPb background
            #"background_low",
            #"background_high",
        ],
    )

    # Job execution configuration
    task_name = "unfolding_hardest_kt"
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    # n_cores_to_allocate = 120
    # n_cores_to_allocate = 110
    # Formerly n_cores_to_allocate (and they're equal if n_cores_per_task==1)
    target_n_tasks_to_run_simultaneously = 8
    walltime = "24:00:00"
    log_level = logging.INFO
    debug_mode = False

    if debug_mode:
        # Usually, we want to run in the short queue
        n_cores_to_allocate = 2
        walltime = "1:59:00"

    #facility="ORNL_b587_long" if _hours_in_walltime(walltime) >= 2 else "ORNL_b587_short",

    # Keep the job executor just to keep it alive
    job_executor, _job_framework_config = setup_job_framework(
        job_framework=job_framework,
        jobs_to_execute=jobs_to_execute,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        conda_environment_name=conda_environment_name,
    )
    futures = setup_and_submit_tasks(
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
    follow_progress_of_futures(futures=futures)

    return futures


if __name__ == "__main__":
    run(job_framework=job_utils.JobFramework.dask_delayed)