#!/usr/bin/env python3

"""" Submit analysis using parsl.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import parsl
import uproot4 as uproot
from parsl.addresses import address_by_hostname
from parsl.app.app import python_app
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture, DataFuture
from parsl.executors import HighThroughputExecutor
from parsl.monitoring.monitoring import MonitoringHub
from parsl.providers import SlurmProvider

from pachyderm import yaml

from jet_substructure.base import helpers

logger = logging.getLogger(__name__)


def read_config(collision_system: str, config_path: Path = Path("config/new_config.yaml")) -> Dict[str, Any]:
    """ Read collision system configuration from YAML file.

    The collision system specification is defined in the YAML file.

    Args:
        collision_system: Name of the collision system.
        config_path: Path to the configuration file. Default: "config/new_config.yaml".
    Returns:
        The collision system configuration.
    """
    y = yaml.yaml()
    with open(config_path, "r") as f:
        full_config = y.load(f)

    config = full_config["execution"][collision_system]["dataset"]

    return config


def setup_parsl_587(nodes_to_allocate: int = 9, jobs_per_node: int = 2,  partition: str = "short", debug: bool = False, use_root: bool = False,)-> Config:
    """ Setup parsl for the 587 cluster.

    The configuration that is defined here is loaded by parsl. We also setup monitoring infrastructure.

    We default to allocating the entire node for simplicity. This helps address any possible memory issues.

    Args:
        partition: Partition to use with slurm. Default: "short".
        debug: If True, enable debugging on the workers. Default: False.
        use_root: If True, we intend to use ROOT in the worker jobs. In that case, we need to
            initialize the worker environment. Default: False.
        jobs_per_node: Number of jobs to run per node. Default: 2.
        nodes_to_allocate: Number of nodes to allocate. Default: 9.
    Returns:
        The parsl configuration for the 587 cluster. Note that it's already been loaded by parsl.
    """
    # Explanation of the parsl config:
    # - Number of blocks is the number of nodes * nodes_per_block. We want one node per block
    #   so we can use blocks as a proxy for nodes.
    #   - Control the number of nodes via init_blocks and max_blocks (we don't use any elasticity).
    # - If we want, for example, two jobs per node (one core per job), we need to set:
    #   - max_workers = 2
    #   - cores_per_node = 2
    # NOTE: If we just try to scale with nodes_per_block, we'll allocate the appropriate nodes,
    #       but then all the jobs will be on just one node, which is definitely not what we want.

    # Setup ROOT if necessary
    slurm_kwargs = {}
    if use_root:
        slurm_kwargs.update(
            dict(
                worker_init="eval `/usr/local/bin/alienv -w /software/rehlers/alice/sw --no-refresh printenv AliPhysics/latest`",
            )
        )

    b587_executor = Config(
        executors=[
            HighThroughputExecutor(
                label="b587",
                worker_debug=debug,
                # Ensures two jobs per job.
                max_workers=jobs_per_node,
                provider=SlurmProvider(
                    partition=partition,
                    # Submitting from pc059, so we have local communication available.
                    channel=LocalChannel(),
                    # This is effectively how many sets of nodes to allocate (assuming nodes_per_block = 1).
                    # See the description above.
                    init_blocks=nodes_to_allocate,
                    max_blocks=nodes_to_allocate,
                    # This is how many jobs to put into a particular block.
                    nodes_per_block=1,
                    # Usually we want one core per task, so we want cores_per_node = max_workers.
                    # NOTE: This must be set. Otherwise, the executor will assume that all cores
                    #       are available on a given node.
                    cores_per_node=jobs_per_node,
                    # Ensures that the jobs are spread out. Also means that we need to be careful
                    # when others are running to avoid running out of memory.
                    exclusive=True,
                    # Format: HH:MM:SS, so we request one hour
                    walltime="01:00:00",
                    # pc051 has too much in swap right now to be useful, so let's just skip and avoid the problems.
                    # pc147 also struggles and we don't want that to come down because it's one of the ceph quorum machines...
                    # pc075 also has high swap right now...
                    scheduler_options="#SBATCH --exclude=pc051,pc147,pc075",
                    # For root
                    **slurm_kwargs,
                ),
            )
        ],
        # Monitoring information
        monitoring=MonitoringHub(
            hub_address=address_by_hostname(),
            hub_port=55055,
            monitoring_debug=False,
            resource_monitoring_interval=10,
        ),
        # Disables resource scaling.
        strategy=None,
    )
    parsl.load(b587_executor)

    return b587_executor


def _determine_number_of_entries_per_file(filenames: Sequence[Path], tree_name: str) -> Dict[Path, int]:
    """ Retrieve the number of tree entries per ROOT file.

    Args:
        filenames: Filenames containing the trees of interest.
        tree_name: Name of the tree.
    Returns:
        Map from the filename to the number of entries.
    """
    number_of_entries_per_file: Dict[Path, int] = {}

    for filename in filenames:
        with uproot.open(filename) as f:
            number_of_entries_per_file[filename] = f[tree_name].num_entries

    return number_of_entries_per_file


def _number_of_entries_per_file(input_filenames: Sequence[Union[Path, str]], tree_name: str, collision_system: str, identifier: str, recreate: bool = False) -> Dict[Path, int]:
    """ Determine number of entries per file.

    Args:
        input_filenames: Filenames to use in the number of entries determination.
        tree_name: Name of the tree.
        collision_system: Collision system.
        identifier: Identifier for the dataset. Used to cache the number of entries per file.
        recreate: Force recreation of the number of entries per file for a dataset, skipping
            over the cache. Default: False.
    Returns:
        Mapping between file and number of entries in the file.
    """
    # Validation
    filenames = helpers.expand_wildcards_in_filenames(input_filenames)

    # Setup
    y = yaml.yaml()
    number_of_entries_file = Path(f"trains/{collision_system}/{identifier}.yaml")

    # We need the number of entries per file to be able to split up the jobs properly later.
    # If it doesn't exist, create it.
    if not number_of_entries_file.exists() or recreate:
        logger.debug("Need to get entries from the input files.")
        number_of_entries_per_file = _determine_number_of_entries_per_file(filenames=filenames, tree_name=tree_name)
        with open(number_of_entries_file, "w") as f:
            # Explicit iteration because we need to convert from Path to str.
            y.dump({str(k): v for k, v in number_of_entries_per_file.items()}, f)

    # Now we know that it exists, we can grab it.
    with open(number_of_entries_file, "r") as f:
        res = y.load(f)
        number_of_entries_per_file = {Path(k): v for k, v in res.items()}

    return number_of_entries_per_file


def _distribute_entries_to_jobs(number_of_entries_per_file: Mapping[Path, int], entries_per_job: int) -> Dict[Path, List[Tuple[int, int]]]:
    """ Distribute a specific number of entries to each job.

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
            splits.append([start, end])
            # Move up to the next iteration.
            start = end

        job_info[filename] = splits

    return job_info


@python_app
def _convert_to_parquet(tree_name: str, prefixes: Sequence[str], branches: Sequence[str], prefix_branches: Sequence[str],
                        event_range: Optional[Tuple[Optional[int], Optional[int]]] = None, inputs=[], outputs=[], stdout=None):
    """ Convert to parquet app. """
    from jet_substructure.base import new_methods
    from pathlib import Path
    res = new_methods.convert_tree_to_parquet(
        filename=Path(inputs[0].filepath), tree_name=tree_name, prefixes=prefixes,
        branches=branches,
        prefix_branches=prefix_branches,
        entries=event_range, output_filename=Path(outputs[0].filepath)
    )
    logger.debug(outputs)
    return res


def setup_convert_to_parquet(collision_system: str, entries_per_job: int = int(1e5)) -> List[AppFuture]:
    """ Setup convert_to_parquet app for execution with parsl.

    Args:
        events_per_job: Number of events to process in each job.
    Returns:
        List of `AppFuture` created when defining the jobs.
    """
    # Setup
    results = []
    dataset_config = read_config(collision_system=collision_system)

    # Based on the number of events desired per job, split the files into jobs with those number of events.
    number_of_entries_per_file = _number_of_entries_per_file(
        input_filenames=dataset_config["files"],
        tree_name=dataset_config["tree_name"],
        collision_system=collision_system,
        identifier=dataset_config["name"],
        #recreate=True,
    )
    job_ranges = _distribute_entries_to_jobs(
        number_of_entries_per_file=number_of_entries_per_file, entries_per_job=entries_per_job
    )
    #logger.debug(job_ranges)
    # Sanity check.
    logger.debug(f"Total entries: {sum([len(l) for l in job_ranges.values()])}")

    # Iterate over the file and event ranges, setting up an app for each entry.
    for i_file, (filename, event_ranges) in enumerate(job_ranges.items()):
        for i, event_range in enumerate(event_ranges):
            # Setup file IO
            parsl_input_file = File(str(filename))
            output_filename = Path(parsl_input_file.filepath)
            # Path becomes .../parquet/events_per_job_.../filename.00.parquet
            output_filename = output_filename.parent / "parquet" / f"events_per_job_{entries_per_job}" / output_filename.name
            output_filename = output_filename.with_suffix(f".{i:02}.parquet")
            parsl_output_file = File(str(output_filename))

            results.append(_convert_to_parquet(
                tree_name=dataset_config["tree_name"],
                prefixes=list(dataset_config["prefixes"].keys()),
                branches=dataset_config["branches"],
                prefix_branches=dataset_config["prefix_branches"],
                event_range=event_range,
                inputs=[parsl_input_file],
                outputs=[parsl_output_file],
                stdout=f"{i_file}_{i}.log"
            ))

    return results


@python_app
def _calculate_embedding_skim(dataset_config: Dict[str, Any], train_directory: Path, iterative_splittings: bool, inputs=[], outputs=[], stdout=None, stderr=None) -> AppFuture:
    """ Calculate embedding skim app. """
    import traceback
    from pathlib import Path
    from jet_substructure.analysis import new_skim_to_flat_tree
    try:
        res = new_skim_to_flat_tree.calculate_embedding_skim(
            input_filename=Path(inputs[0].filepath),
            iterative_splittings=iterative_splittings,
            prefixes=dataset_config["prefixes"],
            scale_factors=dataset_config["scale_factors"],
            train_directory=train_directory,
            jet_R=dataset_config["jet_R"],
            output_filename=Path(outputs[0].filepath),
        )
    except Exception as e:
        # Skip any problems for now
        logger.warning(e)
        res = traceback.format_exc()

    logger.debug(outputs)
    return res


def setup_calculate_embedding_skim(
    collision_system: str,
    entries_per_job: int,
    iterative_splittings: bool = True,
    selected_train_numbers: Optional[Sequence[int]] = None,
    input_files: Optional[Sequence[DataFuture]] = None,
) -> List[AppFuture]:
    """ Setup to calculate embedding skim.

    Args:
        collision_system: Collision system.
        entries_per_job: Number of entries per job.
        iterative_splittings: True if iterative splittings are selected rather than recursive splittings. Default: True.
        selected_train_numbers: Use only a selection of train numbers. Default: None. All train numbers are
            taken from the config.
        input_files: DataFuture from an AppFuture from a previous execution.
    Returns:
        List of `AppFuture` created when defining the jobs.
    """
    # Validation
    if selected_train_numbers is None:
        selected_train_numbers = []

    # Setup
    results = []
    dataset_config = read_config(collision_system=collision_system)
    train_directories = set([Path(filename).parent for filename in dataset_config["files"]])

    # If input files aren't passed, then we need to determine them ourselves.
    if input_files is None:
        input_files = []
        for train_directory in train_directories:
            # Select train numbers.
            if selected_train_numbers and int(train_directory.name) not in selected_train_numbers:
                logger.debug(f"Skipping train number {train_directory.name}")
                continue
            logger.info(f"Processing {train_directory.name}")

            # Then iterate over the directories.
            for filename in Path(f"{train_directory}/parquet/events_per_job_{entries_per_job}/").glob("*.parquet"):
                input_files.append(File(str(filename)))

    # Create the Apps.
    for parsl_input_file in input_files:
        # Setup
        iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
        # Setup file I/O
        output_dir = train_directory / "skim"
        output_filename = output_dir / f"{filename.stem}_{iterative_splittings_label}_splittings.root"
        parsl_output_file = File(str(output_filename))

        results.append(_calculate_embedding_skim(
            dataset_config=dataset_config,
            iterative_splittings=iterative_splittings,
            train_directory=train_directory,
            inputs=[parsl_input_file],
            outputs=[parsl_output_file],
            #stdout=parsl.AUTO_LOGNAME,
            #stderr=parsl.AUTO_LOGNAME,
        ))

    return results


@python_app
def _calculate_data_skim(collision_system: str, dataset_config: Dict[str, Any], iterative_splittings: bool, inputs=[], outputs=[], stdout=None, stderr=None) -> AppFuture:
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
            scale_factors=dataset_config.get("scale_factors", None),
        )
    except Exception as e:
        # Skip any problems for now
        logger.warning(e)
        res = traceback.format_exc()

    logger.debug(outputs)
    return res


def setup_calculate_data_skim(
    collision_system: str,
    entries_per_job: int,
    iterative_splittings: bool = True,
    selected_train_numbers: Optional[Sequence[int]] = None,
    input_files: Optional[Sequence[DataFuture]] = None,
) -> List[AppFuture]:
    """ Setup to calculate data skim.

    Args:
        collision_system: Collision system.
        entries_per_job: Number of entries per job.
        iterative_splittings: True if iterative splittings are selected rather than recursive splittings. Default: True.
        selected_train_numbers: Use only a selection of train numbers. Default: None. All train numbers are
            taken from the config.
        input_files: DataFuture from an AppFuture from a previous execution.
    Returns:
        List of `AppFuture` created when defining the jobs.
    """
    # Validation
    if selected_train_numbers is None:
        selected_train_numbers = []

    # Setup
    results = []
    dataset_config = read_config(collision_system=collision_system)
    train_directories = set([Path(filename).parent for filename in dataset_config["files"]])

    # If input files aren't passed, then we need to determine them ourselves.
    if input_files is None:
        input_files = []
        for train_directory in train_directories:
            # Select train numbers.
            if selected_train_numbers and int(train_directory.name) not in selected_train_numbers:
                logger.debug(f"Skipping train number {train_directory.name}")
                continue
            logger.info(f"Processing {train_directory.name}")

            # Then iterate over the directories.
            for filename in Path(f"{train_directory}/parquet/events_per_job_{entries_per_job}/").glob("*.parquet"):
                input_files.append(File(str(filename)))

    # Create the Apps.
    for parsl_input_file in input_files:
        # Setup
        iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
        # Setup file I/O
        output_dir = train_directory / "skim"
        output_filename = output_dir / f"{filename.stem}_{iterative_splittings_label}_splittings.root"
        parsl_output_file = File(str(output_filename))
        results.append(_calculate_data_skim(
            dataset_config=dataset_config,
            collision_system=collision_system,
            iterative_splittings=iterative_splittings,
            inputs=[parsl_input_file],
            outputs=[parsl_output_file],
            #stdout=parsl.AUTO_LOGNAME,
            #stderr=parsl.AUTO_LOGNAME,
        ))

    return results


if __name__ == "__main__":
    # Settings
    collision_system = "PbPb"
    entries_per_job = int(1e5)

    # Setup parsl
    setup_parsl_587(
        nodes_to_allocate=9,
        jobs_per_node=2,
    )

    # Setup logging. By doing it after parsl, we're able to keep it much quieter.
    # Oddly, I can't seem to select the parsl modules to change their loggers, so this seems
    # to be the only reasonable way to configure logging.
    helpers.setup_logging(logging.INFO)
    # Quiet down parsl
    logging.getLogger("parsl").setLevel(logging.WARNING)

    results = []
    # results = setup_convert_to_parquet(
    #     collision_system=collision_system,
    #     entries_per_job=entries_per_job,
    # )
    # results = setup_calculate_embedding_skim(
    #     collision_system=collision_system,
    #     entries_per_job=entries_per_job,
    #     selected_train_numbers=list(range(5966, 5967)),
    #     input_files=[r.outputs[0] for r in results] if results else None,
    # )
    # results = setup_calculate_data_skim(
    #     collision_system=collision_system,
    #     entries_per_job=entries_per_job,
    #     input_files=[r.outputs[0] for r in results] if results else None,
    # )

    logger.info(f"About to ask for result. len: {len(results)}")
    # Wait on results
    # print each job status, initially all are running
    #print ("Job Status: {}".format([r.done() for r in results]))
    # Wait for all apps to complete
    res = [r.result() for r in results]
    logger.info(res)

    logger.info("Done")

