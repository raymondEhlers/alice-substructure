#!/usr/bin/env python3

"""" Submit analysis using parsl.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import parsl
import uproot4 as uproot
#import uproot as uproot
from parsl.app.app import python_app
from parsl.data_provider.files import File
from pachyderm import yaml

from jet_substructure.base import helpers

logger = logging.getLogger(__name__)


def number_of_entries(filenames: Sequence[Path], tree_name: str) -> Dict[Path, int]:
    number_of_entries_map: Dict[Path, int] = {}

    for filename in filenames:
        print(filename)
        # uproot4
        with uproot.open(filename) as f:
            number_of_entries_map[filename] = f[tree_name].num_entries
        # uproot3
        #number_of_entries_map[filename] = uproot.numentries(str(filename), tree_name)

    return number_of_entries_map

def setup(input_filenames: Sequence[Union[Path, str]], tree_name: str, collision_system:str, identifier: str, recreate: bool = False) -> Dict[Path, int]:
    filenames = helpers.expand_wildcards_in_filenames(input_filenames)

    # Setup
    number_of_entries_file = Path(f"trains/{collision_system}/{identifier}.yaml")
    y = yaml.yaml()

    # We need the number of entries per file to be able to split up jobs properly.
    # If it doesn't exist, create it.
    if not number_of_entries_file.exists() or recreate:
        logger.debug("Need to get entries from the input files.")
        number_of_entries_per_file = number_of_entries(filenames=filenames, tree_name=tree_name)
        with open(number_of_entries_file, "w") as f:
            y.dump({str(k): v for k, v in number_of_entries_per_file.items()}, f)

    # Now we know that it exists, we can grab it.
    with open(number_of_entries_file, "r") as f:
        res = y.load(f)
        number_of_entries_per_file = {Path(k): v for k, v in res.items()}

    return number_of_entries_per_file

def distribute_jobs(number_of_entries_per_file: Mapping[Path, int], number_per_job: int) -> Dict[Path, List[Tuple[int, int]]]:
    job_info = {}

    for filename, number_of_entries in number_of_entries_per_file.items():
        splits = []
        start = 0
        continue_iterating = True
        while continue_iterating:
            end = start + number_per_job
            # Ensure that we never ask for more entries than are in the file.
            if start + number_per_job > number_of_entries:
                end = number_of_entries
                continue_iterating = False
            # Store the start and stop for convenience.
            splits.append([start, end])
            # Move up to the next iteration.
            start = end

        job_info[filename] = splits

    return job_info

@python_app
def convert_to_parquet(tree_name: str, prefixes: Sequence[str], event_range: Optional[Tuple[Optional[int], Optional[int]]] = None, inputs=[], outputs=[], stdout=None):
    from jet_substructure.base import new_methods
    from pathlib import Path
    res = new_methods.convert_tree_to_parquet(filename=Path(inputs[0].filepath), tree_name=tree_name, prefixes=prefixes, entries=event_range, output_filename=Path(outputs[0].filepath))
    print(outputs)
    return res

def run(events_per_job: int):
    collision_system = "embedPythia"

    y = yaml.yaml()

    with open("config/new_config.yaml", "r") as f:
        full_config = y.load(f)
    base_dir = Path(full_config["base_directory"])
    config = full_config["execution"][collision_system]["dataset"]

    res = setup(
        input_filenames=config["files"],
        tree_name=config["tree_name"],
        collision_system=collision_system,
        identifier=config["name"],
        #recreate=True,
    )
    #print(res)
    #print(sum(res.values()))

    #job_ranges = distribute_jobs(number_of_entries_per_file=res, number_per_job=500_000)
    job_ranges = distribute_jobs(number_of_entries_per_file=res, number_per_job=1e5)

    #print(job_ranges)
    print(sum([len(l) for l in job_ranges.values()]))

    return res, job_ranges

if __name__ == "__main__":
    # Setup a test...
    # Setup parsl executor
    from parsl.providers import SlurmProvider
    from parsl.channels import LocalChannel
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    # Explanation:
    # - Number of blocks is the number of nodes * nodes_per_block. We want one node per block
    #   so we can use blocks as a proxy for nodes.
    #   - Control the number of nodes via init_blocks and max_blocks (we don't use an elasticity).
    # - If we want, for example, two jobs per node (one core per job), we need to set:
    #   - max_workers = 2
    #   - cores_per_node = 2
    # NOTE: If we just try to scale with nodes_per_block, we'll allocate the appropriate nodes,
    #       but then all the jobs will be on just one node, which is definitely not what we want.
    jobs_per_node = 2
    nodes_to_allocate = 10
    b587_executor = Config(
        executors=[
            HighThroughputExecutor(
                label="b587",
                worker_debug=True,
                # Ensures two jobs per job.
                max_workers=jobs_per_node,
                provider=SlurmProvider(
                    partition="short",
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
                    walltime="01:30:00",
                    # pc051 has too much in swap right now to be useful, so let's just skip and
                    # avoid the problems.
                    scheduler_options="#SBATCH --exclude=pc051",
                ),
            )
        ],
        # Disables resource scaling.
        strategy=None,
    )
    parsl.load(b587_executor)

    events_per_job = int(1e5)
    entries_per_file, job_ranges = run(events_per_job)

    # Setup single input files
    results = []
    #for filename in list(entries_per_file.keys())[:10]:
    #    ...

    for i_file, (filename, event_ranges) in enumerate(job_ranges.items()):
        # TEMP
        #if i_file > 20:
        #    break
        # ENDTEMP

        for i, event_range in enumerate(event_ranges):
            parsl_input_file = File(str(filename))
            #output_filename = str(Path(parsl_input_file.filepath).with_suffix("")) + f"_{event_range[0]}_{event_range[1]}"
            output_filename = Path(parsl_input_file.filepath)
            output_filename = output_filename.parent / "parquet" / f"events_per_job_{events_per_job}" / output_filename.name
            output_filename = output_filename.with_suffix(f".{i:02}.parquet")
            #output_filename = Path(str(output_filename) + f".{i:02}.parquet")
            #print(f"i: {i}, output_filename: {output_filename}")
            parsl_output_file = File(str(output_filename))

            results.append(convert_to_parquet(
                tree_name="AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
                prefixes=["data", "matched", "detLevel"],
                event_range=event_range,
                inputs=[parsl_input_file],
                outputs=[parsl_output_file],
                stdout=f"{i_file}.log"
            ))


    print(f"About to ask for result. len: {len(results)}")
    # Wait on results
    # print each job status, initially all are running
    #print ("Job Status: {}".format([r.done() for r in results]))
    # wait for all apps to complete
    res = [r.result() for r in results]

    print("Done")

