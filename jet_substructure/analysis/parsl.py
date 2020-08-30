#!/usr/bin/env python3

"""" Submit analysis using parsl.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import uproot4 as uproot
#import uproot as uproot
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


if __name__ == "__main__":
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
    print(res)
    print(sum(res.values()))

    job_ranges = distribute_jobs(number_of_entries_per_file=res, number_per_job=500_000)

    print(job_ranges)
    print(sum([len(l) for l in job_ranges.values()]))

