#!/usr/bin/env python3

""" Analyze the dynamical grooming tree.

.. codeauthor:: Raymnod Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import List, Sequence

#import pandas as pd
#import numpy as np
import coloredlogs
import uproot

#from pachyderm import histogram
from pachyderm import yaml

from jet_substructure.base import helpers
from jet_substructure.analysis import substructure
from jet_substructure.analysis import plot_results

logger = logging.getLogger(__name__)

def analyze_tree(filenames: Sequence[Path]) -> List[helpers.TTree]:
    trees = []
    for filename in filenames:
        trees.append(helpers.get_tree(filename = filename, name = "AliAnalysisTaskJetDynamicalGrooming_RawTree_Data_ConstSub_Incl"))

    return trees

def run(collision_system: str, dataset_config_filename: Path, output: Path) -> None:
    # Setup
    jet_pt_bins = [
        helpers.RangeSelector(min = 60, max = 80),
        helpers.RangeSelector(min = 80, max = 100),
        helpers.RangeSelector(min = 100, max = 120),
        # Most likely where we will actually measure.
        helpers.RangeSelector(min = 80, max = 120),
    ]
    output.mkdir(parents=True, exist_ok=True)
    # Configuration
    y = yaml.yaml()
    with open(dataset_config_filename, "r") as f:
        datasets_config = y.load(f)
    dataset_name = datasets_config["datasets"][collision_system]["dataset"]["name"]

    # Retrieve and normalize data
    #uproot_cache = uproot.ArrayCache("500 MB")
    #arrays = helpers.load_data(filenames = tree_filenames, cache = uproot_cache)
    arrays = helpers.load_data(collision_system=collision_system,
                               dataset_config=datasets_config["datasets"][collision_system]["dataset"],
                               available_datasets_config=datasets_config["available_datasets"])

    # Normalize the array keys.
    arrays = helpers.normalize_array_names(arrays)

    # Calculate the substructure variables
    res = substructure.calculate_substructure_variables(arrays, R = 0.4, prefix = "data")
    dynamical_z, dynamical_kt, dynamical_time, soft_drop, leading_kt, leading_kt_hard_cutoff = res

    plot_results.kt(results = res, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)
    plot_results.z(results = res, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)
    plot_results.delta_R(results = res, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)
    plot_results.theta(results = res, jet_R=0.4, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)
    plot_results.splitting_number(results = res, jet_pt=arrays["data_jetPt"], jet_pt_bins = jet_pt_bins, path = output)


if __name__ == "__main__":
    # Basic setup
    coloredlogs.install(
        level = logging.DEBUG,
        fmt = "%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
    )
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Setup
    collision_system = "PbPb"
    train_number = 5441
    base_filename = Path("trains") / collision_system / str(train_number)
    run(
        collision_system = collision_system,
        dataset_config_filename = Path("config") / "datasets.yaml",
        output = Path("output") / collision_system / str(train_number),
    )
