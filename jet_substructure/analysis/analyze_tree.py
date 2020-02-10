#!/usr/bin/env python3

""" Analyze the dynamical grooming tree.

.. codeauthor:: Raymnod Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path
from typing import Sequence

import uproot

from jet_substructure.base import helpers

def analyze_tree(filenames: Sequence[Path]) -> bool:
    trees = []
    for filename in filenames:
        trees.append(helpers.get_tree(filename = filename, name = "AliAnalysisTaskJetDynamicalGrooming_RawTree_Data_ConstSub_Incl"))

    return trees

if __name__ == "__main__":
    base_filename = Path("trains") / "PbPb" / "5413"
    analyze_tree(
        filenames = [
            base_filename / "AnalysisResults.LHC18q.root",
            base_filename / "AnalysisResults.LHC18r.root",
        ]
    )
