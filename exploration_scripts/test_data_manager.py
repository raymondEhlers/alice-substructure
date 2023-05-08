#!/usr/bin/env python3

""" Test the capabilities of the data manager.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging

import coloredlogs
import IPython

from jet_substructure.base import data_manager, substructure_methods


logger = logging.getLogger(__name__)


def test_setup() -> None:
    prefix = "data"
    branches = [
        f"{prefix}.fJetPt",
        f"{prefix}.fJetConstituents.fPt",
        f"{prefix}.fJetConstituents.fEta",
        f"{prefix}.fJetConstituents.fPhi",
        f"{prefix}.fJetConstituents.fGlobalIndex",
        f"{prefix}.fJetSplittings.fKt",
        f"{prefix}.fJetSplittings.fDeltaR",
        f"{prefix}.fJetSplittings.fZ",
        f"{prefix}.fJetSplittings.fParentIndex",
        f"{prefix}.fSubjets.fPartOfIterativeSplitting",
        f"{prefix}.fSubjets.fSplittingNodeIndex",
        f"{prefix}.fSubjets.fConstituentIndices",
        # f"{prefix}.fSubjets.fConstituentJaggedIndices",
    ]
    dm = data_manager.IterateTrees(
        filenames=["../trains/pythia/2110/gridTestTrain/AnalysisResults.root"],
        tree_name="AliAnalysisTaskJetDynamicalGrooming_Jet_AKTChargedR040_tracks_pT0150_E_scheme_responseTree_PythiaDef_NoSub_Incl",
        branches=branches,
    )

    logger.info("About to iterate")
    for tree in dm.data_for_analysis():
        substructure_methods.SubstructureJetArray.from_tree(tree, prefix="data")

        IPython.embed()
        logger.debug(tree)


if __name__ == "__main__":
    # Basic setup
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s")
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)

    test_setup()
