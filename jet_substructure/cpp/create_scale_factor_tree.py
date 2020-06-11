""" Create friend hists with scale factors.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path

import numpy as np
import uproot
from pachyderm import binned_data

from jet_substructure.base import data_manager, helpers


logger = logging.getLogger(__name__)


def scale_factor_tree(filename: Path) -> None:
    # Setup
    input_file = uproot.open(filename)

    # Retrieve the embedding helper to extract the cross section and ntrials.
    embedding_hists = input_file["AliAnalysisTaskEmcalEmbeddingHelper_histos"]
    h_cross_section_uproot = [h for h in embedding_hists if hasattr(h, "name") and h.name == b"fHistXsection"][0]
    h_cross_section = binned_data.BinnedData.from_existing_data(h_cross_section_uproot)
    h_n_trials = binned_data.BinnedData.from_existing_data(
        [h for h in embedding_hists if hasattr(h, "name") and h.name == b"fHistTrials"][0]
    )
    # Find the first non-zero values bin.
    # argmax will return the index of the first instance of True.
    pt_hard_bin = (h_cross_section.values != 0).argmax(axis=0)

    # The cross section is a profile hist, but we just read the raw values with uproot + binned_data. Consequently, the values
    # aren't scaled down by the number of entries in that bin (as already performed by ROOT), so we just take the
    # cross section / n_trials
    scale_factor = h_cross_section.values[pt_hard_bin] / h_n_trials.values[pt_hard_bin]
    logger.debug(f"Scale factor: {scale_factor}")

    # Get number of entries in the tree to determine
    n_entries = uproot.tree.numentries(
        str(filename),
        "AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
    )
    logger.debug(f"n entries: {n_entries}")

    output_filename = filename.parent / "scale_factor" / filename.name
    output_filename.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Writing scale_factor to {output_filename}")
    branches = {"scale_factor": np.float32}
    with uproot.recreate(output_filename) as output_file:
        output_file["tree"] = uproot.newtree(branches)
        # Write all of the calculations
        output_file["tree"].extend({"scale_factor" : scale_factor * np.ones(n_entries)})


if __name__ == "__main__":
    helpers.setup_logging()

    base_path = Path("trains/embedPythia/{train_number}/AnalysisResults.*.root")
    filenames = data_manager._ensure_and_expand_paths(
        [Path(str(base_path).format(train_number=train_number)) for train_number in range(5988, 6008)]
    )
    for filename in filenames:
        scale_factor_tree(filename=filename)
