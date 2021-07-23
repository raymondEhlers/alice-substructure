""" Adapt from the track skim to the existing code base.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path

import awkward as ak
import numpy as np
from mammoth.framework.analysis import kt_pythia

from jet_substructure.analysis import new_skim_to_flat_tree, parsl
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


def pythia_kt(
    input_filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet"),
    jet_R=0.4,
) -> None:
    jets = kt_pythia.analysis(pythia_filename=input_filename, jet_R=jet_R, min_pythia_jet_pt=20)
    # Now, adapt into the expected format.
    _rename_map = {
        "part_level": "true",
        "det_level": "data",
    }
    all_jets = {
        _rename_map[k]: ak.zip(
            {
                "jet_pt": jets[k].pt,
                "jet_constituents": ak.zip(
                    {
                        "pt": jets[k].constituents.pt,
                        "eta": jets[k].constituents.eta,
                        "phi": jets[k].constituents.pt,
                        "id": jets[k].constituents.index,
                    },
                    with_name="JetConstituent",
                ),
                "jet_splittings": ak.Array(
                    jets[k, "reclustering", "jet_splittings"],
                    with_name="JetSplitting",
                ),
                "subjets": ak.zip(
                    {
                        "part_of_iterative_splitting": jets[
                            k, "reclustering", "subjets", "part_of_iterative_splitting"
                        ],
                        "parent_splitting_index": jets[k, "reclustering", "subjets", "splitting_node_index"],
                        "constituent_indices": jets[k, "reclustering", "subjets", "constituent_indices"],
                    },
                    with_name="Subjet",
                    # We want to apply the behavior for each jet, and then for each subjet
                    # in the jet, so we use a depth limit of 2.
                    depth_limit=2,
                ),
            },
            depth_limit=1,
        )
        for k in _rename_map
    }
    # TODO: Will need to provide the pt hard bin externally because the files
    #       are segmented by pt hard bin
    all_jets["pt_hard_bin"] = np.ones(ak.count(all_jets["true"]["jet_pt"])) * 12

    scale_factors = parsl.read_extracted_scale_factors(
        collision_system="pythia", dataset_name="LHC18b8_pythia_R04_2520"
    )

    new_skim_to_flat_tree.calculate_data_skim_impl(
        all_jets=all_jets,
        input_filename=input_filename,
        collision_system="pythia",
        iterative_splittings=True,
        prefixes={
            "data": "data",
            "true": "true",
        },
        jet_R=jet_R,
        output_filename=input_filename.parent / "skim" / "skim_output.root",
        scale_factors=dict(scale_factors),
    )


if __name__ == "__main__":
    helpers.setup_logging(logging.INFO)
    pythia_kt()
