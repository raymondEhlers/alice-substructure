""" Adapt from the track skim to the existing code base.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path

import awkward as ak
import numpy as np
from mammoth.hardest_kt import analysis_alice

from jet_substructure.analysis import new_skim_to_flat_tree, parsl
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


def hardest_kt_MC(
    input_filename: Path = Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet"),
    jet_R: float = 0.4,
) -> None:
    jets = analysis_alice.analysis_MC(
        arrays=analysis_alice.load_MC(filename=input_filename),
        jet_R=jet_R,
        min_jet_pt={
            "det_level": 20,
        },
    )

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


def hardest_kt_data(
    input_filename: Path,
    collision_system: str,
    jet_R: float,
    iterative_splittings: bool,
    output_filename: Path,
    min_jet_pt: float = 20,
    event_activity: str = "",
) -> None:
    jets = analysis_alice.analysis_data(
        collision_system=collision_system,
        arrays=analysis_alice.load_data(
            filename=input_filename,
            collision_system=collision_system if not event_activity else f"{collision_system}_{event_activity}",
            rename_prefix={"data": "data"} if collision_system != "pythia" else {"data": "det_level"},
        ),
        jet_R=jet_R,
        min_jet_pt=min_jet_pt,
    )

    # Now, adapt into the expected format.
    _rename_map = {
        "data": "data",
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
    # all_jets["pt_hard_bin"] = np.ones(ak.count(all_jets["data"]["jet_pt"])) * 12

    # scale_factors = parsl.read_extracted_scale_factors(
    #    collision_system="pythia", dataset_name="LHC18b8_pythia_R04_2520"
    # )

    new_skim_to_flat_tree.calculate_data_skim_impl(
        all_jets=all_jets,
        input_filename=input_filename,
        collision_system=collision_system,
        iterative_splittings=iterative_splittings,
        prefixes={
            "data": "data",
        },
        jet_R=jet_R,
        output_filename=output_filename,
        # output_filename=input_filename.parent / "skim" / "skim_output.root",
        # scale_factors=scale_factors,
    )


if __name__ == "__main__":
    helpers.setup_logging(logging.INFO)

    collision_system = "PbPb"
    # if collision_system != "pythia":
    if True:
        base_path = Path(f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}")
        hardest_kt_data(
            collision_system=collision_system,
            input_filename=base_path / "AnalysisResults_track_skim.parquet",
            output_filename=base_path / "skim" / "skim_output.root",
            jet_R=0.4,
            iterative_splittings=True,
            min_jet_pt=5 if collision_system == "pp" else 20,
        )
    else:
        hardest_kt_MC(
            input_filename=Path(
                f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults_track_skim.parquet"
            ),
            jet_R=0.4,
        )
