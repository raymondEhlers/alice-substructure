""" Adapt from the track skim to the existing code base.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Mapping, Optional, Union

import awkward as ak
import numpy as np
from mammoth.hardest_kt import analysis_alice

from jet_substructure.analysis import new_skim_to_flat_tree
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


def _hardest_kt_data_skim(
    jets: ak.Array,
    input_filename: Path,
    collision_system: str,
    jet_R: float,
    iterative_splittings: bool,
    convert_data_format_prefixes: Mapping[str, str],
    output_filename: Path,
    scale_factors: Optional[Mapping[int, float]] = None,
    pt_hat_bin: Optional[int] = -1,
) -> None:
    """Implementation of the hardest kt data skim.

    Supports pp, pythia, PbPb, and embedded pythia. The data and jet finding needs to be
    handled in a separate function.
    """
    # Now, adapt into the expected format.
    all_jets = {
        convert_data_format_prefixes[k]: ak.zip(
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
        for k in convert_data_format_prefixes
    }

    scale_factors = None
    prefixes = {"data": "data"}
    if collision_system == "pythia":
        # Store externally provided pt hard bin
        all_jets["pt_hard_bin"] = np.ones(len(all_jets["data"]["jet_pt"])) * pt_hat_bin
        # Add the second prefix for true jets
        prefixes["true"] = "true"

    new_skim_to_flat_tree.calculate_data_skim_impl(
        all_jets=all_jets,
        input_filename=input_filename,
        collision_system=collision_system,
        iterative_splittings=iterative_splittings,
        prefixes=prefixes,
        jet_R=jet_R,
        output_filename=output_filename,
        scale_factors=scale_factors,
    )


def hardest_kt_data_skim(
    input_filename: Path,
    collision_system: str,
    jet_R: float,
    min_jet_pt: Union[float, Mapping[str, float]],
    iterative_splittings: bool,
    output_filename: Path,
    convert_data_format_prefixes: Mapping[str, str],
    event_activity: str = "",
    # Data specific
    loading_data_rename_prefix: Optional[Mapping[str, str]] = None,
    # Pythia specific
    pt_hat_bin: Optional[int] = -1,
    scale_factors: Optional[Mapping[int, float]] = None,
) -> None:
    if loading_data_rename_prefix is None:
        loading_data_rename_prefix = {"data": "data"}

    if len(convert_data_format_prefixes) == 1:
        assert not isinstance(min_jet_pt, dict)  # help out mypy

        jets = analysis_alice.analysis_data(
            collision_system=collision_system,
            arrays=analysis_alice.load_data(
                filename=input_filename,
                collision_system=collision_system if not event_activity else f"{collision_system}_{event_activity}",
                rename_prefix=loading_data_rename_prefix,
            ),
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
        )
    elif collision_system == "pythia":
        assert isinstance(min_jet_pt, dict)  # help out mypy
        jets = analysis_alice.analysis_MC(
            arrays=analysis_alice.load_MC(filename=input_filename, collision_system=collision_system),
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
        )
    else:
        raise NotImplementedError(f"Not yet implemented for {collision_system}...")

    _hardest_kt_data_skim(
        jets=jets,
        input_filename=input_filename,
        collision_system=collision_system,
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
        convert_data_format_prefixes=convert_data_format_prefixes,
        output_filename=output_filename,
        pt_hat_bin=pt_hat_bin,
        scale_factors=scale_factors,
    )


if __name__ == "__main__":
    helpers.setup_logging(logging.INFO)

    collision_system = "PbPb"
    base_path = Path(f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}")
    hardest_kt_data_skim(
        input_filename=base_path / "AnalysisResults_track_skim.parquet",
        collision_system=collision_system,
        jet_R=0.4,
        min_jet_pt=5 if collision_system == "pp" else 20,
        iterative_splittings=True,
        convert_data_format_prefixes={"data": "data"},
        output_filename=base_path / "skim" / "skim_output.root",
    )
