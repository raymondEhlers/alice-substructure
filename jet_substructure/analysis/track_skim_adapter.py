""" Adapt from the track skim to the existing code base.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import awkward as ak
import numpy as np
from mammoth import helpers
from mammoth.framework import sources
from mammoth.hardest_kt import analysis_alice

from jet_substructure.analysis import new_skim_to_flat_tree


logger = logging.getLogger(__name__)

def _convert_analyzed_jets_to_all_jets_for_skim(
    jets: ak.Array,
    convert_data_format_prefixes: Mapping[str, str],
) -> Dict[str, ak.Array]:
    return {
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
    all_jets = _convert_analyzed_jets_to_all_jets_for_skim(
        jets=jets, convert_data_format_prefixes=convert_data_format_prefixes,
    )

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
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    output_filename: Path,
    convert_data_format_prefixes: Mapping[str, str],
    event_activity: str = "",
    # Data specific
    loading_data_rename_prefix: Optional[Mapping[str, str]] = None,
    # Pythia specific
    pt_hat_bin: Optional[int] = -1,
    scale_factors: Optional[Mapping[int, float]] = None,
) -> Tuple[bool, str]:
    if loading_data_rename_prefix is None:
        loading_data_rename_prefix = {"data": "data"}

    # Try to bail out early to avoid reprocessing if possible.
    if output_filename.exists():
        import uproot

        try:
            with uproot.open(output_filename) as f:
                # If the tree exists, can be read, and has more than 0 entries, we should be good
                if f["tree"].num_entries > 0:
                    # Return immediately to indicate that we're done.
                    return (True, f"already processed for {collision_system}, R={jet_R}, input: \"{input_filename}\", output: \"{output_filename}\"")
        except Exception:
            # If it fails for some reason, give up - we want to try again
            pass

    if collision_system in ["pp", "PbPb"]:
        jets = analysis_alice.analysis_data(
            collision_system=collision_system,
            arrays=analysis_alice.load_data(
                filename=input_filename,
                collision_system=collision_system,
                rename_prefix=loading_data_rename_prefix,
            ),
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
        )
    elif collision_system in ["pythia"]:
        # Although we could in principle analyze the MC loading only particle or detector level alone,
        # it's more consistent to analyze it with the data quality conditions applied on both part
        # and det level.
        # (ie. we want to analyze in exactly the same as would provided by the substructure analysis task)
        jets = analysis_alice.analysis_MC(
            arrays=analysis_alice.load_data(
                filename=input_filename,
                collision_system=collision_system,
                rename_prefix=loading_data_rename_prefix,
            ),
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

    return (True, f"success for {collision_system}, R={jet_R}, {input_filename}")


def _hardest_kt_embedding_skim(
    jets: ak.Array,
    input_filename: Path,
    jet_R: float,
    iterative_splittings: bool,
    scale_factor: float,
    convert_data_format_prefixes: Mapping[str, str],
    output_filename: Path,
) -> None:
    # Now, adapt into the expected format.
    all_jets = _convert_analyzed_jets_to_all_jets_for_skim(
        jets=jets, convert_data_format_prefixes=convert_data_format_prefixes,
    )

    # For the thermal model.
    # TODO: Probably should be an argument for embedding, but can start with this for the thermal model
    prefixes = {
        "hybrid": "hybrid",
        #"part_level": "part_level",
        "true": "true",
        "det_level": "det_level",
    }

    new_skim_to_flat_tree.calculate_embedding_skim_impl(
        all_jets=all_jets,
        input_filename=input_filename,
        iterative_splittings=iterative_splittings,
        prefixes=prefixes,
        scale_factor=scale_factor,
        jet_R=jet_R,
        output_filename=output_filename,
    )

def hardest_kt_embedding_skim(
    input_filename: Path,
    jet_R: float,
    min_jet_pt: Union[float, Mapping[str, float]],
    iterative_splittings: bool,
    output_filename: Path,
    thermal_model_parameters: sources.ThermalModelParameters,
    convert_data_format_prefixes: Mapping[str, str],
    scale_factor: float,
    r_max: float,
) -> Tuple[bool, str]:
    # TODO: Remove hard code...
    collision_system = "thermal_model"

    # Setup
    empty_filename = output_filename.with_suffix(".empty")

    # Try to bail out early to avoid reprocessing if possible.
    if empty_filename.exists():
        # It will be empty, so there's nothing to check. Just return
        return (True, f"Done - no jets to recluster for {collision_system}, R={jet_R}, {input_filename}")

    if output_filename.exists():
        import uproot

        try:
            with uproot.open(output_filename) as f:
                # If the tree exists, can be read, and has more than 0 entries, we should be good
                if f["tree"].num_entries > 0:
                    # Return immediately to indicate that we're done.
                    return (True, f"already processed for {collision_system}, R={jet_R}, {input_filename}")
        except Exception:
            # If it fails for some reason, give up - we want to try again
            pass

    if True:
        jets = analysis_alice.analysis_embedding(
            *analysis_alice.load_thermal_model(
                signal_filename=input_filename,
                thermal_model_parameters=thermal_model_parameters,
            ),
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            r_max=r_max,
        )
    else:
        raise NotImplementedError(
            #f"Not yet implemented for {collision_system}..."
            "Not yet implemented..."
        )

    # There were no jets. Note that with a specially crafted empty file
    if len(jets) == 0:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        empty_filename.touch()
        return (True, f"Done - no jets to recluster, so not trying to skim for {collision_system}, R={jet_R}, {input_filename}")

    _hardest_kt_embedding_skim(
        jets=jets,
        input_filename=input_filename,
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
        scale_factor=scale_factor,
        convert_data_format_prefixes=convert_data_format_prefixes,
        output_filename=output_filename,
    )

    return (True, f"success for {collision_system}, R={jet_R}, {input_filename}")


if __name__ == "__main__":
    helpers.setup_logging(level=logging.INFO)

    #for collision_system in ["pp", "PbPb"]:
    for collision_system in ["pp", "pythia", "PbPb"]:
        logger.info(f"Analyzing \"{collision_system}\"")
        base_path = Path(f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}")
        _min_jet_pt = {
            "pp": 5.,
            "pythia": {"det_level": 20.},
            "PbPb": 20.,
        }
        result = hardest_kt_data_skim(
            input_filename=base_path / "AnalysisResults_track_skim.parquet",
            collision_system=collision_system,
            jet_R=0.4,
            min_jet_pt=_min_jet_pt[collision_system],  # type: ignore
            iterative_splittings=True,
            convert_data_format_prefixes={"data": "data"} if collision_system != "pythia" else {"det_level": "data", "part_level": "true"},
            loading_data_rename_prefix={"data": "data"} if collision_system != "pythia" else {"data": "det_level"},
            output_filename=base_path / "skim" / "skim_output.root",
        )
        logger.info(f"Result: {result}")

    import jet_substructure.analysis.parsl
    scale_factors = jet_substructure.analysis.parsl.read_extracted_scale_factors(
        # TODO: Unclear if the collision system should be hard coded
        collision_system="embedPythia",
        dataset_name="LHC20g4_embedded_into_LHC18qr_central_R02_6982_7001",
    )

    base_path = Path(f"/software/rehlers/dev/substructure/trains/pythia/641")
    hardest_kt_embedding_skim(
        #input_filename=base_path / "run_by_run/LHC20g4/295612/11/AnalysisResults.20g4.016.root",
        input_filename=base_path / "run_by_run/LHC20g4/297544/19/AnalysisResults.20g4.005.root",
        jet_R=0.2,
        min_jet_pt={"hybrid": 20},
        iterative_splittings=True,
        output_filename=base_path / "skim" / "test" / "thermal_model_skim_output.root",
        thermal_model_parameters=sources.THERMAL_MODEL_SETTINGS["central"],
        convert_data_format_prefixes={"hybrid": "hybrid", "det_level": "det_level", "part_level": "true"},
        #scale_factor=scale_factors[11],
        scale_factor=scale_factors[19],
        r_max=0.25,
    )