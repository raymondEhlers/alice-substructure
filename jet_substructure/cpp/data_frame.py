""" RDataFrame based analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from typing import List

from pathlib import Path

from jet_substructure.base import data_manager, helpers


logger = logging.getLogger(__name__)


def run(collision_system: str, train_numbers: List[int], tree_name: str) -> None:
    # Delay ROOT import so we don't explicitly rely on it.
    import ROOT

    # Setup
    # Enable multithreading
    ROOT.ROOT.EnableImplicitMT(8)
    # Sumw2
    ROOT.TH1.SetDefaultSumw2(True)
    # Parameters
    jet_R = 0.4
    grooming_method = "leading_kt_z_cut_02"
    prefix = "data"

    base_path = Path("trains/") / collision_system / "{train_number}/AnalysisResults.*.root"
    filenames = data_manager._ensure_and_expand_paths(
        [Path(str(base_path).format(train_number=train_number)) for train_number in train_numbers]
    )
    main_tree = ROOT.TChain(
        tree_name
    )
    for filename in filenames:
        main_tree.Add(str(filename))
    if collision_system == "embedPythia":
        friend_tree = ROOT.TChain("tree")
        for filename in filenames:
            friend_tree.Add(str(filename.parent / "scale_factor" / filename.name))
        # Add friends with scale factors
        main_tree.AddFriend(friend_tree)

    df = ROOT.RDataFrame(main_tree)
    # df = ROOT.RDataFrame("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl", "")
    hybrid_jet_pt_cut = "data_jet_pt >= 40 && data_jet_pt < 120"
    df = df.Filter(hybrid_jet_pt_cut)
    if collision_system == "embedPythia":
        double_counting_cut = "det_level_leading_track_pt >= data_leading_track_pt"
        df = df.Filter(double_counting_cut)

    # Add scale factor column with 1s if it doesn't exist yet.
    if "scale_factor" not in df.GetColumnNames():
        df = df.Define("scale_factor", "1")

    # TODO: Loop for prefixes for embedded?
    jet_pt_axis = (28, 0, 140)
    hists = []
    kt = df.Histo2D(
        (f"{grooming_method}_{prefix}_kt", f"{grooming_method}_{prefix}_kt", *jet_pt_axis, 26, -1, 25),
        f"{prefix}_jet_pt",
        f"{grooming_method}_{prefix}_kt",
        "scale_factor",
    )
    hists.append(kt)
    delta_R = df.Histo2D(
        (f"{grooming_method}_{prefix}_delta_R", f"{grooming_method}_{prefix}_delta_R", *jet_pt_axis, 21, -0.02, jet_R),
        f"{prefix}_jet_pt",
        f"{grooming_method}_{prefix}_delta_R",
        "scale_factor",
    )
    hists.append(delta_R)
    z = df.Histo2D(
        (f"{grooming_method}_{prefix}_z", f"{grooming_method}_{prefix}_z", *jet_pt_axis, 21, -0.025, 0.5),
        f"{prefix}_jet_pt",
        f"{grooming_method}_{prefix}_z",
        "scale_factor",
    )
    hists.append(z)
    n_to_split = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_n_to_split",
            f"{grooming_method}_{prefix}_n_to_split",
            *jet_pt_axis,
            10,
            -0.5,
            9.5,
        ),
        f"{prefix}_jet_pt",
        f"{grooming_method}_{prefix}_n_to_split",
        "scale_factor",
    )
    hists.append(n_to_split)
    n_groomed_to_split = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_n_groomed_to_split",
            f"{grooming_method}_{prefix}_n_groomed_to_split",
            *jet_pt_axis,
            10,
            -0.5,
            9.5,
        ),
        f"{prefix}_jet_pt",
        f"{grooming_method}_{prefix}_n_groomed_to_split",
        "scale_factor",
    )
    hists.append(n_groomed_to_split)
    n_passed_grooming = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_n_passed_grooming",
            f"{grooming_method}_{prefix}_n_passed_grooming",
            *jet_pt_axis,
            10,
            -0.5,
            9.5,
        ),
        f"{prefix}_jet_pt",
        f"{grooming_method}_{prefix}_n_passed_grooming",
        "scale_factor",
    )
    hists.append(n_passed_grooming)
    df = df.Define("lund_plane_x_axis", f"log(1.0 / {grooming_method}_{prefix}_delta_R)").Define(f"{grooming_method}_{prefix}_log_kt", f"log({grooming_method}_{prefix}_kt)")
    lund_plane = df.Histo2D(
        (f"{grooming_method}_{prefix}_lund_plane", f"{grooming_method}_{prefix}_lund_plane", 100, 0, 5, 100, -5.0, 5.0),
        "lund_plane_x_axis",
        f"{grooming_method}_{prefix}_log_kt",
        "scale_factor",
    )
    hists.append(lund_plane)

    # Responses
    if collision_system == "embedPythia":
        # General responses.
        # Hybrid-det level
        kt_hybrid_det_level_response = df.Histo2D(
            (
                f"{grooming_method}_hybrid_det_level_kt_response_matching_type_all",
                f"{grooming_method}_hybrid_det_level_kt_response_matching_type_all",
                26, -1, 25,
                26, -1, 25,
            ),
            f"{grooming_method}_data_kt",
            f"{grooming_method}_det_level_kt",
            "scale_factor",
        )
        hists.append(kt_hybrid_det_level_response)
        # Hybrid-true
        kt_hybrid_true_response = df.Histo2D(
            (
                f"{grooming_method}_hybrid_true_kt_response_matching_type_all",
                f"{grooming_method}_hybrid_true_kt_response_matching_type_all",
                26, -1, 25,
                26, -1, 25,
            ),
            f"{grooming_method}_data_kt",
            f"{grooming_method}_matched_kt",
            "scale_factor",
        )
        hists.append(kt_hybrid_true_response)
        # Det-level true
        kt_det_level_true_response = df.Histo2D(
            (
                f"{grooming_method}_det_level_true_kt_response_matching_type_all",
                f"{grooming_method}_det_level_true_kt_response_matching_type_all",
                26, -1, 25,
                26, -1, 25,
            ),
            f"{grooming_method}_det_level_kt",
            f"{grooming_method}_matched_kt",
            "scale_factor",
        )
        hists.append(kt_det_level_true_response)

        # Matching and matching dependent responses.
        # From here, we require a splitting at det level.
        df = df.Filter(f"{grooming_method}_det_level_n_passed_grooming > 0")

        # Matching and response
        det_level_axis = (150, 0, 150)
        h_hybrid_det_matching_all = df.Histo1D(
            (
                f"{grooming_method}_hybrid_det_level_matching_all",
                f"{grooming_method}_hybrid_det_level_matching_all",
                *det_level_axis
            ),
            "det_level_jet_pt",
            "scale_factor",
        )
        hists.append(h_hybrid_det_matching_all)

        matching_map: Dict[str, str] = {
            "pure": f"{grooming_method}_hybrid_det_level_matching_leading == 1 &&"
                    f"{grooming_method}_hybrid_det_level_matching_subleading == 1",
            "leading_untagged_subleading_correct":
                f"{grooming_method}_hybrid_det_level_matching_leading != 1 &&"
                f"{grooming_method}_hybrid_det_level_matching_leading != 2 &&"
                f"{grooming_method}_hybrid_det_level_matching_subleading == 1",
            "leading_correct_subleading_untagged":
                f"{grooming_method}_hybrid_det_level_matching_leading == 1 &&"
                f"{grooming_method}_hybrid_det_level_matching_subleading != 1 &&"
                f"{grooming_method}_hybrid_det_level_matching_subleading != 2",
            "leading_untagged_subleading_mistag":
                f"{grooming_method}_hybrid_det_level_matching_leading != 1 &&"
                f"{grooming_method}_hybrid_det_level_matching_leading != 2 &&"
                f"{grooming_method}_hybrid_det_level_matching_subleading == 2",
            "leading_mistag_subleading_untagged":
                f"{grooming_method}_hybrid_det_level_matching_leading == 2 &&"
                f"{grooming_method}_hybrid_det_level_matching_subleading != 1 &&"
                f"{grooming_method}_hybrid_det_level_matching_subleading != 2",
            "swap": f"{grooming_method}_hybrid_det_level_matching_leading == 2 &&"
                    f"{grooming_method}_hybrid_det_level_matching_subleading == 2",
            "both_untagged":
                f"{grooming_method}_hybrid_det_level_matching_leading != 1 &&"
                f"{grooming_method}_hybrid_det_level_matching_leading != 2 &&"
                f"{grooming_method}_hybrid_det_level_matching_subleading != 1 &&"
                f"{grooming_method}_hybrid_det_level_matching_subleading != 2",
        }
        for matching_type, selection in matching_map.items():
            df_selection = df.Filter(selection)
            # Matching
            h_matching = df_selection.Histo1D(
                (
                    f"{grooming_method}_hybrid_det_level_matching_{matching_type}",
                    f"{grooming_method}_hybrid_det_level_matching_{matching_type}",
                    *det_level_axis
                ),
                "det_level_jet_pt",
                "scale_factor",
            )
            hists.append(h_matching)
            # Hybrid-det level
            kt_hybrid_det_level_response = df_selection.Histo2D(
                (
                    f"{grooming_method}_hybrid_det_level_kt_response_matching_type_{matching_type}",
                    f"{grooming_method}_hybrid_det_level_kt_response_matching_type_{matching_type}",
                    26, -1, 25,
                    26, -1, 25,
                ),
                f"{grooming_method}_data_kt",
                f"{grooming_method}_det_level_kt",
                "scale_factor",
            )
            hists.append(kt_hybrid_det_level_response)
            # Hybrid-true
            kt_hybrid_true_response = df_selection.Histo2D(
                (
                    f"{grooming_method}_hybrid_true_kt_response_matching_type_{matching_type}",
                    f"{grooming_method}_hybrid_true_kt_response_matching_type_{matching_type}",
                    26, -1, 25,
                    26, -1, 25,
                ),
                f"{grooming_method}_data_kt",
                f"{grooming_method}_matched_kt",
                "scale_factor",
            )
            hists.append(kt_hybrid_true_response)
            # Det-level true
            kt_det_level_true_response = df_selection.Histo2D(
                (
                    f"{grooming_method}_det_level_true_kt_response_matching_type_{matching_type}",
                    f"{grooming_method}_det_level_true_kt_response_matching_type_{matching_type}",
                    26, -1, 25,
                    26, -1, 25,
                ),
                f"{grooming_method}_det_level_kt",
                f"{grooming_method}_matched_kt",
                "scale_factor",
            )
            hists.append(kt_det_level_true_response)

    # TODO: Disentangle response output...
    logger.info(f"Creating output file for {collision_system}, {grooming_method}, {prefix}")
    output_filename = Path("output") / collision_system / "RDF" / f"{grooming_method}_{prefix}.root"
    output = ROOT.TFile(str(output_filename), "RECREATE")
    output.cd()
    for h in hists:
        h.SetDirectory(output)
        # Why doesn't h.Write() work? Because ROOT. It fucking sucks.
        #h.Write()
    output.Write()
    #output.ls()
    output.Close()

    logger.info("Done!")


if __name__ == "__main__":
    helpers.setup_logging()
    run(
        collision_system="embedPythia",
        train_numbers=list(range(5988, 6008)),
        #train_numbers=list(range(5988, 5991)),
        tree_name="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"
    )
    run(
        collision_system="PbPb",
        train_numbers=[5987],
        tree_name="AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl"
    )

