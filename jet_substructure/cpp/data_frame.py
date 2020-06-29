""" RDataFrame based analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jet_substructure.base import data_manager, helpers


logger = logging.getLogger(__name__)

# Typing helpers
RDF = Any
RootHist = Any


def matching_hists(  # noqa: C901
    df: RDF,
    grooming_method: str,
    hist_suffix: str,
    general_selection: str,
    matching_level: str = "hybrid_det_level",
    det_level_axis: Optional[Tuple[int, float, float]] = None,
    create_subjet_in_hybrid_hists: bool = False,
) -> List[RootHist]:
    # Validation
    if det_level_axis is None:
        det_level_axis = (150, 0, 150)

    # Setup
    hists = []
    matching_map: Dict[str, str] = {
        "all": "",
        "pure": f"{grooming_method}_{matching_level}_matching_leading == 1"
        f" && {grooming_method}_{matching_level}_matching_subleading == 1",
        "leading_untagged_subleading_correct": f"{grooming_method}_{matching_level}_matching_leading == 3"
        f" && {grooming_method}_{matching_level}_matching_subleading == 1",
        "leading_correct_subleading_untagged": f"{grooming_method}_{matching_level}_matching_leading == 1"
        f" && {grooming_method}_{matching_level}_matching_subleading == 3",
        "leading_correct_subleading_mistag": f"{grooming_method}_{matching_level}_matching_leading == 1"
        f" && {grooming_method}_{matching_level}_matching_subleading == 2",
        "leading_mistag_subleading_correct": f"{grooming_method}_{matching_level}_matching_leading == 2"
        f" && {grooming_method}_{matching_level}_matching_subleading == 1",
        "leading_untagged_subleading_mistag": f"{grooming_method}_{matching_level}_matching_leading == 3"
        f" && {grooming_method}_{matching_level}_matching_subleading == 2",
        "leading_mistag_subleading_untagged": f"{grooming_method}_{matching_level}_matching_leading == 2"
        f" && {grooming_method}_{matching_level}_matching_subleading == 3",
        "swap": f"{grooming_method}_{matching_level}_matching_leading == 2"
        f" && {grooming_method}_{matching_level}_matching_subleading == 2",
        "both_untagged": f"{grooming_method}_{matching_level}_matching_leading == 3"
        f" && {grooming_method}_{matching_level}_matching_subleading == 3",
    }

    # First, apply the general selection
    if general_selection:
        df = df.Filter(general_selection)

    for matching_type, selection in matching_map.items():
        # Empty string will break the filter, so we need to only apply it if there is a valid selection.
        if selection:
            df_selection = df.Filter(selection)
        else:
            df_selection = df

        # Matching
        name = f"{grooming_method}_{matching_level}_matching_{matching_type}"
        if hist_suffix:
            name += f"_{hist_suffix}"
        h_matching = df_selection.Histo1D((name, name, *det_level_axis), "det_level_jet_pt", "scale_factor",)
        hists.append(h_matching)
        # Hybrid-true
        name = f"{grooming_method}_hybrid_true_kt_response_{matching_level}_matching_type_{matching_type}"
        if hist_suffix:
            name += f"_{hist_suffix}"
        kt_hybrid_true_response = df_selection.Histo2D(
            (name, name, 26, -1, 25, 26, -1, 25,),
            f"{grooming_method}_data_kt",
            f"{grooming_method}_matched_kt",
            "scale_factor",
        )
        hists.append(kt_hybrid_true_response)
        # Hybrid-det level
        if "hybrid" in matching_level:
            name = f"{grooming_method}_hybrid_det_level_kt_response_{matching_level}_matching_type_{matching_type}"
            if hist_suffix:
                name += f"_{hist_suffix}"
            kt_hybrid_det_level_response = df_selection.Histo2D(
                (name, name, 26, -1, 25, 26, -1, 25,),
                f"{grooming_method}_data_kt",
                f"{grooming_method}_det_level_kt",
                "scale_factor",
            )
            hists.append(kt_hybrid_det_level_response)

            # Does the subjet stay in the hybird jet?
            if create_subjet_in_hybrid_hists:
                for subjet_name in ["leading", "subleading"]:
                    name = f"{grooming_method}_{matching_level}_matching_{subjet_name}_pt_fraction_in_hybrid_{matching_type}"
                    if hist_suffix:
                        name += f"_{hist_suffix}"
                    h_subjet_pt_fraction = df_selection.Histo1D(
                        (name, name, 50, 0, 1,),
                        f"{grooming_method}_{matching_level}_matching_{subjet_name}_pt_fraction_in_hybrid_jet",
                        "scale_factor",
                    )
                    hists.append(h_subjet_pt_fraction)

        # Det-level true
        if "true" in matching_level:
            name = f"{grooming_method}_det_level_true_kt_response_{matching_level}_matching_type_{matching_type}"
            if hist_suffix:
                name += f"_{hist_suffix}"
            kt_det_level_true_response = df_selection.Histo2D(
                (name, name, 26, -1, 25, 26, -1, 25,),
                f"{grooming_method}_det_level_kt",
                f"{grooming_method}_matched_kt",
                "scale_factor",
            )
            hists.append(kt_det_level_true_response)

    return hists


def run(collision_system: str, train_numbers: List[int], tree_name: str, prefix: str) -> None:
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

    base_path = Path("trains/") / collision_system / "{train_number}/AnalysisResults.*.root"
    # TODO: Fix this bullshit!
    # base_path = Path("../../clusterfs4/rehlers/substructure/trains/") / collision_system / "{train_number}/AnalysisResults.*.root"
    filenames = data_manager._ensure_and_expand_paths(
        [Path(str(base_path).format(train_number=train_number)) for train_number in train_numbers]
    )
    main_tree = ROOT.TChain(tree_name)
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
    df = df.Define("lund_plane_x_axis", f"log(1.0 / {grooming_method}_{prefix}_delta_R)").Define(
        f"{grooming_method}_{prefix}_log_kt", f"log({grooming_method}_{prefix}_kt)"
    )
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
                f"{grooming_method}_hybrid_det_level_kt_response",
                f"{grooming_method}_hybrid_det_level_kt_response",
                26,
                -1,
                25,
                26,
                -1,
                25,
            ),
            f"{grooming_method}_data_kt",
            f"{grooming_method}_det_level_kt",
            "scale_factor",
        )
        hists.append(kt_hybrid_det_level_response)
        # Hybrid-true
        kt_hybrid_true_response = df.Histo2D(
            (
                f"{grooming_method}_hybrid_true_kt_response",
                f"{grooming_method}_hybrid_true_kt_response",
                26,
                -1,
                25,
                26,
                -1,
                25,
            ),
            f"{grooming_method}_data_kt",
            f"{grooming_method}_matched_kt",
            "scale_factor",
        )
        hists.append(kt_hybrid_true_response)
        # Det-level true
        kt_det_level_true_response = df.Histo2D(
            (
                f"{grooming_method}_det_level_true_kt_response",
                f"{grooming_method}_det_level_true_kt_response",
                26,
                -1,
                25,
                26,
                -1,
                25,
            ),
            f"{grooming_method}_det_level_kt",
            f"{grooming_method}_matched_kt",
            "scale_factor",
        )
        hists.append(kt_det_level_true_response)

        # Debug code for the RDF Filtering.
        # We explicitly require splittings at both the det level and hybrid level.
        # From here, we require a splitting at det level.
        # df = df.Filter(f"{grooming_method}_det_level_n_passed_grooming > 0 && {grooming_method}_data_n_passed_grooming > 0")
        # extra = df.Filter(f"{grooming_method}_hybrid_det_level_matching_leading == 1 && {grooming_method}_hybrid_det_level_matching_subleading == 2").Count()
        # logger.debug(f"Extra: {extra.GetValue()}")

        # Matrix of possible counts values.
        # counts = {}
        # for leading_value in range(-1, 4):
        #    for subleading_value in range(-1, 4):
        #        counts[
        #            f"{grooming_method}_hybrid_det_level_matching_leading == {leading_value}"
        #            f" && {grooming_method}_hybrid_det_level_matching_subleading == {subleading_value}"
        #        ] = 0
        # for selection in counts:
        #    counts[selection] = df.Filter(selection).Count()
        ## Get the values:
        # for selection, values in counts.items():
        #    logger.info(f"Selection: {selection}: {values.GetValue()}")

        # Matching and matching dependent responses.
        matching_level_map = {
            "hybrid_det_level": ("data", "det_level"),
            "det_level_true": ("det_level", "matched"),
        }
        for matching_level, (measured_like_label, generator_like_label) in matching_level_map.items():
            # We explicitly require splittings at both the det level (generator-like) and hybrid level (measured-like).
            # This excludes matching_leading and matching_subleading == 0.
            df_selection = df.Filter(
                f"{grooming_method}_{measured_like_label}_n_passed_grooming > 0 && {grooming_method}_{generator_like_label}_n_passed_grooming > 0"
            )
            # No selection
            hists.extend(
                matching_hists(
                    df=df_selection,
                    grooming_method=grooming_method,
                    hist_suffix="",
                    general_selection="",
                    matching_level=matching_level,
                    create_subjet_in_hybrid_hists=True,
                )
            )
            # For now, we skip because it slows down RDF to have more hists...
            ## n_groomed_to_split > 1
            # hists.extend(
            #    matching_hists(
            #        df=df_selection, grooming_method=grooming_method, hist_suffix = f"{measured_like_label}_n_groomed_to_split_greater_than_1", general_selection = f"{grooming_method}_{measured_like_label}_n_groomed_to_split > 1",
            #        matching_level=matching_level,
            #    )
            # )
            ## n_groomed_to_split < 2
            # hists.extend(
            #    matching_hists(
            #        df=df_selection, grooming_method=grooming_method, hist_suffix = f"{measured_like_label}_n_groomed_to_split_less_than_2", general_selection = f"{grooming_method}_{measured_like_label}_n_groomed_to_split < 2",
            #        matching_level=matching_level,
            #    )
            # )
            # n_to_split > 4
            hists.extend(
                matching_hists(
                    df=df_selection,
                    grooming_method=grooming_method,
                    hist_suffix=f"{generator_like_label}_n_to_split_greater_than_4",
                    general_selection=f"{grooming_method}_{generator_like_label}_n_to_split > 4",
                    matching_level=matching_level,
                    create_subjet_in_hybrid_hists=True,
                )
            )
            # n_to_split < 3
            hists.extend(
                matching_hists(
                    df=df_selection,
                    grooming_method=grooming_method,
                    hist_suffix=f"{generator_like_label}_n_to_split_less_than_3",
                    general_selection=f"{grooming_method}_{generator_like_label}_n_to_split < 3",
                    matching_level=matching_level,
                    create_subjet_in_hybrid_hists=True,
                )
            )

    # If we want to save the dot graph. Unfortunately, it won't really be so insightful because we create many branches for the histograms.
    # ROOT.RDF.SaveGraph(df)

    # TODO: Disentangle response output...
    logger.info(f"Creating output file for {collision_system}, {grooming_method}, {prefix}")
    # Add the train dir into the output path name if we're processing single pt hard bins for embed pythia.
    # It's frustrating that this is necessary, but so it's ROOT - what else is new?
    base_filename = Path("output") / collision_system / "RDF"
    if len(train_numbers) == 1 and collision_system == "embedPythia":
        base_filename = base_filename / str(train_numbers[0])
    base_filename.mkdir(parents=True, exist_ok=True)
    output_filename = base_filename / f"{grooming_method}_{prefix}.root"

    output = ROOT.TFile(str(output_filename), "RECREATE")
    output.cd()
    for h in hists:
        h.SetDirectory(output)
        # Why doesn't h.Write() work? Because ROOT. It fucking sucks.
        # h.Write()
    output.Write()
    # output.ls()
    output.Close()

    logger.info("Done!")


def embed_pythia_entry_point() -> None:
    """ Allow processing one pt hard bin at a time.

    Why? Because RDF has awful performance for jitted filter statements. See: https://root-forum.cern.ch/t/rdataframe-is-very-slow-for-many-histograms/37875/15
    """
    parser = argparse.ArgumentParser(description=f"Skim cross-check task using ROOT RDF.")

    parser.add_argument("-t", "--trainNumber", type=int)
    args = parser.parse_args()

    run(
        collision_system="embedPythia",
        train_numbers=[args.trainNumber],
        # train_numbers=list(range(6007, 6008)),
        tree_name="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        prefix="hybrid",
    )


if __name__ == "__main__":
    helpers.setup_logging()
    run(
        collision_system="embedPythia",
        train_numbers=list(range(5988, 6008)),
        # train_numbers=list(range(6017, 6018)),
        # train_numbers=list(range(5988, 5989)),
        tree_name="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        # prefix="det_level",
        prefix="data",
    )
    # run(
    #    collision_system="PbPb",
    #    train_numbers=[5987],
    #    tree_name="AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl",
    #    prefix="data",
    # )
