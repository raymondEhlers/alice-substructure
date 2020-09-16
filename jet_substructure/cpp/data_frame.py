""" RDataFrame based analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Sequence, List, Optional, Tuple

import numpy as np

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
        h_matching = df_selection.Histo1D((name, name, *det_level_axis), "jet_pt_det_level", "scale_factor",)
        hists.append(h_matching)
        # Hybrid-true
        name = f"{grooming_method}_hybrid_true_kt_response_{matching_level}_matching_type_{matching_type}"
        if hist_suffix:
            name += f"_{hist_suffix}"
        kt_hybrid_true_response = df_selection.Histo2D(
            (name, name, 26, -1, 25, 26, -1, 25,),
            f"{grooming_method}_hybrid_kt",
            f"{grooming_method}_true_kt",
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
                f"{grooming_method}_hybrid_kt",
                f"{grooming_method}_det_level_kt",
                "scale_factor",
            )
            hists.append(kt_hybrid_det_level_response)

            # Does the subjet stay in the hybrid jet?
            #if create_subjet_in_hybrid_hists:
            # TODO: Fix this...
            if False:
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
                f"{grooming_method}_true_kt",
                "scale_factor",
            )
            hists.append(kt_det_level_true_response)

    return hists


def _substructure_hists(
    df: RDF,
    jet_pt_column_format: str,
    jet_pt_axis: Tuple[float],
    jet_R: float,
    prefix: str,
    grooming_method: str,
    tag: str,
    jet_pt_prefix: Optional[str] = None,
) -> List[RootHist]:
    # Validation
    if jet_pt_prefix is None:
        jet_pt_prefix = prefix

    # Setup
    hists = []

    # Apply no additional pt cuts beyond those applied outside, but plot against the relevant jet pt.
    jet_pt_column = jet_pt_column_format.format(prefix=jet_pt_prefix)

    kt = df.Histo2D(
        (f"{grooming_method}_{prefix}_kt{tag}", f"{grooming_method}_{prefix}_kt{tag}", *jet_pt_axis, 26, -1, 25),
        jet_pt_column,
        f"{grooming_method}_{prefix}_kt",
        "scale_factor",
    )
    hists.append(kt)
    delta_R = df.Histo2D(
        (f"{grooming_method}_{prefix}_delta_R{tag}", f"{grooming_method}_{prefix}_delta_R{tag}", *jet_pt_axis, 21, -0.02, jet_R),
        jet_pt_column,
        f"{grooming_method}_{prefix}_delta_R",
        "scale_factor",
    )
    hists.append(delta_R)
    z = df.Histo2D(
        (f"{grooming_method}_{prefix}_z{tag}", f"{grooming_method}_{prefix}_z{tag}", *jet_pt_axis, 21, -0.025, 0.5),
        jet_pt_column,
        f"{grooming_method}_{prefix}_z",
        "scale_factor",
    )
    hists.append(z)
    n_to_split = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_n_to_split{tag}",
            f"{grooming_method}_{prefix}_n_to_split{tag}",
            *jet_pt_axis,
            10,
            -0.5,
            9.5,
        ),
        jet_pt_column,
        f"{grooming_method}_{prefix}_n_to_split",
        "scale_factor",
    )
    hists.append(n_to_split)
    n_groomed_to_split = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_n_groomed_to_split{tag}",
            f"{grooming_method}_{prefix}_n_groomed_to_split{tag}",
            *jet_pt_axis,
            10,
            -0.5,
            9.5,
        ),
        jet_pt_column,
        f"{grooming_method}_{prefix}_n_groomed_to_split",
        "scale_factor",
    )
    hists.append(n_groomed_to_split)
    n_passed_grooming = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_n_passed_grooming{tag}",
            f"{grooming_method}_{prefix}_n_passed_grooming{tag}",
            *jet_pt_axis,
            10,
            -0.5,
            9.5,
        ),
        jet_pt_column,
        f"{grooming_method}_{prefix}_n_passed_grooming",
        "scale_factor",
    )
    hists.append(n_passed_grooming)
    df_lund_plane = df.Define("lund_plane_x_axis", f"log(1.0 / {grooming_method}_{prefix}_delta_R)").Define(
        f"{grooming_method}_{prefix}_log_kt", f"log({grooming_method}_{prefix}_kt)"
    )
    lund_plane = df_lund_plane.Histo2D(
        (f"{grooming_method}_{prefix}_lund_plane{tag}", f"{grooming_method}_{prefix}_lund_plane{tag}", 100, 0, 5, 100, -5.0, 5.0),
        "lund_plane_x_axis",
        f"{grooming_method}_{prefix}_log_kt",
        "scale_factor",
    )
    hists.append(lund_plane)

    return hists


def run(
    collision_system: str,
    input_filenames: Sequence[Path],
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,
    output_filename: Path,
    jet_pt_prefix_first: bool = False,
    n_cores: int = 8
) -> bool:
    # Delay ROOT import so we don't explicitly rely on it.
    import ROOT

    # Setup for ROOT
    # Enable multithreading
    ROOT.ROOT.EnableImplicitMT(n_cores)
    # Sumw2
    ROOT.TH1.SetDefaultSumw2(True)

    # Parameters
    jet_pt_column_format = "jet_pt_{prefix}"
    if jet_pt_prefix_first:
        jet_pt_column_format = "{prefix}_jet_pt"

    # Setup tree
    main_tree = ROOT.TChain(tree_name)
    for filename in input_filenames:
        main_tree.Add(str(filename))
    #if collision_system == "embedPythia":
    if False:
        friend_tree = ROOT.TChain("tree")
        for filename in input_filenames:
            friend_tree.Add(str(filename.parent / "scale_factor" / filename.name))
        # Add friends with scale factors
        main_tree.AddFriend(friend_tree)

    df_original = ROOT.RDataFrame(main_tree)
    # df = ROOT.RDataFrame("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl", "")

    # Add scale factor column with 1s if it doesn't exist yet.
    if "scale_factor" not in df_original.GetColumnNames():
        logger.info("Defining scale_factor column")
        df_original = df_original.Define("scale_factor", "1")

    # Apply general cuts.
    # Double couting must be applied for embedding.
    if collision_system == "embedPythia":
        double_counting_cut = "det_level_leading_track_pt >= hybrid_leading_track_pt"
        df_original = df_original.Filter(double_counting_cut)
        smeared_cut_prefix = "hybrid"
    else:
        smeared_cut_prefix = "data"

    # For the substructure variables, we also apply a jet pt cut at the measured level (ie. data or hybrid).
    jet_pt_cut = f"{jet_pt_column_format.format(prefix=smeared_cut_prefix)} >= 40 && {jet_pt_column_format.format(prefix=smeared_cut_prefix)} < 120"
    df_measured_selections = df_original.Filter(jet_pt_cut)

    hists = []
    jet_pt_axis = (28, 0, 140)
    for measured_min_kt in [-1, 2, 3, 5]:
        tag = f"_jet_pt_{smeared_cut_prefix}_40_120"
        # kt == -1 is the cause that includes the untagged. There, we don't want to include any kt tag.
        if measured_min_kt == -1:
            df = df_measured_selections
            tag += ""
        else:
            df = df_measured_selections.Filter(f"{grooming_method}_{smeared_cut_prefix}_kt > {measured_min_kt}")
            tag += f"_min_smeared_kt_{measured_min_kt}"

        # General substructure histograms.
        for prefix in prefixes:
            hists.extend(
                _substructure_hists(
                    df=df, jet_pt_column_format=jet_pt_column_format,
                    jet_pt_axis=jet_pt_axis,
                    jet_R=jet_R,
                    prefix=prefix,
                    grooming_method=grooming_method,
                    tag=tag,
                    jet_pt_prefix=prefix,
                )
            )

            # Apply no additional pt cuts beyond those above, but plot against the relevant jet pt.
            jet_pt_column = jet_pt_column_format.format(prefix=prefix)

            kt = df.Histo2D(
                (f"{grooming_method}_{prefix}_kt{tag}", f"{grooming_method}_{prefix}_kt{tag}", *jet_pt_axis, 26, -1, 25),
                jet_pt_column,
                f"{grooming_method}_{prefix}_kt",
                "scale_factor",
            )
            hists.append(kt)
            delta_R = df.Histo2D(
                (f"{grooming_method}_{prefix}_delta_R{tag}", f"{grooming_method}_{prefix}_delta_R{tag}", *jet_pt_axis, 21, -0.02, jet_R),
                jet_pt_column,
                f"{grooming_method}_{prefix}_delta_R",
                "scale_factor",
            )
            hists.append(delta_R)
            z = df.Histo2D(
                (f"{grooming_method}_{prefix}_z{tag}", f"{grooming_method}_{prefix}_z{tag}", *jet_pt_axis, 21, -0.025, 0.5),
                jet_pt_column,
                f"{grooming_method}_{prefix}_z",
                "scale_factor",
            )
            hists.append(z)
            n_to_split = df.Histo2D(
                (
                    f"{grooming_method}_{prefix}_n_to_split{tag}",
                    f"{grooming_method}_{prefix}_n_to_split{tag}",
                    *jet_pt_axis,
                    10,
                    -0.5,
                    9.5,
                ),
                jet_pt_column,
                f"{grooming_method}_{prefix}_n_to_split",
                "scale_factor",
            )
            hists.append(n_to_split)
            n_groomed_to_split = df.Histo2D(
                (
                    f"{grooming_method}_{prefix}_n_groomed_to_split{tag}",
                    f"{grooming_method}_{prefix}_n_groomed_to_split{tag}",
                    *jet_pt_axis,
                    10,
                    -0.5,
                    9.5,
                ),
                jet_pt_column,
                f"{grooming_method}_{prefix}_n_groomed_to_split",
                "scale_factor",
            )
            hists.append(n_groomed_to_split)
            n_passed_grooming = df.Histo2D(
                (
                    f"{grooming_method}_{prefix}_n_passed_grooming{tag}",
                    f"{grooming_method}_{prefix}_n_passed_grooming{tag}",
                    *jet_pt_axis,
                    10,
                    -0.5,
                    9.5,
                ),
                jet_pt_column,
                f"{grooming_method}_{prefix}_n_passed_grooming",
                "scale_factor",
            )
            hists.append(n_passed_grooming)
            df_lund_plane = df.Define("lund_plane_x_axis", f"log(1.0 / {grooming_method}_{prefix}_delta_R)").Define(
                f"{grooming_method}_{prefix}_log_kt", f"log({grooming_method}_{prefix}_kt)"
            )
            lund_plane = df_lund_plane.Histo2D(
                (f"{grooming_method}_{prefix}_lund_plane{tag}", f"{grooming_method}_{prefix}_lund_plane{tag}", 100, 0, 5, 100, -5.0, 5.0),
                "lund_plane_x_axis",
                f"{grooming_method}_{prefix}_log_kt",
                "scale_factor",
            )
            hists.append(lund_plane)

    # Some more specialized substructure variables.
    if collision_system == "embedPythia":
        # Plot all substructure variables, but instead with a constant generator level pt cut.
        # Apply generator level jet pt cut
        jet_pt_cut = f"{jet_pt_column_format.format(prefix='true')} >= 40 && {jet_pt_column_format.format(prefix='true')} < 140"
        tag = "_jet_pt_true_40_140"
        df = df_original.Filter(jet_pt_cut)

        for prefix in prefixes:
            hists.extend(_substructure_hists(
                df=df, jet_pt_column_format=jet_pt_column_format,
                jet_pt_axis=jet_pt_axis,
                jet_R=jet_R,
                prefix=prefix,
                grooming_method=grooming_method,
                tag=tag,
                jet_pt_prefix="true",
            ))

    # Responses
    #if collision_system == "embedPythia":
    if False:
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
            f"{grooming_method}_hybrid_kt",
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
            f"{grooming_method}_hybrid_kt",
            f"{grooming_method}_true_kt",
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
            f"{grooming_method}_true_kt",
            "scale_factor",
        )
        hists.append(kt_det_level_true_response)

        # Debug code for the RDF Filtering.
        # We explicitly require splittings at both the det level and hybrid level.
        # From here, we require a splitting at det level.
        # df = df.Filter(f"{grooming_method}_det_level_n_passed_grooming > 0 && {grooming_method}_hybrid_n_passed_grooming > 0")
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
            "hybrid_det_level": ("hybrid", "det_level"),
            "det_level_true": ("det_level", "true"),
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
    logger.info(f"Creating output file for {collision_system}, {grooming_method}, {prefixes}")
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

    return True


def run_standalone(collision_system: str, train_numbers: Sequence[int], tree_name: str, prefixes: Sequence[str], grooming_method: str, jet_R: float, jet_pt_prefix_first: bool = False, n_cores: int = 8) -> bool:
    # Determine the filenames based on the train numbers and predefined path here.
    base_path = Path("trains/") / collision_system / "{train_number}/skim/AnalysisResults.*.root"
    # TODO: Fix this bullshit!
    # base_path = Path("../../clusterfs4/rehlers/substructure/trains/") / collision_system / "{train_number}/AnalysisResults.*.root"
    filenames = data_manager._ensure_and_expand_paths(
        [Path(str(base_path).format(train_number=train_number)) for train_number in train_numbers]
    )

    # Output filename
    # Add the train dir into the output path name if we're processing single pt hard bins for embed pythia.
    # It's frustrating that this is necessary, but so it's ROOT - what else is new?
    base_filename = Path("output") / collision_system / "RDF" / "standalone"
    if len(train_numbers) == 1 and collision_system == "embedPythia":
        base_filename = base_filename / str(train_numbers[0])
    base_filename.mkdir(parents=True, exist_ok=True)
    output_filename = base_filename / f"{grooming_method}_{'_'.join(prefixes)}.root"

    return run(
        collision_system=collision_system,
        input_filenames=filenames,
        tree_name=tree_name,
        prefixes=prefixes,
        grooming_method=grooming_method,
        jet_R=jet_R,
        output_filename=output_filename,
        jet_pt_prefix_first=jet_pt_prefix_first,
        n_cores=n_cores
    )


def embed_pythia_entry_point() -> None:
    """ Allow processing one pt hard bin at a time.

    Why? Because RDF has awful performance for jitted filter statements. See: https://root-forum.cern.ch/t/rdataframe-is-very-slow-for-many-histograms/37875/15
    """
    helpers.setup_logging()

    parser = argparse.ArgumentParser(description=f"Skim cross-check task using ROOT RDF.")

    parser.add_argument("-t", "--trainNumber", type=int)
    parser.add_argument("-g", "--groomingMethod", type=str)
    args = parser.parse_args()

    run_standalone(
        collision_system="embedPythia",
        #train_numbers=[args.trainNumber],
        train_numbers=list(range(5966, 5986)),
        #tree_name="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        tree_name="tree",
        prefix="hybrid",
        grooming_method=args.groomingMethod,
    )


if __name__ == "__main__":
    helpers.setup_logging()
    prefixes = ["hybrid", "true", "det_level"]
    run_standalone(
        collision_system="embedPythia",
        train_numbers=list(range(5791, 5792)),
        #train_numbers=list(range(5966, 5968)),
        # train_numbers=list(range(6017, 6018)),
        # train_numbers=list(range(5988, 5989)),
        #tree_name="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        tree_name="tree",
        # prefix="det_level",
        prefixes=prefixes,
        grooming_method="leading_kt",
        jet_R=0.4,
        n_cores=6,
        jet_pt_prefix_first=True,
    )
    # for grooming_method in ["leading_kt", "leading_kt_z_cut_02", "leading_kt_z_cut_04", "dynamical_z", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02", "soft_drop_z_cut_04"]:
    #     run_standalone(
    #         collision_system="PbPb",
    #         train_numbers=[5537],
    #         #tree_name="AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl",
    #         tree_name="tree",
    #         prefixes=["data"],
    #         grooming_method=grooming_method,
    #         jet_R=0.4,
    #         n_cores=6,
    #         jet_pt_prefix_first=True,
    #     )

