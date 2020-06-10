""" RDataFrame based analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path

from jet_substructure.base import data_manager, helpers


def run(collision_system: str) -> None:
    # Delay ROOT import so we don't explicitly rely on it.
    import ROOT

    # Setup
    # Enable multithreading
    ROOT.ROOT.EnableImplicitMT(2)
    # Sumw2
    ROOT.TH1.SetSumw2()
    # Parameters
    jet_R = 0.4
    grooming_method = "leading_kt_z_cut_02"
    prefix = "data"

    base_path = Path("trains/") / collision_system / "{train_number}/AnalysisResults.*.root"
    filenames = data_manager._ensure_and_expand_paths(
        [Path(str(base_path).format(train_number=train_number)) for train_number in range(5988, 6008)]
    )
    main_tree = ROOT.TChain(
        "AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"
    )
    for filename in filenames:
        main_tree.Add(str(filename))
    if collision_system == "embedPythia":
        friend_tree = ROOT.TChain("tree")
        for filename in filenames:
            friend_tree.Add(str(filename.with_suffix("")) + "_scale_factors.root")
        # Add friends with scale factors
        main_tree.AddFriend(friend_tree)

    df = ROOT.RDataFrame(main_tree)
    # df = ROOT.RDataFrame("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl", "")
    double_counting_cut = "det_level_leading_track_pt >= data_leading_track_pt"
    hybrid_jet_pt_cut = "data_jet_pt >= 40 && data_jet_pt < 120"
    df = df.Filter(hybrid_jet_pt_cut).Filter(double_counting_cut)
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
    lund_plane = df.Histo2D(
        (f"{grooming_method}_{prefix}_lund_plane", f"{grooming_method}_{prefix}_lund_plane", 100, 0, 5, 100, -5.0, 5.0),
        f"log(1.0 / {grooming_method}_{prefix}_delta_R)",
        f"log({grooming_method}_{prefix}_kt)",
        "scale_factor",
    )
    hists.append(lund_plane)

    output = ROOT.TFile("output.root")
    for h in hists:
        h.Save()
    output.Close()


if __name__ == "__main__":
    helpers.setup_logging()
