"""Skim the cross check task to update the names.

"""

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from jet_substructure.base import skim_analysis_objects


def names_to_export(
    grooming_method: str,
    prefixes: Mapping[str, str],
) -> Dict[str, str]:
    branch_names = {}

    substructure_variables = [
        "{grooming_method}_{prefix}_delta_R",
        "{grooming_method}_{prefix}_kt",
        "{grooming_method}_{prefix}_z",
        "{grooming_method}_{prefix}_n_to_split",
        "{grooming_method}_{prefix}_n_groomed_to_split",
        "{grooming_method}_{prefix}_n_passed_grooming",
    ]

    # Contain 8 * 3 + 1 (scale_factor) + 1 (hybrid_leading_track_pt_sub)
    branch_names["scale_factor"] = "float"
    for prefix in prefixes:
        # Jet properties
        for var_name in ["{prefix}_jet_pt", "{prefix}_leading_track_pt"]:
            branch_names[var_name.format(prefix=prefix)] = "float"
        if prefix == "hybrid":
            branch_names["hybrid_leading_track_pt_sub"] = "float"

        # Substructure properties
        for var_name in substructure_variables:
            branch_names[var_name.format(grooming_method=grooming_method, prefix=prefix)] = "float"

    # Matching properties
    for measured_like, generator_like in [("det_level", "true"), ("hybrid", "det_level")]:
        for level in ["leading", "subleading"]:
            # branch_names[f"{grooming_method}_{measured_like}_{generator_like}_matching_{level}"] = "bool"
            branch_names[f"{grooming_method}_{measured_like}_{generator_like}_matching_{level}"] = "int8_t"
            if measured_like == "hybrid":
                branch_names[
                    f"{grooming_method}_{measured_like}_{generator_like}_matching_{level}_pt_fraction_in_hybrid_jet"
                ] = "float"

    return branch_names


def _branch_name_shim_to_map_for_ROOT(branch_renames: Mapping[str, str]) -> Any:
    # Delayed import to avoid direct dependence.
    import ROOT

    map = ROOT.std.map("std::string", "std::string")()
    for k, h in branch_renames.items():
        # Why not via __setitem__? Because that would be too easy...
        map.insert((k, h))
        # map[k] = ROOT.addressof(h, True)

    return map


def skim(
    n_cores: int,
    input_filenames: Sequence[Path],
    grooming_method: str,
    tree_name: str,
    prefixes: Mapping[str, str],
) -> bool:

    # Delay ROOT import so we don't explicitly rely on it.
    import ROOT

    # Setup for ROOT
    # Enable multithreading
    ROOT.ROOT.EnableImplicitMT(n_cores)
    # Sumw2
    ROOT.TH1.SetDefaultSumw2(True)

    # Setup tree
    main_tree = ROOT.TChain(tree_name)
    for filename in input_filenames:
        main_tree.Add(str(filename))
    # TODO: Better: just pass in the scale factors.
    friend_tree = ROOT.TChain("tree")
    for filename in input_filenames:
        friend_tree.Add(str(filename.parent.parent / "scale_factor" / filename.name))
    # Add friends with scale factors
    main_tree.AddFriend(friend_tree)

    df_original = ROOT.RDataFrame(main_tree)
    # df = ROOT.RDataFrame("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl", "")

    # Add the aliases. This has to be done after the df is defined because apparently they don't carry over.
    renames = skim_analysis_objects.cross_check_task_branch_name_shim(
        grooming_method=grooming_method, input_branches=df_original.GetColumnNames()
    )
    for k, v in renames.items():
        df_original = df_original.Alias(k, v)

    rename_branches_map = _branch_name_shim_to_map_for_ROOT(renames)  # noqa: F841

    # Add scale factor column with 1s if it doesn't exist yet.
    # if "scale_factor" not in df_original.GetColumnNames():
    #    logger.info("Defining scale_factor column")
    #    df_original = df_original.Define("scale_factor", "1")

    branch_names = names_to_export(grooming_method=grooming_method, prefixes=prefixes)
    output_branch_types = _branch_name_shim_to_map_for_ROOT(branch_names)  # noqa: F841

    # This is objectively dumb...
    branch_names_list_as_str = '", "'.join(list(branch_names))
    # Need to wrap it in double quotes.
    branch_names_list_as_str = f'"{branch_names_list_as_str}"'

    output_tree_name = "tree"
    output_filename = Path("test.root")
    # This continues to be objectively dumb...
    cpp_code = f"""
    void snapshot(ROOT::RDF::RNode df) {{
        std::vector<std::string> branches = {{ {branch_names_list_as_str} }};
        df.Snapshot<{", ".join(list(branch_names.values()))}>("{output_tree_name}", "{str(output_filename)}", branches);
    }}

    void skimCrossCheckTask(std::string inputFilename, double scaleFactor, std::map<std::string, std::string> renameBranchMap, std::map<std::string, std::string> outputBranchTypes) {{
        TChain mainTree("{tree_name}");
        mainTree.Add(inputFilename.c_str());
        ROOT::RDataFrame df(mainTree);

        // Add scale factor branch.
        std::string scaleFactorInput = "float(" + std::to_string(scaleFactor) + ")";
        auto df_defined = df.Define("scale_factor", scaleFactorInput.c_str());

        // Add aliases
        for (const auto & m : renameBranchMap) {{
            std::string output = outputBranchTypes[m.first] + "(" + m.second + ")";
            //df_defined = df_defined.Alias(m.first.c_str(), m.second.c_str());
            df_defined = df_defined.Define(m.first.c_str(), output.c_str());
        }}
        snapshot(df_defined);
    }}"""
    ROOT.gInterpreter.Declare(cpp_code)
    print(cpp_code)

    import IPython

    IPython.embed()

    return True


if __name__ == "__main__":
    ...

    skim(
        n_cores=2,
        input_filenames=[Path("trains/embedPythia/6458/skim/AnalysisResults.18q.root")],
        grooming_method="dynamical_core",
        tree_name="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR020_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        prefixes={"hybrid": "data", "true": "matched", "det_level": "det_level"},
    )
