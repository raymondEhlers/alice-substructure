""" Extract scale factors from all repaired files.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import uproot
from pachyderm import binned_data, yaml

from jet_substructure.base import helpers, skim_analysis_objects


logger = logging.getLogger(__name__)


def scale_factor_ROOT_wrapper(base_path: Path, train_number: int) -> Tuple[int, Any, Any]:
    # Setup
    filenames = helpers.ensure_and_expand_paths([Path(str(base_path).format(train_number=train_number))])

    return scale_factor_ROOT(filenames)


def scale_factor_ROOT(filenames: Sequence[Path]) -> Tuple[int, Any, Any]:
    # Delay import to avoid direct dependence
    import ROOT

    cross_section_hists = []
    n_trials_hists = []
    n_entries = 0
    for filename in filenames:
        f = ROOT.TFile(str(filename), "READ")
        embedding_hists = f.Get("AliAnalysisTaskEmcalEmbeddingHelper_histos")
        cross_section_hists.append(embedding_hists.FindObject("fHistXsection"))
        cross_section_hists[-1].SetDirectory(0)
        n_entries += cross_section_hists[-1].GetEntries()
        n_trials_hists.append(embedding_hists.FindObject("fHistTrials"))
        n_trials_hists[-1].SetDirectory(0)

        f.Close()

    cross_section = cross_section_hists[0]
    # Add the rest...
    [cross_section.Add(other) for other in cross_section_hists[1:]]
    n_trials = n_trials_hists[0]
    # Add the rest...
    [n_trials.Add(other) for other in n_trials_hists[1:]]

    return n_entries, cross_section, n_trials


def scale_factor_uproot_wrapper(
    base_path: Path, train_number: int, run_despite_issues: bool = False
) -> Tuple[int, Any, Any]:
    # Setup
    filenames = helpers.ensure_and_expand_paths([Path(str(base_path).format(train_number=train_number))])

    return scale_factor_uproot(filenames=filenames, run_despite_issues=run_despite_issues)


def scale_factor_uproot(filenames: Sequence[Path], run_despite_issues: bool = False) -> Tuple[int, Any, Any]:
    # Validation
    if not run_despite_issues:
        raise RuntimeError("Pachyderm binned data doesn't add profile histograms correctly...")

    cross_section_hists = []
    n_trials_hists = []
    n_entries: np.ndarray = []
    for filename in filenames:
        with uproot.open(filename) as input_file:
            # Retrieve the embedding helper to extract the cross section and ntrials.
            embedding_hists = input_file["AliAnalysisTaskEmcalEmbeddingHelper_histos"]
            cross_section_hist = [
                h for h in embedding_hists if h.has_member("fName") and h.member("fName") == "fHistXsection"
            ][0]
            n_entries += cross_section_hist.effective_entries()
            cross_section_hists.append(binned_data.BinnedData.from_existing_data(cross_section_hist))
            n_trials_hists.append(
                binned_data.BinnedData.from_existing_data(
                    [h for h in embedding_hists if h.has_member("fName") and h.member("fName") == "fHistTrials"][0]
                )
            )

    # Take the first non-zero value of n_entries (there should only be 1)
    return n_entries[(n_entries != 0).argmax(axis=0)], sum(cross_section_hists), sum(n_trials_hists)


def scale_factor_from_hists(n_entries: int, cross_section: Any, n_trials: Any) -> float:
    scale_factor = skim_analysis_objects.ScaleFactor.from_hists(
        cross_section=cross_section,
        n_trials=n_trials,
        n_entries=n_entries,
    )

    return scale_factor.value()


def test() -> None:
    scale_factors_ROOT = {}
    scale_factors_uproot = {}
    train_numbers = list(range(6316, 6318))

    base_path = Path("trains/embedPythia/{train_number}/AnalysisResults.*.repaired.root")
    for train_number in train_numbers:
        scale_factors_ROOT[train_number] = scale_factor_from_hists(
            *scale_factor_ROOT_wrapper(base_path=base_path, train_number=train_number)
        )
        scale_factors_uproot[train_number] = scale_factor_from_hists(
            *scale_factor_uproot_wrapper(base_path=base_path, train_number=train_number, run_despite_issues=True)
        )
        # res_ROOT = scale_factor_ROOT(base_path=base_path, train_number=train_number)
        # res_uproot = scale_factor_uproot(base_path=base_path, train_number=train_number)

    y = yaml.yaml(classes_to_register=[skim_analysis_objects.ScaleFactor])
    with open("test.yaml", "w") as f:
        y.dump(scale_factors_ROOT, f)

    print(f"scale_factors_ROOT: {scale_factors_ROOT}")
    print(f"scale_factors_uproot: {scale_factors_uproot}")
    import IPython

    IPython.embed()


if __name__ == "__main__":
    helpers.setup_logging()
    test()
