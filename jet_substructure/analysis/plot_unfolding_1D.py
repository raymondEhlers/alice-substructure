import attr
import logging
from pathlib import Path
from typing import Dict, Tuple

import uproot
from pachyderm import binned_data

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


@attr.s
class InputFileJetPt:
    smeared_pt_range: helpers.JetPtRange = attr.ib()
    n_iter_compare: int = attr.ib(default=4)
    max_iter: int = attr.ib(default=10)
    smeared_input: bool = attr.ib(default=False)
    pure_matches: bool = attr.ib(default=False)
    suffix: str = attr.ib(default="")

    @property
    def identifier(self) -> str:
        name = "jet_pt"
        name += f"_smeared_{self.smeared_pt_range}"
        if self.pure_matches:
            name += "_pureMatches"
        if self.suffix:
            name += f"_{self.suffix}"
        return name

    @property
    def filename(self) -> str:
        return f"unfolding_{self.identifier}.root"


def setup(input_file: InputFileJetPt, collision_system: str) -> Tuple[Dict[str, binned_data.BinnedData], Path]:
    base_dir = Path("output") / collision_system / "unfolding"
    input_filename = base_dir / input_file.filename
    output_dir = base_dir / "jet_pt" / input_file.identifier
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing file {input_filename}")
    logger.info(f"Output dir: {output_dir}")

    # Extract with uproot and convert to BinnedData
    hists = {}
    f = uproot.open(input_filename)
    for k in f.keys(cycle=False):
        hists[k] = binned_data.BinnedData.from_existing_data(f[k])

    # Hists:
    # [
    #     "correff20-40", "correff40-60", "correff60-80", "correff80-120",
    #     "raw", "smeared", "trueptd", "true", "truef",
    #     "Bayesian_Unfoldediter1", "Bayesian_Foldediter1", "Bayesian_Unfoldediter2", "Bayesian_Foldediter2", "Bayesian_Unfoldediter3", "Bayesian_Foldediter3",
    #     "Bayesian_Unfoldediter4", "Bayesian_Foldediter4", "Bayesian_Unfoldediter5", "Bayesian_Foldediter5", "Bayesian_Unfoldediter6", "Bayesian_Foldediter6",
    #     "Bayesian_Unfoldediter7", "Bayesian_Foldediter7", "Bayesian_Unfoldediter8", "Bayesian_Foldediter8", "Bayesian_Unfoldediter9", "Bayesian_Foldediter9",
    #     "pearsonmatrix_iter8_binshape0", "pearsonmatrix_iter8_binshape1", "pearsonmatrix_iter8_binshape2", "pearsonmatrix_iter8_binshape3",
    #     "pearsonmatrix_iter8_binpt0", "pearsonmatrix_iter8_binpt1", "pearsonmatrix_iter8_binpt2", "pearsonmatrix_iter8_binpt3", "pearsonmatrix_iter8_binpt4",
    #     "pearsonmatrix_iter8_binpt5", "pearsonmatrix_iter8_binpt6", "pearsonmatrix_iter8_binpt7",
    # ]

    return hists, output_dir
