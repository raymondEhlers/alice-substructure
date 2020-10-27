""" 2D substructure unfolding implemented via RooUnfold.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
from typing_extensions import Final

import attr
import numpy as np

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)

# Type helpers
RooUnfoldErrorTreatment = Any
RooUnfoldResponse = Any
TH2D = Any
TMatrixD = Any


def _np_array_converter(value: Any, dtype: np.dtype = np.float64) -> np.ndarray:
    """ Convert the given value to a numpy array.

    Normally, we would just use np.array directly as the converter function. However, mypy will complain if
    the converter is untyped. So we add (trivial) typing here.  See: https://github.com/python/mypy/issues/6172.

    Args:
        value: Value to be converted to a numpy array.
    Returns:
        The converted numpy array.
    """
    return np.array(value, dtype=dtype)


@attr.s
class ParameterSettings:
    true_bins: np.ndarray = attr.ib(converter=_np_array_converter)
    smeared_bins: np.ndarray = attr.ib(converter=_np_array_converter)


@attr.s
class SubstructureVariableSettings(ParameterSettings):
    name: str = attr.ib()
    variable_name: str = attr.ib()
    min_smeared: float = attr.ib()
    max_smeared: float = attr.ib()
    untagged_value: float = attr.ib()

    @classmethod
    def from_binning(
        cls: Type["SubstructureVariableSettings"],
        true_bins: np.ndarray,
        smeared_bins: np.ndarray,
        name: str,
        variable_name: str,
        untagged_bin_below_range: bool = True,
    ) -> "SubstructureVariableSettings":
        if untagged_bin_below_range:
            min_smeared = smeared_bins[1]
            max_smeared = smeared_bins[-1]
            untagged_value = (smeared_bins[1] - smeared_bins[0]) / 2 + smeared_bins[0]
            # TODO: Sort out
            # untaggedBinDescription = std::to_string(static_cast<int>(smearedSplittingVariableBins[0] * printFactor)) + "_" + static_cast<int>(smearedSplittingVariableBins[1] * printFactor);
        else:
            min_smeared = smeared_bins[0]
            max_smeared = smeared_bins[-2]
            untagged_value = (smeared_bins[-1] - smeared_bins[-2]) / 2 + smeared_bins[-2]
            # TODO: Sort out
            # untaggedBinDescription = std::to_string(static_cast<int>(smearedSplittingVariableBins[lastBin - 1] * printFactor)) + "_" + static_cast<int>(smearedSplittingVariableBins[lastBin] * printFactor);

        return cls(
            true_bins=true_bins,
            smeared_bins=smeared_bins,
            name=name,
            variable_name=variable_name,
            min_smeared=min_smeared,
            max_smeared=max_smeared,
            untagged_value=untagged_value,
        )


@attr.s
class Settings:
    grooming_method: str = attr.ib()
    jet_pt: ParameterSettings = attr.ib()
    substructure_variable: SubstructureVariableSettings = attr.ib()


def _pass_filenames_to_ROOT(filenames: Sequence[Path]) -> List[str]:
    return [str(f) for f in filenames]


def _array_to_ROOT(arr: np.ndarray, type_name: str = "double") -> Any:
    """ Convert numpy array to std::vector via ROOT.

    Because it apparently can't handle conversions directly. Which is really dumb...

    In principle, we could convert the numpy dtype into the c++ type, but that's a lot of mapping
    to be done for a function that (hopefully) isn't used so often. So we let the user decide.

    Args:
        arr: Numpy array to be converted.
        type_name: c++ type name to be used for the vector. Default: "double".
    Returns:
        std::vector containing the numpy array values.
    """
    import ROOT

    vector = ROOT.std.vector(type_name)()
    for a in arr:
        vector.push_back(a)
    return vector


def correlation_hist_substructure_var(cov: TMatrixD, name: str, title: str, na: int, nb: int, kbin: int) -> TH2D:
    """ Correlation histogram for the substructure variable.

    Varies from the pt by the indexing of the covariance matrix.

    Args:
        cov: Covariance matrix derived from the unfolding.
        name: Name of the covariance matrix.
        title: Title of the covariance matrix.
        na: Number of x bins.
        nb: Number of y bins.
        kbin: Bin in the selected dimension.
    Returns:
        The correlation histogram.
    """
    import ROOT

    h = ROOT.TH2D(name, title, nb, 0, nb, nb, 0, nb)

    for l in range(0, nb):
        for n in range(0, nb):
            index1 = kbin + na * l
            index2 = kbin + na * n
            Vv = cov(index1, index1) * cov(index2, index2)
            if Vv > 0.0:
                h.SetBinContent(l + 1, n + 1, cov(index1, index2) / np.sqrt(Vv))
    return h


def correlation_hist_pt(cov: TMatrixD, name: str, title: str, na: int, nb: int, kbin: int) -> TH2D:
    """ Correlation histogram for the jet pt.

    Varies from the substructure variable by the indexing of the covariance matrix.

    Args:
        cov: Covariance matrix derived from the unfolding.
        name: Name of the covariance matrix.
        title: Title of the covariance matrix.
        na: Number of x bins.
        nb: Number of y bins.
        kbin: Bin in the selected dimension.
    Returns:
        The correlation histogram.
    """
    import ROOT

    h = ROOT.TH2D(name, title, na, 0, na, na, 0, na)

    for l in range(0, na):
        for n in range(0, na):
            index1 = l + na * kbin
            index2 = n + na * kbin
            Vv = cov(index1, index1) * cov(index2, index2)
            if Vv > 0.0:
                h.SetBinContent(l + 1, n + 1, cov(index1, index2) / np.sqrt(Vv))
    return h


def unfolding_2D(
    response: RooUnfoldResponse,
    h2_true: TH2D,
    input_spectra: TH2D,
    error_treatment: Optional[RooUnfoldErrorTreatment] = None,
    tag: str = "",
    max_iter: Final[int] = 20,
    n_iter_for_covariance: Final[int] = 8,
) -> Dict[str, TH2D]:
    """ Perform unfolding in 2D.

    Args:
        response: Response matrix.
        h2_true: True histogram.
        input_spectra: Input histogram.
        error_treatment: Error treatment to be used for unfolding.
        tag: Tag...
        max_iter: Maximum number of iterations for unfolding. Default: 20.
        n_iter_for_covariance: Number of iterations that should be used for calculating the covariance. Default: 8.
    Returns:
        Unfolded and folded hists per iter, as well as the covariance matrices. See the hist names in the code.
    """
    # Delayed import for convenience.
    import ROOT

    # Validation
    if error_treatment is None:
        error_treatment = ROOT.RooUnfold.ErrorTreatment.kCovariance

    # Setup
    logger.info("=======================================================")
    logger.info(f'Unfolding for tag "{tag}"')
    # Determine the tag. If we have a non-empty tag, we append it to all of the histograms.
    if tag != "":
        tag += "_"
    output_hists = {}

    for n_iter in range(1, max_iter):
        logger.debug(f"Iteration {n_iter}")

        # Setup the response for unfolding.
        unfold = ROOT.RooUnfoldBayes(response, input_spectra, n_iter)
        # And then unfold.
        h_unfold = unfold.Hreco(error_treatment)

        # Refold the truth (ie. fold back).
        h_fold = response.ApplyToTruth(h_unfold, "")

        # Clone unfolded and refolded hists to write to the output file.
        name = f"{tag}Bayesian_Unfoldediter{n_iter}"
        output_hists[name] = h_unfold.Clone()
        name = f"{tag}Bayesian_Foldediter{n_iter}"
        output_hists[name] = h_fold.Clone(name)

        # Retrieve the covariance matrix. Only for a selected iteration.
        if n_iter == n_iter_for_covariance:
            covariance_matrix = unfold.Ereco(ROOT.RooUnfold.kCovariance)
            # Substructure variable.
            for k in range(0, h2_true.GetNbinsX()):
                h_corr = correlation_hist_substructure_var(
                    covariance_matrix, f"{tag}corr{k}", "Covariance matrix", h2_true.GetNbinsX(), h2_true.GetNbinsY(), k
                )
                name = f"{tag}pearsonmatrix_iter{n_iter}_bin_substructure_var{k}"
                cov_substructure_var = h_corr.Clone(name)
                cov_substructure_var.SetDrawOption("colz")
                # Save
                output_hists[name] = cov_substructure_var

            # Jet pt.
            for k in range(0, h2_true.GetNbinsY()):
                h_corr = correlation_hist_pt(
                    covariance_matrix,
                    f"{tag}corr{k}pt",
                    "Covariance matrix",
                    h2_true.GetNbinsX(),
                    h2_true.GetNbinsY(),
                    k,
                )
                name = f"{tag}pearsonmatrix_iter{n_iter}_binpt{k}"
                cov_pt = h_corr.Clone(name)
                cov_pt.SetDrawOption("colz")
                # Save
                output_hists[name] = cov_pt

    logger.info("Finished unfolding!")
    logger.info("=======================================================")

    return output_hists


def setup(grooming_method: str):
    settings = Settings(
        grooming_method=grooming_method,
        jet_pt=ParameterSettings(
            true_bins=np.array([0, 30, 40, 60, 80, 100, 120, 160], dtype=np.float64),
            smeared_bins=np.array([30, 40, 50, 60, 80, 100, 120], dtype=np.float64),
        ),
        substructure_variable=SubstructureVariableSettings.from_binning(
            true_bins=np.array(
                # NOTE: (-0.05, 0) is the untagged bin.
                [-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100],
                dtype=np.float64,
            ),
            smeared_bins=np.array([1, 2, 3, 4, 5, 7, 10, 15], dtype=np.float64),
            name="kt",
            variable_name="kt",
            untagged_bin_below_range=True,
        ),
    )

    return settings


def _default_hists(settings: Settings) -> Tuple[Dict[str, TH2D], Any]:
    import ROOT

    hists = {}
    # the raw correlation (ie. data)
    hists["h2_raw"] = ROOT.TH2D(
        "raw",
        "raw",
        len(settings.substructure_variable.smeared_bins) - 1,
        settings.substructure_variable.smeared_bins,
        len(settings.jet_pt.smeared_bins) - 1,
        settings.jet_pt.smeared_bins,
    )
    # detector measure level (ie. hybrid)
    hists["h2_smeared"] = ROOT.TH2D(
        "smeared",
        "smeared",
        len(settings.substructure_variable.smeared_bins) - 1,
        settings.substructure_variable.smeared_bins,
        len(settings.jet_pt.smeared_bins) - 1,
        settings.jet_pt.smeared_bins,
    )
    # detector measure level no cuts (ie. hybrid, but no cuts).
    hists["h2_smeared_no_cuts"] = ROOT.TH2D(
        "smearednocuts",
        "smearednocuts",
        len(settings.substructure_variable.smeared_bins) - 1,
        settings.substructure_variable.smeared_bins,
        # NOTE: We're actually going to fill hybrid jet pt. But we want a wider range, so we use true jet pt bins for convenience.
        len(settings.jet_pt.true_bins) - 1,
        settings.jet_pt.true_bins,
    )
    # true correlations with measured cuts
    hists["h2_true"] = ROOT.TH2D(
        "true",
        "true",
        len(settings.substructure_variable.true_bins) - 1,
        settings.substructure_variable.true_bins,
        len(settings.jet_pt.true_bins) - 1,
        settings.jet_pt.true_bins,
    )
    # full true correlation (without cuts)
    hists["h2_full_eff"] = ROOT.TH2D(
        "truef",
        "truef",
        len(settings.substructure_variable.true_bins) - 1,
        settings.substructure_variable.true_bins,
        len(settings.jet_pt.true_bins) - 1,
        settings.jet_pt.true_bins,
    )
    # Correlation between the splitting variables at true and hybrid (with cuts).
    hists["h2_substructure_variable"] = ROOT.TH2D(
        "h2SplittingVariable",
        "h2SplittingVariable",
        len(settings.substructure_variable.smeared_bins) - 1,
        settings.substructure_variable.smeared_bins,
        len(settings.substructure_variable.true_bins) - 1,
        settings.substructure_variable.true_bins,
    )

    # Sumw2 for all hists, store for passing them...
    hists_map_for_root = ROOT.std.map("std::string", "TH2D *")()
    for k, h in hists.items():
        h.Sumw2()
        # Why not via __setitem__? Because that would be too easy...
        hists_map_for_root.insert((k, h))
        # hists_to_root[k] = ROOT.addressof(h, True)

    return hists, hists_map_for_root


def run_unfolding_fall_back(settings: Settings,) -> bool:
    # TODO: Determine input settings, etc here. This is the point of having the python code, but calling down to c++

    # Delayed import to avoid direct dependence.
    import ROOT

    # Setup
    # Load RooUnfold
    ROOT.gSystem.Load("libRooUnfold")
    # Load the unfolding utilities. We're careful to be (relatively) position independent.
    # This just assumes that this file is in the same directory as the unfolding.cxx file, which should
    # usually be a reasonable assumption.
    unfolding_cxx = Path(__file__).resolve().parent / "unfolding.cxx"
    ROOT.gInterpreter.ProcessLine(f"""#include "{str(unfolding_cxx)}" """)
    # Nominally additional setup for MT. It's not really going to do us any good here, but it doesn't hurt anything.
    # NOTE: We do need to specify 1 to ensure that we don't use extra cores.
    ROOT.ROOT.EnableImplicitMT(1)

    # Define hists (and the map to pass them into ROOT for unfolding)
    hists, hists_map_for_root = _default_hists(settings=settings)

    # TODO: Determine the untagged bin value
    # TODO: Make args...
    data_prefix = "data"
    data_jet_pt_name = f"{data_prefix}_jet_pt"
    data_substructure_variable_name = (
        f"{settings.grooming_method}_{data_prefix}_{settings.substructure_variable.variable_name}"
    )
    output_filename = "output_filename.root"

    # NOTE: TChain can only handle one "*" in the filename.
    data_filenames = [
        Path("trains/PbPb/5863/skim/*.root"),
    ]
    embedded_filenames = [Path(f"trains/embedPythia/{train_number}/skim/*.root") for train_number in range(5966, 5986)]

    # print(hists)
    res = ROOT.run_unfolding_2D(
        hists_map_for_root,
        settings.grooming_method,
        settings.substructure_variable.variable_name,
        _array_to_ROOT(settings.jet_pt.smeared_bins, "double"),
        _array_to_ROOT(settings.jet_pt.true_bins, "double"),
        _array_to_ROOT(settings.substructure_variable.smeared_bins, "double"),
        _array_to_ROOT(settings.substructure_variable.true_bins, "double"),
        settings.substructure_variable.untagged_value,
        settings.substructure_variable.min_smeared,
        settings.substructure_variable.max_smeared,
        _array_to_ROOT(_pass_filenames_to_ROOT(data_filenames), "std::string"),
        _array_to_ROOT(_pass_filenames_to_ROOT(embedded_filenames), "std::string"),
        output_filename,
    )

    logger.debug(res)
    # import IPython; IPython.embed()

    # TODO: From here, we have the responses, as well as the filled hists (stored in our map).
    output_hists = unfolding_2D(
        response=res.response,
        # response=res._0,
        h2_true=hists["h2_true"],
        input_spectra=hists["h2_raw"],
    )

    return True


def run_unfolding(
    grooming_method: str,
    substructure_variable_name: str,
    smeared_substructure_variable_bins: np.ndarray,
    smeared_jet_pt_bins: np.ndarray,
    true_substructure_variable_bins: np.ndarray,
    true_jet_pt_bins: np.ndarray,
    # data_filenames: Sequence[Path],
    # embedded_filenames: Sequence[Path],
    # output_filename: Path,
) -> bool:
    ...

    # Delayed import to avoid direct dependence.
    import ROOT

    # Configuration (not totally clear if this actually does anything for this script...)
    ROOT.ROOT.EnableImplicitMT(1)

    data_chain = ROOT.TChain("tree")
    data_chain.Add("trains/PbPb/5863/skim/*.root")

    # the raw correlation (ie. data)
    h2_raw = ROOT.TH2D(
        "r",
        "raw",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(smeared_jet_pt_bins) - 1,
        smeared_jet_pt_bins,
    )
    # detector measure level (ie. hybrid)
    h2_smeared = ROOT.TH2D(
        "smeared",
        "smeared",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(smeared_jet_pt_bins) - 1,
        smeared_jet_pt_bins,
    )
    # detector measure level no cuts (ie. hybrid, but no cuts).
    # NOTE: Strictly speaking, the y axis binning is at the hybrid level, but we want a wider range. So we use the true_jet_pt_bins.
    h2_smeared_no_cuts = ROOT.TH2D(
        "smearednocuts",
        "smearednocuts",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(true_jet_pt_bins) - 1,
        true_jet_pt_bins,
    )
    # true correlations with measured cuts
    h2_true = ROOT.TH2D(
        "true",
        "true",
        len(true_substructure_variable_bins) - 1,
        true_substructure_variable_bins,
        len(true_jet_pt_bins) - 1,
        true_jet_pt_bins,
    )
    # full true correlation (without cuts)
    h2_full_eff = ROOT.TH2D(
        "truef",
        "truef",
        len(true_substructure_variable_bins) - 1,
        true_substructure_variable_bins,
        len(true_jet_pt_bins) - 1,
        true_jet_pt_bins,
    )
    # Correlation between the splitting variables at true and hybrid (with cuts).
    h2_substructure_variable = ROOT.TH2D(
        "h2SplittingVariable",
        "h2SplittingVariable",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(true_substructure_variable_bins) - 1,
        true_substructure_variable_bins,
    )

    # TODO: Determine the untagged bin value

    data_prefix = "data"
    data_jet_pt_name = f"{data_prefix}_jet_pt"
    data_substructure_variable_name = f"{grooming_method}_{data_prefix}_{substructure_variable_name}"

    # TEMP for quick performance test
    smeared_untagged_bin_value = 2.5
    min_smeared_substructure_variable = 3
    max_smeared_substructure_variable = 15

    logger.info("Starting loop")
    # TTreeReaderValue<float> dataJetPt(dataReader, ("jet_pt_" + dataPrefix).c_str());
    # TTreeReaderValue<float> dataSubstructureVariable(dataReader, (groomingMethod + "_" + dataPrefix + "_" + substructureVariableName).c_str());
    for jet in data_chain:
        # Since the names are dynamic, we need to retrieve them each event using getattr.
        data_jet_pt = getattr(jet, data_jet_pt_name)
        data_substructure_variable = getattr(jet, data_substructure_variable_name)
        # Jet pt cut.
        if data_jet_pt < smeared_jet_pt_bins[0] or data_jet_pt > smeared_jet_pt_bins[-1]:
            continue
        # Substructure variable cut.
        if data_substructure_variable < 0:
            # Assign to the untagged bin.
            data_substructure_variable = smeared_untagged_bin_value
        else:
            if (
                data_substructure_variable < min_smeared_substructure_variable
                or data_substructure_variable > max_smeared_substructure_variable
            ):
                continue
        h2_raw.Fill(data_substructure_variable, data_jet_pt)
    logger.info("Done with loop")

    # Embedding


def run_unfolding_rdf(
    grooming_method: str,
    substructure_variable_name: str,
    smeared_substructure_variable_bins: np.ndarray,
    smeared_jet_pt_bins: np.ndarray,
    true_substructure_variable_bins: np.ndarray,
    true_jet_pt_bins: np.ndarray,
    # data_filenames: Sequence[Path],
    # embedded_filenames: Sequence[Path],
    # output_filename: Path,
    jet_pt_prefix_first: bool = True,
) -> bool:
    # Validation
    # Parameters
    jet_pt_column_format = "jet_pt_{prefix}"
    if jet_pt_prefix_first:
        jet_pt_column_format = "{prefix}_jet_pt"

    # Delayed import to avoid direct dependence.
    import ROOT

    # Configuration (not totally clear if this actually does anything for this script...)
    ROOT.ROOT.EnableImplicitMT(1)

    data_chain_data = ROOT.TChain("tree")
    data_chain_data.Add("trains/PbPb/5863/skim/*.root")
    df_data = ROOT.RDataFrame(data_chain_data)

    # the raw correlation (ie. data)
    h2_raw_args = (
        "r",
        "raw",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(smeared_jet_pt_bins) - 1,
        smeared_jet_pt_bins,
    )
    # detector measure level (ie. hybrid)
    h2_smeared_args = (
        "smeared",
        "smeared",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(smeared_jet_pt_bins) - 1,
        smeared_jet_pt_bins,
    )
    # detector measure level no cuts (ie. hybrid, but no cuts).
    # NOTE: Strictly speaking, the y axis binning is at the hybrid level, but we want a wider range. So we use the true_jet_pt_bins.
    h2_smeared_no_cuts_args = (
        "smearednocuts",
        "smearednocuts",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(true_jet_pt_bins) - 1,
        true_jet_pt_bins,
    )
    # true correlations with measured cuts
    h2_true_args = (
        "true",
        "true",
        len(true_substructure_variable_bins) - 1,
        true_substructure_variable_bins,
        len(true_jet_pt_bins) - 1,
        true_jet_pt_bins,
    )
    # full true correlation (without cuts)
    h2_full_eff_args = (
        "truef",
        "truef",
        len(true_substructure_variable_bins) - 1,
        true_substructure_variable_bins,
        len(true_jet_pt_bins) - 1,
        true_jet_pt_bins,
    )
    # Correlation between the splitting variables at true and hybrid (with cuts).
    h2_substructure_variable_args = (
        "h2SplittingVariable",
        "h2SplittingVariable",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(true_substructure_variable_bins) - 1,
        true_substructure_variable_bins,
    )

    # TODO: Determine the untagged bin value
    # TODO: Make arguments, cleanup, consolidate...
    n_cores = 1
    data_prefix = "data"
    data_jet_pt_name = f"{data_prefix}_jet_pt"
    data_substructure_variable_name = f"{grooming_method}_{data_prefix}_{substructure_variable_name}"

    # TEMP for quick performance test
    smeared_untagged_bin_value = 2.5
    min_smeared_substructure_variable = 3
    max_smeared_substructure_variable = 15

    # Define RooUnfold Objects
    # We have to do this in c++ for ROOT to be able to access them for the RDF.
    # Might also be necessary...
    ROOT.gSystem.Load("libRooUnfold")
    r = f"""
    #include <vector>
    #include <RooUnfoldBayes.h>
    #include <RooUnfoldResponse.h>

    // We can access these directly.
    std::vector<std::unique_ptr<RooUnfoldResponse>> responses;
    std::vector<std::unique_ptr<RooUnfoldResponse>> responses_no_trunc;

    // NOTE: You can't have a for loop directly in a Declare, apparently...
    // NOTE: You can't do this directly with RooUnfold objects without setting up the response because the
    //       copy constructor calls setup...
    //       RooUnfold sucks!
    void setupResponses() {{
        for (unsigned int i = 0; i < {n_cores}; i++) {{
            responses.emplace_back(std::unique_ptr<RooUnfoldResponse>());
            responses_no_trunc.emplace_back(std::unique_ptr<RooUnfoldResponse>());
        }}
    }}

    /*double getSubstructureVariable(double {data_substructure_variable_name}) {{
        return {data_substructure_variable_name} < 0 ? {smeared_untagged_bin_value} : {data_substructure_variable_name};
    }}*/
    double getSubstructureVariable(double var) {{
        return var < 0 ? {smeared_untagged_bin_value} : var;
    }}
    std::vector <TRandom3> randoms;
    void setupRandomNumbers() {{
        for (unsigned int i = 0; i < {n_cores}; i++) {{
            randoms.emplace_back(TRandom3(0));
        }}
    }}
    // Alternatively, we could use the special rdfentry_ to do something like this, but less well...
    bool randomSample(unsigned int slotNumber, double jetPt) {{
        return randoms[slotNumber].Rndm() > 0.9;
    }}
    //auto testRandomSample = [randoms](unsigned int slotNumber,

    void setup() {{
        setupResponses();
        setupRandomNumbers();
    }}
    std::vector<std::string> colName = {{ "{jet_pt_column_format.format(prefix='hybrid')}" }};
    """
    print(r)
    ROOT.gInterpreter.Declare(r)

    logger.info(ROOT.RooUnfold.kCovariance)
    logger.info(ROOT.setup())
    logger.info(ROOT.responses)
    logger.info(ROOT.randomSample(0, 12.0))

    logger.info("Starting data frame")
    # TTreeReaderValue<float> dataJetPt(dataReader, ("jet_pt_" + dataPrefix).c_str());
    # TTreeReaderValue<float> dataSubstructureVariable(dataReader, (groomingMethod + "_" + dataPrefix + "_" + substructureVariableName).c_str());

    smeared_jet_pt_filter = (
        f"{data_jet_pt_name} >= {smeared_jet_pt_bins[0]} && {data_jet_pt_name} <= {smeared_jet_pt_bins[-1]}"
    )
    substructure_variable_value_filter = f"({data_substructure_variable_name} >= {min_smeared_substructure_variable} && {data_substructure_variable_name} <= {max_smeared_substructure_variable}) || ({data_substructure_variable_name} < 0)"
    df_data = df_data.Filter(f"({smeared_jet_pt_filter}) && ({substructure_variable_value_filter})")
    print(data_substructure_variable_name)
    df_data = df_data.Define(
        "data_substructure_variable",
        f"getSubstructureVariable({data_substructure_variable_name})",
        # f"[](double {data_substructure_variable_name}) {{ return {data_substructure_variable_name} < 0 ? {smeared_untagged_bin_value} : {data_substructure_variable_name} }}",
        # lambda substructure_variable: smeared_untagged_bin_value if substructure_variable < 0 else substructure_variable,
        # lambda leading_kt_data_kt: smeared_untagged_bin_value if leading_kt_data_kt < 0 else leading_kt_data_kt,
        # [data_substructure_variable_name]
    )

    # Data
    h2_raw = df_data.Histo2D(h2_raw_args, "data_substructure_variable", data_jet_pt_name,)

    logger.info("Starting calculation")
    # logger.info(f"Entries: {h2_raw.GetEntries()}")

    # Starting embedding from here, but needs cleanup...
    data_chain_embedded = ROOT.TChain("tree")
    # TODO: Make these an argument...
    data_chain_embedded.Add("trains/embedPythia/5966/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5967/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5968/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5969/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5970/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5971/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5972/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5973/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5974/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5975/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5976/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5977/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5978/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5979/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5980/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5981/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5982/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5983/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5984/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5985/skim/*.root")
    df_embedded = ROOT.RDataFrame(data_chain_embedded)

    # TODO: Make these arguments
    smeared_cut_prefix = "hybrid"

    # Implement all of the embedding cuts via filters.
    # NOTE: Many of these values are hard coded and should be refactored at some point.
    true_jet_pt_filter = f"{jet_pt_column_format.format(prefix='true')} <= 160"
    true_substructure_variable_filter = f"{grooming_method}_true_kt <= 100"
    # Double counting is already applied above.
    # Now, the hybrid cuts.
    # Hybrid jet pt
    hybrid_jet_pt_filter = f"{jet_pt_column_format.format(prefix=smeared_cut_prefix)} >= 40 && {jet_pt_column_format.format(prefix=smeared_cut_prefix)} < 120"
    # Hybrid substructure variable
    df_embedded = df_embedded.Filter(
        "(" + ") && (".join([true_jet_pt_filter, true_substructure_variable_filter, hybrid_jet_pt_filter]) + ")"
    )
    # df_embedded.ForeachSlot(ROOT.randomSample)
    # df_embedded.ForeachSlot(ROOT.randomSample, [jet_pt_column_format.format(prefix=smeared_cut_prefix)])
    # df_embedded.ForeachSlot(lambda slot, val: (slot, val), "jet_pt_hybrid")
    # df_embedded.ForeachSlot("randomSample", ROOT.colName)
    # df_embedded.Foreach("randomSample")

    # TEST
    smearedJetPtBins = np.array([30, 40, 50, 60, 80, 100, 120], dtype=np.float64)
    trueJetPtBins = np.array([0, 30, 40, 60, 80, 100, 120, 160], dtype=np.float64)
    smearedSplittingVariableBins = np.array([1, 2, 3, 4, 5, 7, 10, 15], dtype=np.float64)
    # NOTE: (-0.05, 0) is the untagged bin.
    trueSplittingVariableBins = np.array([-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100], dtype=np.float64)

    h1_test = ROOT.TH1D("smeared_1", "smeared_1", len(smearedJetPtBins) - 1, smearedJetPtBins)
    h2_smeared = ROOT.TH2D(
        "smeared",
        "smeared",
        len(smearedSplittingVariableBins) - 1,
        smearedSplittingVariableBins,
        len(smearedJetPtBins) - 1,
        smearedJetPtBins,
    )
    h2_true = ROOT.TH2D(
        "true",
        "true",
        len(trueSplittingVariableBins) - 1,
        trueSplittingVariableBins,
        len(trueJetPtBins) - 1,
        trueJetPtBins,
    )
    response = ROOT.RooUnfoldResponse()
    response.Setup(h2_smeared, h2_true)
    # df_embedded.Fill(response, ["hybridSubstructureVariableValue", jet_pt_column_format.format(prefix=smeared_cut_prefix), "trueSubstructureVariable", jet_pt_column_format.format(prefix="true"), "scale_factor"])
    # df_embedded.Fill["double", "double", "double", "double"](response, ["hybridSubstructureVariableValue", jet_pt_column_format.format(prefix=smeared_cut_prefix), "trueSubstructureVariable", jet_pt_column_format.format(prefix="true")])
    df_embedded.Fill("double")(h1_test, ROOT.colName)

    logger.info("Done with loop")


if __name__ == "__main__":
    helpers.setup_logging()
    # run_unfolding_rdf(
    #    grooming_method="leading_kt",
    #    substructure_variable_name="kt",
    #    smeared_substructure_variable_bins=np.array(
    #        [1, 2, 3, 4, 5, 7, 10, 15],
    #        dtype=np.float64,
    #    ),
    #    smeared_jet_pt_bins=np.array(
    #        [30, 40, 50, 60, 80, 100, 120],
    #        dtype=np.float64,
    #    ),
    #    true_substructure_variable_bins=np.array(
    #        # NOTE: (-0.05, 0) is the untagged bin.
    #        [-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100],
    #        dtype=np.float64,
    #    ),
    #    true_jet_pt_bins=np.array(
    #        [0, 30, 40, 60, 80, 100, 120, 160],
    #        dtype=np.float64,
    #    ),
    # )

    run_unfolding_fall_back(
        settings=setup("dynamical_kt"),
        # smeared_substructure_variable_bins=np.array([1, 2, 3, 4, 5, 7, 10, 15], dtype=np.float64,),
        # smeared_jet_pt_bins=np.array([30, 40, 50, 60, 80, 100, 120], dtype=np.float64,),
        # true_substructure_variable_bins=np.array(
        #    # NOTE: (-0.05, 0) is the untagged bin.
        #    [-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100],
        #    dtype=np.float64,
        # ),
        # true_jet_pt_bins=np.array([0, 30, 40, 60, 80, 100, 120, 160], dtype=np.float64,),
    )
