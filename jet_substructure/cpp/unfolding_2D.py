""" 2D substructure unfolding implemented via RooUnfold.

"""

import logging
from pathlib import Path
from typing import Sequence

import attr
import numpy as np

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


@attr.s
class SubstructureVariable:
    name: str = attr.ib()
    variable_name: str = attr.ib()


def run_unfolding(
    grooming_method: str,
    substructure_variable_name: str,
    smeared_substructure_variable_bins: np.ndarray,
    smeared_jet_pt_bins: np.ndarray,
    true_substructure_variable_bins: np.ndarray,
    true_jet_pt_bins: np.ndarray,
    #data_filenames: Sequence[Path],
    #embedded_filenames: Sequence[Path],
    #output_filename: Path,
) -> bool:
    ...

    # Delayed import to avoid direct dependence.
    import ROOT

    # Configuration (not totally clear if this actually does anything for this script...)
    ROOT.ROOT.EnableImplicitMT(1)

    data_chain = ROOT.TChain("tree")
    data_chain.Add("trains/PbPb/5863/skim/*.root");

    # the raw correlation (ie. data)
    h2_raw = ROOT.TH2D("r", "raw", len(smeared_substructure_variable_bins) - 1, smeared_substructure_variable_bins, len(smeared_jet_pt_bins) - 1, smeared_jet_pt_bins);
    # detector measure level (ie. hybrid)
    h2_smeared = ROOT.TH2D("smeared", "smeared", len(smeared_substructure_variable_bins) - 1, smeared_substructure_variable_bins, len(smeared_jet_pt_bins) - 1, smeared_jet_pt_bins);
    # detector measure level no cuts (ie. hybrid, but no cuts).
    # NOTE: Strictly speaking, the y axis binning is at the hybrid level, but we want a wider range. So we use the true_jet_pt_bins.
    h2_smeared_no_cuts = ROOT.TH2D("smearednocuts", "smearednocuts", len(smeared_substructure_variable_bins) - 1, smeared_substructure_variable_bins, len(true_jet_pt_bins) - 1, true_jet_pt_bins);
    # true correlations with measured cuts
    h2_true = ROOT.TH2D("true", "true", len(true_substructure_variable_bins) - 1, true_substructure_variable_bins, len(true_jet_pt_bins) - 1, true_jet_pt_bins);
    # full true correlation (without cuts)
    h2_full_eff = ROOT.TH2D("truef", "truef", len(true_substructure_variable_bins) - 1, true_substructure_variable_bins, len(true_jet_pt_bins) - 1, true_jet_pt_bins);
    # Correlation between the splitting variables at true and hybrid (with cuts).
    h2_substructure_variable = ROOT.TH2D("h2SplittingVariable", "h2SplittingVariable", len(smeared_substructure_variable_bins) - 1, smeared_substructure_variable_bins, len(true_substructure_variable_bins) - 1, true_substructure_variable_bins);

    # TODO: Determine the untagged bin value

    data_prefix = "data";
    data_jet_pt_name = f"{data_prefix}_jet_pt"
    data_substructure_variable_name = f"{grooming_method}_{data_prefix}_{substructure_variable_name}"

    # TEMP for quick performance test
    smeared_untagged_bin_value = 2.5
    min_smeared_substructure_variable = 3
    max_smeared_substructure_variable = 15

    logger.info("Starting loop")
    #TTreeReaderValue<float> dataJetPt(dataReader, ("jet_pt_" + dataPrefix).c_str());
    #TTreeReaderValue<float> dataSubstructureVariable(dataReader, (groomingMethod + "_" + dataPrefix + "_" + substructureVariableName).c_str());
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
            if data_substructure_variable < min_smeared_substructure_variable or data_substructure_variable > max_smeared_substructure_variable:
                continue
        h2_raw.Fill(data_substructure_variable, data_jet_pt);
    logger.info("Done with loop")

    # Embedding


def run_unfolding_rdf(
    grooming_method: str,
    substructure_variable_name: str,
    smeared_substructure_variable_bins: np.ndarray,
    smeared_jet_pt_bins: np.ndarray,
    true_substructure_variable_bins: np.ndarray,
    true_jet_pt_bins: np.ndarray,
    #data_filenames: Sequence[Path],
    #embedded_filenames: Sequence[Path],
    #output_filename: Path,
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
    data_chain_data.Add("trains/PbPb/5863/skim/*.root");
    df_data = ROOT.RDataFrame(data_chain_data)

    # the raw correlation (ie. data)
    h2_raw_args = ("r", "raw", len(smeared_substructure_variable_bins) - 1, smeared_substructure_variable_bins, len(smeared_jet_pt_bins) - 1, smeared_jet_pt_bins);
    # detector measure level (ie. hybrid)
    h2_smeared_args = ("smeared", "smeared", len(smeared_substructure_variable_bins) - 1, smeared_substructure_variable_bins, len(smeared_jet_pt_bins) - 1, smeared_jet_pt_bins);
    # detector measure level no cuts (ie. hybrid, but no cuts).
    # NOTE: Strictly speaking, the y axis binning is at the hybrid level, but we want a wider range. So we use the true_jet_pt_bins.
    h2_smeared_no_cuts_args = ("smearednocuts", "smearednocuts", len(smeared_substructure_variable_bins) - 1, smeared_substructure_variable_bins, len(true_jet_pt_bins) - 1, true_jet_pt_bins);
    # true correlations with measured cuts
    h2_true_args = ("true", "true", len(true_substructure_variable_bins) - 1, true_substructure_variable_bins, len(true_jet_pt_bins) - 1, true_jet_pt_bins);
    # full true correlation (without cuts)
    h2_full_eff_args = ("truef", "truef", len(true_substructure_variable_bins) - 1, true_substructure_variable_bins, len(true_jet_pt_bins) - 1, true_jet_pt_bins);
    # Correlation between the splitting variables at true and hybrid (with cuts).
    h2_substructure_variable_args = ("h2SplittingVariable", "h2SplittingVariable", len(smeared_substructure_variable_bins) - 1, smeared_substructure_variable_bins, len(true_substructure_variable_bins) - 1, true_substructure_variable_bins);

    # TODO: Determine the untagged bin value
    # TODO: Make arguments, cleanup, consolidate...
    n_cores = 1;
    data_prefix = "data";
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
    logger.info(ROOT.randomSample(0, 12.))

    logger.info("Starting data frame")
    #TTreeReaderValue<float> dataJetPt(dataReader, ("jet_pt_" + dataPrefix).c_str());
    #TTreeReaderValue<float> dataSubstructureVariable(dataReader, (groomingMethod + "_" + dataPrefix + "_" + substructureVariableName).c_str());

    smeared_jet_pt_filter = f"{data_jet_pt_name} >= {smeared_jet_pt_bins[0]} && {data_jet_pt_name} <= {smeared_jet_pt_bins[-1]}"
    substructure_variable_value_filter = f"({data_substructure_variable_name} >= {min_smeared_substructure_variable} && {data_substructure_variable_name} <= {max_smeared_substructure_variable}) || ({data_substructure_variable_name} < 0)"
    df_data = df_data.Filter(f"({smeared_jet_pt_filter}) && ({substructure_variable_value_filter})")
    print(data_substructure_variable_name)
    df_data = df_data.Define("data_substructure_variable",
                             f"getSubstructureVariable({data_substructure_variable_name})",
                             #f"[](double {data_substructure_variable_name}) {{ return {data_substructure_variable_name} < 0 ? {smeared_untagged_bin_value} : {data_substructure_variable_name} }}",
                             #lambda substructure_variable: smeared_untagged_bin_value if substructure_variable < 0 else substructure_variable,
                             #lambda leading_kt_data_kt: smeared_untagged_bin_value if leading_kt_data_kt < 0 else leading_kt_data_kt,
                             #[data_substructure_variable_name]
                             )


    # Data
    h2_raw = df_data.Histo2D(
        h2_raw_args,
        "data_substructure_variable",
        data_jet_pt_name,
    )

    logger.info("Starting calculation")
    #logger.info(f"Entries: {h2_raw.GetEntries()}")

    # Starting embedding from here, but needs cleanup...
    data_chain_embedded = ROOT.TChain("tree")
    # TODO: Make these an argument...
    data_chain_embedded.Add("trains/embedPythia/5966/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5967/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5968/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5969/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5970/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5971/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5972/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5973/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5974/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5975/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5976/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5977/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5978/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5979/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5980/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5981/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5982/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5983/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5984/skim/*.root");
    data_chain_embedded.Add("trains/embedPythia/5985/skim/*.root");
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
    #df_embedded.ForeachSlot(ROOT.randomSample)
    #df_embedded.ForeachSlot(ROOT.randomSample, [jet_pt_column_format.format(prefix=smeared_cut_prefix)])
    #df_embedded.ForeachSlot(lambda slot, val: (slot, val), "jet_pt_hybrid")
    #df_embedded.ForeachSlot("randomSample", ROOT.colName)
    #df_embedded.Foreach("randomSample")

    # TEST
    smearedJetPtBins = np.array([30, 40, 50, 60, 80, 100, 120], dtype=np.float64)
    trueJetPtBins = np.array([0, 30, 40, 60, 80, 100, 120, 160], dtype=np.float64)
    smearedSplittingVariableBins = np.array([1, 2, 3, 4, 5, 7, 10, 15], dtype=np.float64)
    # NOTE: (-0.05, 0) is the untagged bin.
    trueSplittingVariableBins = np.array([-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100], dtype=np.float64)

    h1_test = ROOT.TH1D("smeared_1", "smeared_1", len(smearedJetPtBins) - 1, smearedJetPtBins)
    h2_smeared = ROOT.TH2D("smeared", "smeared", len(smearedSplittingVariableBins) - 1, smearedSplittingVariableBins, len(smearedJetPtBins) - 1, smearedJetPtBins)
    h2_true = ROOT.TH2D("true", "true", len(trueSplittingVariableBins) - 1, trueSplittingVariableBins, len(trueJetPtBins) - 1, trueJetPtBins)
    response = ROOT.RooUnfoldResponse()
    response.Setup(h2_smeared, h2_true)
    #df_embedded.Fill(response, ["hybridSubstructureVariableValue", jet_pt_column_format.format(prefix=smeared_cut_prefix), "trueSubstructureVariable", jet_pt_column_format.format(prefix="true"), "scale_factor"])
    #df_embedded.Fill["double", "double", "double", "double"](response, ["hybridSubstructureVariableValue", jet_pt_column_format.format(prefix=smeared_cut_prefix), "trueSubstructureVariable", jet_pt_column_format.format(prefix="true")])
    df_embedded.Fill("double")(h1_test, ROOT.colName)

    logger.info("Done with loop")



if __name__ == "__main__":
    helpers.setup_logging()
    run_unfolding_rdf(
        grooming_method="leading_kt",
        substructure_variable_name="kt",
        smeared_substructure_variable_bins=np.array(
            [1, 2, 3, 4, 5, 7, 10, 15],
            dtype=np.float64,
        ),
        smeared_jet_pt_bins=np.array(
            [30, 40, 50, 60, 80, 100, 120],
            dtype=np.float64,
        ),
        true_substructure_variable_bins=np.array(
            # NOTE: (-0.05, 0) is the untagged bin.
            [-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100],
            dtype=np.float64,
        ),
        true_jet_pt_bins=np.array(
            [0, 30, 40, 60, 80, 100, 120, 160],
            dtype=np.float64,
        ),
    )
