//#if !(defined(__CINT__) || defined(__CLING__)) || defined(__ACLIC__)
#include <iostream>
#include <memory>
#include <vector>

#include <TTree.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TNtuple.h>
#include <TPostScript.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TRandom.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>

#include <RooUnfoldBayes.h>
#include <RooUnfoldResponse.h>
//#endif

//=========================
// Utility functions
//=========================

/**
 * Correlation histogram for the substructure variable.
 *
 * Varies from the pt by the indexing of the covariance matrix.
 *
 * @param[in] cov Covariance matrix derived from the unfolding.
 * @param[in] name Name of the covariance matrix.
 * @param[in] title Title of the covariance matrix.
 * @param[in] na Number of x bins.
 * @param[in] nb Number of y bins.
 * @param[in] kbin Bin in the selected dimension.
 *
 * @returns The correlation histogram.
 */
TH2D* CorrelationHistSubstructureVar(const TMatrixD& cov, const char* name, const char* title, Int_t na, Int_t nb, Int_t kbin)
{
  TH2D* h = new TH2D(name, title, nb, 0, nb, nb, 0, nb);

  for (int l = 0; l < nb; l++) {
    for (int n = 0; n < nb; n++) {
      int index1 = kbin + na * l;
      int index2 = kbin + na * n;
      Double_t Vv = cov(index1, index1) * cov(index2, index2);
      if (Vv > 0.0)
        h->SetBinContent(l + 1, n + 1, cov(index1, index2) / sqrt(Vv));
    }
  }
  return h;
}

/**
 * Correlation histogram for the substructure variable.
 *
 * Varies from the substructure variable by the indexing of the covariance matrix.
 *
 * @param[in] cov Covariance matrix derived from the unfolding.
 * @param[in] name Name of the covariance matrix.
 * @param[in] title Title of the covariance matrix.
 * @param[in] na Number of x bins.
 * @param[in] nb Number of y bins.
 * @param[in] kbin Bin in the selected dimension.
 *
 * @returns The correlation histogram.
 */
TH2D* CorrelationHistPt(const TMatrixD& cov, const char* name, const char* title, Int_t na, Int_t nb, Int_t kbin)
{
  TH2D* h = new TH2D(name, title, na, 0, na, na, 0, na);

  for (int l = 0; l < na; l++) {
    for (int n = 0; n < na; n++) {
      int index1 = l + na * kbin;
      int index2 = n + na * kbin;
      Double_t Vv = cov(index1, index1) * cov(index2, index2);
      if (Vv > 0.0)
        h->SetBinContent(l + 1, n + 1, cov(index1, index2) / sqrt(Vv));
    }
  }
  return h;
}

/**
 * Calculate the correlation matrix for a given set of properties.
 *
 * As of October 2020, it's not used.
 *
 * @param[in] cov Covariance matrix derived from the unfolding.
 * @param[in] name Name of the covariance matrix.
 * @param[in] title Title of the covariance matrix.
 * @param[in] lo Lower bin limit.
 * @param[in] hi Higher bin limit.
 *
 * @returns The correlation histogram.
 */
TH2D* CorrelationHist(const TMatrixD& cov, const char* name, const char* title, Double_t lo, Double_t hi/*, Double_t lon, Double_t hin*/)
{
  Int_t nb = cov.GetNrows();
  Int_t na = cov.GetNcols();
  std::cout << nb << " " << na << "\n";
  TH2D* h = new TH2D(name, title, nb, 0, nb, na, 0, na);
  h->SetAxisRange(-1.0, 1.0, "Z");
  for (int i = 0; i < na; i++) {
    for (int j = 0; j < nb; j++) {
      Double_t Viijj = cov(i, i) * cov(j, j);
      if (Viijj > 0.0)
        h->SetBinContent(i + 1, j + 1, cov(i, j) / sqrt(Viijj));
    }
  }
  return h;
}

/**
 * Normalize a given 2D histogram.
 *
 * This is something that ROOT won't do itself...
 *
 * @param[in, out] h Histogram to be normalized.
 */
void Normalize2D(TH2* h)
{
  Int_t nbinsYtmp = h->GetNbinsY();
  const Int_t nbinsY = nbinsYtmp;
  Double_t norm[nbinsY];
  for (Int_t biny = 1; biny <= nbinsY; biny++) {
    norm[biny - 1] = 0;
    for (Int_t binx = 1; binx <= h->GetNbinsX(); binx++) {
      norm[biny - 1] += h->GetBinContent(binx, biny);
    }
  }

  for (Int_t biny = 1; biny <= nbinsY; biny++) {
    for (Int_t binx = 1; binx <= h->GetNbinsX(); binx++) {
      if (norm[biny - 1] == 0)
        continue;
      else {
        h->SetBinContent(binx, biny, h->GetBinContent(binx, biny) / norm[biny - 1]);
        h->SetBinError(binx, biny, h->GetBinError(binx, biny) / norm[biny - 1]);
      }
    }
  }
}

/**
 * Determine pt hard scale factor of the currently open file.
 *
 * This should only be necessary as a cross check, or when the scaling information isn't stored directly
 * in the tree (ie. if it's not from a skim).
 *
 * @param[in] f Current file.
 *
 * @returns Pt hard scale factor.
 */
double GetScaleFactor(TFile * f)
{
  // Retrieve the embedding helper to extract the cross section and ntrials.
  //auto task = dynamic_cast<TList*>(f->Get("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_Raw_EventSub_Incl"));
  auto task = dynamic_cast<TList*>(f->Get("AliAnalysisTaskEmcalEmbeddingHelper_histos"));

  auto hcross = dynamic_cast<TProfile*>(task->FindObject("fHistXsection"));
  auto htrials = dynamic_cast<TH1D*>(task->FindObject("fHistTrials"));

  //std::cout << "hcross=" << hcross << ", htrials=" << htrials << "\n";

  int ptHardBin = 0;
  for (Int_t i = 1; i <= htrials->GetXaxis()->GetNbins(); i++) {
    if (htrials->GetBinContent(i) != 0) {
      ptHardBin = i;
    }
  }
  // Code adapted from Leticia.
  // The values don't seem to be correct...
  /*double scaleFactor = hcross->Integral(ptHardBin, ptHardBin) / htrials->Integral(ptHardBin, ptHardBin);
  // Needs a -1 to offset for the bin counting.
  std::cout << f->GetName() << ": ptHardBin=" << (ptHardBin - 1) << ", scaleFactor=" << scaleFactor;*/

  // These values seem more reasonable...
  double crossSection = hcross->GetBinContent(ptHardBin) * hcross->GetEntries();
  double nTrials = htrials->GetBinContent(ptHardBin);
  double scaleFactor = crossSection / nTrials;
  std::cout << f->GetName() << ": ptHardBin=" << (ptHardBin - 1) << ", scaleFactor=" << scaleFactor << "\n";

  return scaleFactor;
}

/**
 * Determine if we have a pure match.
 *
 * @param[in] matchingLeading Matching status for the leading subjet.
 * @param[in] matchingSubleading Matching status for the subleading subjet.
 * @param[in] hybridSubstructureVariableValue Substructure variable value.
 * @param[in] smearedUntaggedBinValue Value that will be assigned to the untagged bin for checking if we have this value.
 *
 * @returns True if we have a pure match.
 */
inline bool isPureMatch(int matchingLeading, int matchingSubleading, double hybridSubstructureVariableValue, double smearedUntaggedBinValue)
{
  return (((std::abs(matchingLeading - 1) < 0.001) && (std::abs(matchingSubleading - 1) < 0.001)) ||
      (std::abs(hybridSubstructureVariableValue - smearedUntaggedBinValue) < 0.001));
}

//=========================
// Main functionality
//=========================

/**
 * Perform the actual unfolding.
 *
 * We separate this functionality so that we can run the unfolding with different input spectra for the same
 * response, such as the when we used the hybrid as the smeared input as a trivial closure.
 *
 * We also calculate the covariance matrix at a specified iteration.
 *
 * @param[in] response Roounfold response.
 * @param[in] h2true True spectra. Only used for binning, so retrieved as a const.
 * @param[in] inputSpectra Input spectra to be unfolded (for example, data). Spectra could be 2D: pt + kt (or Rg, etc...)
 * @param[in] errorTreatment Roounfold error treatment.
 * @param[in] fout ROOT output file. It's never called explicitly, but I like to pass it because we're implicitly writing to it.
 * @param[in] tag Tag to be prepended to all histograms generated in the unfolding. Default: "".
 * @param[in] nIter Number of iterations. Default: 10.
 * @param[in] nIterForCovariance Number of iterations where we will calculate the covariance. Default: 8.
 */
void Unfold2D(RooUnfoldResponse & response, const TH2D & h2true, TH2D & inputSpectra,
    RooUnfold::ErrorTreatment errorTreatment, TFile * fout, std::string tag = "", const int nIter = 10, const int nIterForCovariance = 8)
{
  std::cout << "\n=======================================================\n"
       << "Unfolding for tag \"" << tag << "\".\n";
  // Determine the tag. If we have a non-empty tag, we append it to all of the histograms.
  if (tag != "") {
    tag += "_";
  }

  for (int iter = 1; iter < nIter; iter++) {
    std::cout << "Iteration " << iter << "\n";

    // Setup the response for unfolding.
    RooUnfoldBayes unfold(&response, &inputSpectra, iter);
    // And then unfold.
    TH2D* hunf = dynamic_cast<TH2D*>(unfold.Hreco(errorTreatment));

    // Refold the truth (ie. fold back).
    TH1* hfold = response.ApplyToTruth(hunf, "");

    // Clone unfolded and refolded hists to write to the output file.
    TH2D* htempUnf = dynamic_cast<TH2D*>(hunf->Clone(TString::Format("%sBayesian_Unfoldediter%d", tag.c_str(), iter)));
    TH2D* htempFold = dynamic_cast<TH2D*>(hfold->Clone(TString::Format("%sBayesian_Foldediter%d", tag.c_str(), iter)));
    htempUnf->Write();
    htempFold->Write();

    // Retrieve the covariance matrix. Only for a selected iteration.
    if (iter == nIterForCovariance) {
      TMatrixD covmat = unfold.Ereco(RooUnfold::kCovariance);
      // Substructure variable.
      for (Int_t k = 0; k < h2true.GetNbinsX(); k++) {
        auto hCorr = dynamic_cast<TH2D*>(CorrelationHistSubstructureVar(covmat, TString::Format("%scorr%d", tag.c_str(), k), "Covariance matrix",
                             h2true.GetNbinsX(), h2true.GetNbinsY(), k));
        auto covSubstructureVar = dynamic_cast<TH2D*>(hCorr->Clone("covSubstructureVar"));
        covSubstructureVar->SetName(TString::Format("%spearsonmatrix_iter%d_binSubstructureVar%d", tag.c_str(), iter, k));
        covSubstructureVar->SetDrawOption("colz");
        covSubstructureVar->Write();
      }

      // Jet pt.
      for (Int_t k = 0; k < h2true.GetNbinsY(); k++) {
        auto hCorr = dynamic_cast<TH2D*>(CorrelationHistPt(covmat, TString::Format("%scorr%dpt", tag.c_str(), k), "Covariance matrix",
                            h2true.GetNbinsX(), h2true.GetNbinsY(), k));
        auto covpt = dynamic_cast<TH2D*>(hCorr->Clone("covpt"));
        covpt->SetName(TString::Format("%spearsonmatrix_iter%d_binpt%d", tag.c_str(), iter, k));
        covpt->SetDrawOption("colz");
        covpt->Write();
      }
    }
  }

  std::cout << "Finished unfolding!\n"
       << "=======================================================\n";
}

/**
 * Substructure unfolding variable.
 */
enum UnfoldingType_t {
  kt = 0,
  zg = 1,
  rg = 2
};

/**
 * Wrapper around the RooUnfoldResponse. Just for convenience.
 *
 * Since this wrapper is passed back to python, the field are named for Python.
 */
struct ResponseResult {
  std::shared_ptr<RooUnfoldResponse> response;
  std::shared_ptr<RooUnfoldResponse> response_no_trunc;
};

/**
 * Interface to 2D unfolding with python.
 *
 */
ResponseResult create_response_2D(
    std::map<std::string, TH2D *> hists,
    const std::string groomingMethod,
    const std::string substructureVariableName,
    std::vector<double> smearedJetPtBins,
    std::vector<double> trueJetPtBins,
    std::vector<double> smearedSplittingVariableBins,
    std::vector<double> trueSplittingVariableBins,
    double smearedUntaggedBinValue,
    double minSmearedSplittingVariable,
    double maxSmearedSplittingVariable,
    const std::vector<std::string> & dataFilenames,
    const std::vector<std::string> & embeddedFilenames,
    const bool usePureMatches = false,
    const std::string & dataTreeName = "tree",
    const std::string & dataPrefix = "data",
    const std::string & embeddedTreeName = "tree",
    const std::string & truePrefix = "true",
    const std::string & hybridPrefix = "hybrid",
    const std::string & detLevelPrefix = "det_level"
    )
{
  // First, we handle the data. Setup the Reader, the columns, and store the data in the appropriate hists.
  TChain dataChain(dataTreeName.c_str());
  for (auto filename : dataFilenames) {
    dataChain.Add(filename.c_str());
  }
  TTreeReader dataReader(&dataChain);

  TTreeReaderValue<double> dataJetPt(dataReader, (dataPrefix + "_jet_pt").c_str());
  TTreeReaderValue<double> dataSubstructureVariable(dataReader, (groomingMethod + "_" + dataPrefix + "_" + substructureVariableName).c_str());
  while (dataReader.Next()) {
    // Jet pt cut.
    if (*dataJetPt < smearedJetPtBins[0] || *dataJetPt > smearedJetPtBins[smearedJetPtBins.size() - 1]) {
      continue;
    }
    // Substructure variable cut.
    double dataSubstructureVariableValue = *dataSubstructureVariable;
    if (dataSubstructureVariableValue < 0) {
      // Assign to the untagged bin.
      dataSubstructureVariableValue = smearedUntaggedBinValue;
    }
    else {
      if (dataSubstructureVariableValue < minSmearedSplittingVariable || dataSubstructureVariableValue > maxSmearedSplittingVariable) {
        continue;
      }
    }
    hists["h2_raw"]->Fill(dataSubstructureVariableValue, *dataJetPt);
  }

  // Now, switch to embedded
  // First, setup the response
  // NOTE: We allocate a shared_ptr, but don't delete here because we want to return the response
  //       without copying. If we do copy, RooUnfold doesn't seem to behave identically. It may not make
  //       a difference, but better not to tempt fate. Instead, we pass ownership to the caller.
  auto response = std::make_shared<RooUnfoldResponse>();
  auto responsenotrunc = std::make_shared<RooUnfoldResponse>();
  response->Setup(hists["h2_smeared"], hists["h2_true"]);
  responsenotrunc->Setup(hists["h2_smeared_no_cuts"], hists["h2_full_eff"]);

  // Next, we setup the Reader, the columns, and store the data in the appropriate hists.
  TChain embeddedChain(embeddedTreeName.c_str());
  for (auto filename : embeddedFilenames) {
    embeddedChain.Add(filename.c_str());
  }
  TTreeReader mcReader(&embeddedChain);

  // Values
  TTreeReaderValue<double> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<double> hybridJetPt(mcReader, (hybridPrefix + "_jet_pt").c_str());
  TTreeReaderValue<double> hybridSubstructureVariable(mcReader, (groomingMethod + "_" + hybridPrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<double> trueJetPt(mcReader, (truePrefix + "_jet_pt").c_str());
  TTreeReaderValue<double> trueSubstructureVariable(mcReader, (groomingMethod + "_" + truePrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<long long> matchingLeading(mcReader, (groomingMethod + "_hybrid_det_level_matching_leading").c_str());
  TTreeReaderValue<long long> matchingSubleading(mcReader, (groomingMethod + "_hybrid_det_level_matching_subleading").c_str());
  // For the double counting cut.
  TTreeReaderValue<double> hybridUnsubLeadingTrackPt(mcReader, (hybridPrefix + "_leading_track_pt").c_str());
  TTreeReaderValue<double> detLevelLeadingTrackPt(mcReader, (detLevelPrefix + "_leading_track_pt").c_str());

  int treeNumber = -1;
  //double scaleFactor = 0;
  while (mcReader.Next()) {
    // Check if the file changed.
    if (treeNumber < embeddedChain.GetTreeNumber()) {
      // File changed. Update the scale factor.
      //auto f = embeddedChain.GetFile();
      //scaleFactor = GetScaleFactor(f);
      // Update the tree number so we hold onto the scale factor until the next time we need to update.
      treeNumber = embeddedChain.GetTreeNumber();
    }
    // Ensure that we are in the right true pt and substructure variable range.
    if (*trueJetPt > trueJetPtBins[trueJetPtBins.size() - 1]) {
      continue;
    }
    if (*trueSubstructureVariable > trueSplittingVariableBins[trueSplittingVariableBins.size() - 1]) {
      continue;
    }
    // Double counting cut
    if (*hybridUnsubLeadingTrackPt > *detLevelLeadingTrackPt) {
      continue;
    }

    // Full efficiency hists (and response).
    hists["h2_full_eff"]->Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    hists["h2_smeared_no_cuts"]->Fill(*hybridSubstructureVariable, *hybridJetPt, *scaleFactor);
    responsenotrunc->Fill(*hybridSubstructureVariable, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor);

    // Now start making cuts on the hybrid level.
    // Jet pt
    if (*hybridJetPt < smearedJetPtBins[0] || *hybridJetPt > smearedJetPtBins[smearedJetPtBins.size() - 1]) {
      continue;
    }
    // Also cut on hybrid substructure variable.
    double hybridSubstructureVariableValue = *hybridSubstructureVariable;
    if (hybridSubstructureVariableValue < 0) {
      // Assign to the untagged bin.
      hybridSubstructureVariableValue = smearedUntaggedBinValue;
    }
    else {
      if (hybridSubstructureVariableValue < minSmearedSplittingVariable || hybridSubstructureVariableValue > maxSmearedSplittingVariable) {
        continue;
      }
    }
    // Matching cuts: Requiring a pure match.
    if (usePureMatches && !isPureMatch(*matchingLeading, *matchingSubleading, hybridSubstructureVariableValue, smearedUntaggedBinValue)) {
      continue;
    }

    // At this point, we've passed all of our cuts, so we store the result.
    hists["h2_smeared"]->Fill(hybridSubstructureVariableValue, *hybridJetPt, *scaleFactor);
    hists["h2_true"]->Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    hists["h2_substructure_variable"]->Fill(hybridSubstructureVariableValue, *trueSubstructureVariable, *scaleFactor);
    response->Fill(hybridSubstructureVariableValue, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor);
  }

  return ResponseResult{response, responsenotrunc};
}

/**
 * Find reweighting bin, ensuring that we take the last good bin if we're out of range.
 *
 * For example, if we're below the lowest bin, then we take the first bin. Effectively,
 * we neglect the overflow bins.
 *
 * @param[in] value Value that we want to reweight.
 * @param[in] bins Binning of the reweighting hist.
 *
 * @returns Bin to use when retrieving the reweighting value.
 */
int findReweightingBin(double value, const std::vector<double>& bins)
{
  auto bin = std::distance(bins.begin(), std::upper_bound(bins.begin(), bins.end(), value));
  if (bin == 0) {
    ++bin;
  }
  if (bin >= bins.size()) {
    --bin;
  }

  return bin;
}

enum ClosureVariation_t {
  splitMC = 0,
  reweightPseudoData = 1,
  reweightResponse = 2
};

/**
 * Create a reweighted response (or pseudo-data).
 *
 * Since we utilize pseudo-data here, we split statistics according to a specified fraction.
 * Consequently, this also covers the split MC closure.
 *
 * NOTE: We rely on the user to validate that the binning matches the reweighting hist.
 *
 */
ResponseResult create_closure_response_2D(
    std::map<std::string, TH2D *> hists,
    const std::string groomingMethod,
    const std::string substructureVariableName,
    std::vector<double> smearedJetPtBins,
    std::vector<double> trueJetPtBins,
    std::vector<double> smearedSplittingVariableBins,
    std::vector<double> trueSplittingVariableBins,
    double smearedUntaggedBinValue,
    double minSmearedSplittingVariable,
    double maxSmearedSplittingVariable,
    const std::vector<std::string> & embeddedFilenames,
    const ClosureVariation_t closureVariation,
    const double fractionForResponse = 0.75,
    const bool usePureMatches = false,
    TH2D * hReweighting = nullptr,
    const std::string & embeddedTreeName = "tree",
    const std::string & truePrefix = "true",
    const std::string & hybridPrefix = "hybrid",
    const std::string & detLevelPrefix = "det_level"
    )
{
  // NOTE: We rely on the user to validate that the binning matches the reweighting hist.
  // Setup
  TRandom3 random(0);

  // We don't have to deal with data for this closure test. It's all based around the embedded data.
  // So we go directly to the response.
  // First, setup the response
  // NOTE: We allocate a shared_ptr, but don't delete here because we want to return the response
  //       without copying. If we do copy, RooUnfold doesn't seem to behave identically. It may not make
  //       a difference, but better not to tempt fate. Instead, we pass ownership to the caller.
  auto response = std::make_shared<RooUnfoldResponse>();
  auto responsenotrunc = std::make_shared<RooUnfoldResponse>();
  response->Setup(hists["h2_smeared"], hists["h2_true"]);
  responsenotrunc->Setup(hists["h2_smeared_no_cuts"], hists["h2_full_eff"]);

  // Next, we setup the Reader, the columns, and store the data in the appropriate hists.
  TChain embeddedChain(embeddedTreeName.c_str());
  for (auto filename : embeddedFilenames) {
    embeddedChain.Add(filename.c_str());
  }
  TTreeReader mcReader(&embeddedChain);

  // Values
  TTreeReaderValue<double> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<double> hybridJetPt(mcReader, (hybridPrefix + "_jet_pt").c_str());
  TTreeReaderValue<double> hybridSubstructureVariable(mcReader, (groomingMethod + "_" + hybridPrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<double> trueJetPt(mcReader, (truePrefix + "_jet_pt").c_str());
  TTreeReaderValue<double> trueSubstructureVariable(mcReader, (groomingMethod + "_" + truePrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<long long> matchingLeading(mcReader, (groomingMethod + "_hybrid_det_level_matching_leading").c_str());
  TTreeReaderValue<long long> matchingSubleading(mcReader, (groomingMethod + "_hybrid_det_level_matching_subleading").c_str());
  // For the double counting cut.
  TTreeReaderValue<double> hybridUnsubLeadingTrackPt(mcReader, (hybridPrefix + "_leading_track_pt").c_str());
  TTreeReaderValue<double> detLevelLeadingTrackPt(mcReader, (detLevelPrefix + "_leading_track_pt").c_str());

  int treeNumber = -1;
  //double scaleFactor = 0;
  while (mcReader.Next()) {
    // Check if the file changed.
    if (treeNumber < embeddedChain.GetTreeNumber()) {
      // File changed. Update the scale factor.
      //auto f = embeddedChain.GetFile();
      //scaleFactor = GetScaleFactor(f);
      // Update the tree number so we hold onto the scale factor until the next time we need to update.
      treeNumber = embeddedChain.GetTreeNumber();
    }
    // Ensure that we are in the right true pt and substructure variable range.
    if (*trueJetPt > trueJetPtBins[trueJetPtBins.size() - 1]) {
      continue;
    }
    if (*trueSubstructureVariable > trueSplittingVariableBins[trueSplittingVariableBins.size() - 1]) {
      continue;
    }
    // Double counting cut
    if (*hybridUnsubLeadingTrackPt > *detLevelLeadingTrackPt) {
      continue;
    }

    // Full efficiency hists (and response).
    hists["h2_full_eff"]->Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    hists["h2_smeared_no_cuts"]->Fill(*hybridSubstructureVariable, *hybridJetPt, *scaleFactor);
    responsenotrunc->Fill(*hybridSubstructureVariable, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor);

    // Now start making cuts on the hybrid level.
    // Jet pt
    if (*hybridJetPt < smearedJetPtBins[0] || *hybridJetPt > smearedJetPtBins[smearedJetPtBins.size() - 1]) {
      continue;
    }
    // Also cut on hybrid substructure variable.
    double hybridSubstructureVariableValue = *hybridSubstructureVariable;
    if (hybridSubstructureVariableValue < 0) {
      // Assign to the untagged bin.
      hybridSubstructureVariableValue = smearedUntaggedBinValue;
    }
    else {
      if (hybridSubstructureVariableValue < minSmearedSplittingVariable || hybridSubstructureVariableValue > maxSmearedSplittingVariable) {
        continue;
      }
    }

    // Potentially Reweight
    // NOTE: We intentionally look at the true values even though it's binned at detector level.
    // We need to handle the binning carefully, so we use a dedicated function.
    int ktBin = findReweightingBin(*trueSubstructureVariable, smearedSplittingVariableBins);
    int jetPtBin = findReweightingBin(*trueJetPt, smearedJetPtBins);
    // NOTE: In the case of the split MC, both reweight factors will automatically be 1.
    //       However, we set the factor to 1 here just to be safe.
    double reweightFactor = closureVariation == ClosureVariation_t::splitMC ? 1 : hReweighting->GetBinContent(ktBin, jetPtBin);

    // The matching cuts should only be applied to the response, so we start storing hists here.
    double randomValue = random.Rndm();
    if (randomValue >= fractionForResponse) {
      // Variation 1 is where we reweight the pseudo-data.
      double pseudoReweightFactor = closureVariation == ClosureVariation_t::reweightPseudoData ? reweightFactor : 1;
      hists["h2_pseudo_data"]->Fill(hybridSubstructureVariableValue, *hybridJetPt, *scaleFactor * pseudoReweightFactor);
      hists["h2_pseudo_true"]->Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor * pseudoReweightFactor);
    }

    // Matching cuts: Requiring a pure match.
    if (usePureMatches && !isPureMatch(*matchingLeading, *matchingSubleading, hybridSubstructureVariableValue, smearedUntaggedBinValue)) {
      continue;
    }

    // Determine the response reweighting factor.
    // Variation 2 is where we reweight the response.
    double responseReweightFactor = closureVariation == ClosureVariation_t::reweightResponse ? reweightFactor : 1;

    // At this point, we've passed all of our cuts, so we store the result.
    // These don't matter so terribly much for our closure test, but it's not a bad thing to have.
    // Note that they will always get the full stats and correspond to the response, as is the convention for the others.
    hists["h2_smeared"]->Fill(hybridSubstructureVariableValue, *hybridJetPt, *scaleFactor * responseReweightFactor);
    hists["h2_true"]->Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor * responseReweightFactor);
    hists["h2_substructure_variable"]->Fill(hybridSubstructureVariableValue, *trueSubstructureVariable, *scaleFactor * responseReweightFactor);

    // We've filled the pseudo data above (before the pure matches requirement), but we still
    // need to fill the response when appropriate.
    if (randomValue < fractionForResponse) {
      response->Fill(hybridSubstructureVariableValue, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor * responseReweightFactor);
    }
  }

  return ResponseResult{response, responsenotrunc};
}

