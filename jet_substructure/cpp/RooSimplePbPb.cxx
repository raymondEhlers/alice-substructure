#if !(defined(__CINT__) || defined(__CLING__)) || defined(__ACLIC__)
#include <iostream>
#include <vector>

#include <TTree.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TCanvas.h>
#include <TFile.h>
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
#include <TVectorD.h>

#include <RooUnfoldBayes.h>
#include <RooUnfoldResponse.h>
#endif

//==============================================================================
// Gaussian smearing, systematic translation, and variable inefficiency
//==============================================================================

TH2D* CorrelationHistShape(const TMatrixD& cov, const char* name, const char* title, Int_t na, Int_t nb, Int_t kbin)
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

TH2D* CorrelationHist(const TMatrixD& cov, const char* name, const char* title, Double_t lo, Double_t hi, Double_t lon,
           Double_t hin)
{
  Int_t nb = cov.GetNrows();
  Int_t na = cov.GetNcols();
  std::cout << nb << " " << na << "\n";
  TH2D* h = new TH2D(name, title, nb, 0, nb, na, 0, na);
  h->SetAxisRange(-1.0, 1.0, "Z");
  for (int i = 0; i < na; i++)
    for (int j = 0; j < nb; j++) {
      Double_t Viijj = cov(i, i) * cov(j, j);
      if (Viijj > 0.0)
        h->SetBinContent(i + 1, j + 1, cov(i, j) / sqrt(Viijj));
    }
  return h;
}

//==============================================================================
// Example Unfolding
//==============================================================================

void RooSimplePbPb()
{
#ifdef __CINT__
  gSystem->Load("libRooUnfold");
#endif
  std::cout
   << "==================================== pick up the response matrix for background==========================\n";
  ///////////////////parameter setting
  RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

  // Setup
  std::string groomingMethod = "leading_kt_z_cut_04";
  // If true, use hybrid as input data refolding test.
  bool hybridAsInputData = false;
  // If true, use pure matches
  bool usePureMatches = false;
  // Determine output filename
  std::string outputFilename = "unfolding_" + groomingMethod;
  if (hybridAsInputData == true) {
    outputFilename += "_hybrid_as_input";
  }
  if (usePureMatches == true) {
    outputFilename += "_pureMatches";
  }
  outputFilename += "_test.root";
  std::cout << "*********** Settings ***********\n\n";
  std::cout << "output filename: " << outputFilename << "\n";
  std::cout << "********************************\n\n";

  // Configuration (not totally clear if this actually does anything...)
  ROOT::EnableImplicitMT();

  //***************************************************

  // Define binning
  //std::vector<double> smearedJetPtBins = {40, 50, 60, 80, 100, 120};
  std::vector<double> smearedJetPtBins = {40, 50, 60, 70, 90, 120};
  std::vector<double> trueJetPtBins = {0, 20, 40, 60, 80, 100, 120, 140, 160};
  std::vector<double> smearedKtBins = {1, 2, 3, 4, 5, 7, 10, 15};
  std::vector<double> trueKtBins = {0, 1, 2, 3, 4, 5, 7, 10, 15, 100};

  // the raw correlation (ie. data)
  TH2D* h2raw = new TH2D("r", "raw", smearedKtBins.size() - 1, smearedKtBins.data(), smearedJetPtBins.size() - 1, smearedJetPtBins.data());
  // detector measure level (ie. hybrid)
  TH2D* h2smeared = new TH2D("smeared", "smeared", smearedKtBins.size() - 1, smearedKtBins.data(), smearedJetPtBins.size() - 1, smearedJetPtBins.data());
  // detector measure level no cuts (ie. hybrid, but no cuts).
  // NOTE: Strictly speaking, the y axis binning is at the hybrid level, but we want a wider range. So we use the trueJetPtBins.
  TH2D* h2smearednocuts = new TH2D("smearednocuts", "smearednocuts", smearedKtBins.size() - 1, smearedKtBins.data(), trueJetPtBins.size() - 1, trueJetPtBins.data());
  // true correlations with measured cuts
  TH2D* h2true = new TH2D("true", "true", trueKtBins.size() - 1, trueKtBins.data(), trueJetPtBins.size() - 1, trueJetPtBins.data());
  // full true correlation (without cuts)
  TH2D* h2fulleff = new TH2D("truef", "truef", trueKtBins.size() - 1, trueKtBins.data(), trueJetPtBins.size() - 1, trueJetPtBins.data());

  TH2D* hcovariance = new TH2D("covariance", "covariance", 10, 0., 1., 10, 0, 1.);

  TH2D* effnum = (TH2D*)h2fulleff->Clone("effnum");
  TH2D* effdenom = (TH2D*)h2fulleff->Clone("effdenom");

  effnum->Sumw2();
  effdenom->Sumw2();
  h2smeared->Sumw2();
  h2true->Sumw2();
  h2raw->Sumw2();
  h2fulleff->Sumw2();

  // Read the data and create the raw data hist.
  // First, setup the input data.
  TChain dataChain("tree");
  dataChain.Add("trains/PbPb/5537/skim/*.root");
  // Print out for logs (and to mirror Leticia).
  //dataChain.ls();
  TTreeReader dataReader(&dataChain);

  // Determines the type of data that we use. Usually, this is going to be "data" for raw data.
  std::string data_prefix = "data";

  TTreeReaderValue<float> dataJetPt(dataReader, ("jet_pt_" + data_prefix).c_str());
  TTreeReaderValue<float> dataKt(dataReader, (groomingMethod + "_" + data_prefix + "_kt").c_str());
  while (dataReader.Next()) {
    // Jet pt cut.
    if (*dataJetPt < smearedJetPtBins[0] || *dataJetPt > smearedJetPtBins[smearedJetPtBins.size() - 1]) {
      continue;
    }
    // Kt cut.
    if (*dataKt < smearedKtBins[0] || *dataKt > smearedKtBins[smearedKtBins.size() - 1]) {
      continue;
    }
    h2raw->Fill(*dataKt, *dataJetPt);
  }

  // Setup response tree.
  TChain embeddedChain("tree");
  // We are specific on the filenames to avoid the friend trees.
  // It appears that it can only handle one * per call. So we have to enuemrate each train.
  //embeddedChain.Add("temp_cache/embedPythia/55*/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5517/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5518/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5519/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5520/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5521/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5522/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5523/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5524/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5525/skim/*_iterative_splittings.root");
  embeddedChain.Add("temp_cache/embedPythia/5526/skim/*_iterative_splittings.root");
  //embeddedChain.Add("trains/embedPythia/55*/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5527/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5528/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5529/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5530/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5531/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5532/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5533/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5534/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5535/skim/*_iterative_splittings.root");
  embeddedChain.Add("trains/embedPythia/5536/skim/*_iterative_splittings.root");
  //embeddedChain.ls();

  // Define the reader and process.
  std::string truePrefix = "true";
  std::string hybridPrefix = "hybrid";
  TTreeReader mcReader(&embeddedChain);
  TTreeReaderValue<float> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<float> hybridJetPt(mcReader, ("jet_pt_" + hybridPrefix).c_str());
  TTreeReaderValue<float> hybridKt(mcReader, (groomingMethod + "_" + hybridPrefix + "_kt").c_str());
  TTreeReaderValue<float> trueJetPt(mcReader, ("jet_pt_" + truePrefix).c_str());
  TTreeReaderValue<float> trueKt(mcReader, (groomingMethod + "_" + truePrefix + "_kt").c_str());
  TTreeReaderValue<long long> matchingLeading(mcReader, (groomingMethod + "_hybrid_detector_matching_leading").c_str());
  TTreeReaderValue<long long> matchingSubleading(mcReader, (groomingMethod + "_hybrid_detector_matching_subleading").c_str());

  // Setup for the response
  RooUnfoldResponse response;
  RooUnfoldResponse responsenotrunc;
  response.Setup(h2smeared, h2true);
  responsenotrunc.Setup(h2smearednocuts, h2fulleff);

  while (mcReader.Next()) {
    // Ensure that we are in the right true pt and kt range.
    if (*trueJetPt > trueJetPtBins[trueJetPtBins.size() - 1]) {
      continue;
    }
    if (*trueKt > trueKtBins[trueKtBins.size() - 1]) {
      continue;
    }

    h2fulleff->Fill(*trueKt, *trueJetPt, *scaleFactor);
    h2smearednocuts->Fill(*hybridKt, *hybridJetPt, *scaleFactor);
    responsenotrunc.Fill(*hybridKt, *hybridJetPt, *trueKt, *trueJetPt, *scaleFactor);

    // Now start making cuts on the hybrid level.
    if (*hybridJetPt < smearedJetPtBins[0] || *hybridJetPt > smearedJetPtBins[smearedJetPtBins.size() - 1]) {
      continue;
    }
    // Also cut on hybrid kt
    if (*hybridKt < smearedKtBins[0] || *hybridKt > smearedKtBins[smearedKtBins.size() - 1]) {
      continue;
    }
    // Try matching cuts
    if (usePureMatches && !(*matchingLeading == 1 && *matchingSubleading == 1)) {
      continue;
    }
    h2smeared->Fill(*hybridKt, *hybridJetPt, *scaleFactor);
    h2true->Fill(*trueKt, *trueJetPt, *scaleFactor);
    response.Fill(*hybridKt, *hybridJetPt, *trueKt, *trueJetPt, *scaleFactor);
  }

  TH1D* htrueptd = (TH1D*)h2fulleff->ProjectionX("trueptd", 1, -1);
  TH1D* htruept = (TH1D*)h2fulleff->ProjectionY("truept", 1, -1);

  //////////efficiencies done////////////////////////////////////
  TH1D* effok = (TH1D*)h2true->ProjectionX("effok", 2, 2);
  TH1D* effok1 = (TH1D*)h2fulleff->ProjectionX("effok2", 2, 2);
  effok->Divide(effok1);
  effok->SetName("correff20-40");

  TH1D* effok3 = (TH1D*)h2true->ProjectionX("effok3", 3, 3);
  TH1D* effok4 = (TH1D*)h2fulleff->ProjectionX("effok4", 3, 3);
  effok3->Divide(effok4);
  effok3->SetName("correff40-60");

  TH1D* effok5 = (TH1D*)h2true->ProjectionX("effok5", 4, 4);
  TH1D* effok6 = (TH1D*)h2fulleff->ProjectionX("effok6", 4, 4);
  effok5->Divide(effok6);
  effok5->SetName("correff60-80");

  TH1D* effok7 = (TH1D*)h2true->ProjectionX("effok7", 5, 6);
  TH1D* effok8 = (TH1D*)h2fulleff->ProjectionX("effok8", 5, 6);
  effok7->Divide(effok8);
  effok7->SetName("correff80-120");

  TFile* fout = new TFile(outputFilename.c_str(), "RECREATE");
  fout->cd();
  effok->Write();
  effok3->Write();
  effok5->Write();
  effok7->Write();
  h2raw->SetName("raw");
  h2raw->Write();
  h2smeared->SetName("smeared");
  h2smeared->Write();
  htrueptd->Write();
  h2true->SetName("true");
  h2true->Write();
  h2fulleff->Write();
  for (int jar = 1; jar < 10; jar++) {
    Int_t iter = jar;
    std::cout << "iteration" << iter << "\n";
    std::cout << "==============Unfold h1====================="
         << "\n";

    // Allow for the possibility of using the hybrid as input data for closure.
    RooUnfoldBayes unfold(&response, (hybridAsInputData ? h2smeared : h2raw), iter); // OR
    TH2D* hunf = (TH2D*)unfold.Hreco(errorTreatment);
    // FOLD BACK
    TH1* hfold = response.ApplyToTruth(hunf, "");

    TH2D* htempUnf = (TH2D*)hunf->Clone("htempUnf");
    htempUnf->SetName(Form("Bayesian_Unfoldediter%d", iter));

    TH2D* htempFold = (TH2D*)hfold->Clone("htempFold");
    htempFold->SetName(Form("Bayesian_Foldediter%d", iter));

    htempUnf->Write();
    htempFold->Write();

    /// HERE I GET THE COVARIANCE MATRIX/////

    if (iter == 8) {
      TMatrixD covmat = unfold.Ereco((RooUnfold::ErrorTreatment)RooUnfold::kCovariance);
      for (Int_t k = 0; k < h2true->GetNbinsX(); k++) {
        TH2D* hCorr = (TH2D*)CorrelationHistShape(covmat, Form("corr%d", k), "Covariance matrix",
                             h2true->GetNbinsX(), h2true->GetNbinsY(), k);
        TH2D* covshape = (TH2D*)hCorr->Clone("covshape");
        covshape->SetName(Form("pearsonmatrix_iter%d_binshape%d", iter, k));
        covshape->SetDrawOption("colz");
        covshape->Write();
      }

      for (Int_t k = 0; k < h2true->GetNbinsY(); k++) {
        TH2D* hCorr = (TH2D*)CorrelationHistPt(covmat, Form("corr%d", k), "Covariance matrix",
                            h2true->GetNbinsX(), h2true->GetNbinsY(), k);
        TH2D* covpt = (TH2D*)hCorr->Clone("covpt");
        covpt->SetName(Form("pearsonmatrix_iter%d_binpt%d", iter, k));
        covpt->SetDrawOption("colz");
        covpt->Write();
      }
    }
  }
}

#ifndef __CINT__
int main()
{
  RooSimplePbPb();
  return 0;
} // Main program when run stand-alone
#endif
