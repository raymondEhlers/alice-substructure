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
#include <TH2F.h>
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

void RooSimplePbPb(TString cFiles2 = "files1.txt")
{
#ifdef __CINT__
  gSystem->Load("libRooUnfold");
#endif
  Int_t difference = 1;
  Int_t Ppol = 0;
  std::cout
   << "==================================== pick up the response matrix for background==========================\n";
  ///////////////////parameter setting
  RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

  // Setup
  ROOT::EnableImplicitMT();

  // Get the tree for data
  //TString fnamedata;
  //fnamedata = "../../trains/PbPb/5537/skim/*.root";
  TChain dataChain("tree");
  dataChain.Add("trains/PbPb/5537/skim/*.root");
  dataChain.ls();
  //TFile* inputdata;
  //inputdata = TFile::Open(fnamedata);
  //inputdata->ls();

  //***************************************************

  std::vector<double> smearedJetPtBins = {40, 50, 60, 80, 100, 120};
  std::vector<double> smearedKtBins = {0, 5, 10, 15, 20};
  std::vector<double> trueKtBins = {0, 5, 10, 15, 20};
  //Double_t xbins[6];
  //xbins[0] = 40;
  //xbins[1] = 50;
  //xbins[2] = 60;
  //xbins[3] = 80;
  //xbins[4] = 100;
  //xbins[5] = 120;

  Double_t xbinsb[6];

  xbinsb[0] = 0.;
  xbinsb[1] = 1;
  xbinsb[2] = 2;
  xbinsb[3] = 3;
  xbinsb[4] = 4;
  //	 xbinsb[5]=5;
  //	 xbinsb[6]=6;
  // xbinsb[7]=7;
  xbinsb[5] = 8;
  //         xbinsb[9]=9;
  //         xbinsb[10]=10;
  //         xbinsb[11]=11;

  // the raw correlation (ie. data)
  TH2F* h2raw = new TH2F("r", "raw", smearedKtBins.size() - 1, smearedKtBins.data(), smearedJetPtBins.size() - 1, smearedJetPtBins.data());
  // detector measure level (ie. hybrid)
  TH2F* h2smeared = new TH2F("smeared", "smeared", smearedKtBins.size() - 1, smearedKtBins.data(), smearedJetPtBins.size() - 1, smearedJetPtBins.data());
  // detector measure level no cuts (ie. hybrid, but no cuts)
  TH2F* h2smearednocuts = new TH2F("smearednocuts", "smearednocuts", smearedKtBins.size() - 1, smearedKtBins.data(), 8, 0, 160);
  // true correlations with measured cuts
  TH2F* h2true = new TH2F("true", "true", trueKtBins.size() - 1, trueKtBins.data(), 8, 0, 160);
  // full true correlation (without cuts)
  TH2F* h2fulleff = new TH2F("truef", "truef", trueKtBins.size() - 1, trueKtBins.data(), 8, 0, 160);

  TH2F* hcovariance = new TH2F("covariance", "covariance", 10, 0., 1., 10, 0, 1.);

  TH2F* effnum = (TH2F*)h2fulleff->Clone("effnum");
  TH2F* effdenom = (TH2F*)h2fulleff->Clone("effdenom");

  effnum->Sumw2();
  effdenom->Sumw2();
  h2smeared->Sumw2();
  h2true->Sumw2();
  h2raw->Sumw2();
  h2fulleff->Sumw2();

  // Read in the data and store the raw values.
  //TTreeReader dataReader("tree", inputdata);
  TTreeReader dataReader(&dataChain);

  std::string grooming_method = "leading_kt";
  std::string data_prefix = "data";

  TTreeReaderValue<float> ptJet(dataReader, ("jet_pt_" + data_prefix).c_str());
  TTreeReaderValue<float> kt(dataReader, (grooming_method + "_" + data_prefix + "_kt").c_str());
  while (dataReader.Next()) {
    if (*ptJet > 120 || *ptJet < 40) {
      continue;
    }
    h2raw->Fill(*kt, *ptJet);
  }

  //ifstream infile2;
  //infile2.open(cFiles2.Data());
  //char filename2[300];
  // Setup
  RooUnfoldResponse response;
  RooUnfoldResponse responsenotrunc;
  response.Setup(h2smeared, h2true);
  responsenotrunc.Setup(h2smearednocuts, h2fulleff);

  TChain embeddedChain("tree");
  // We are specific on the filenames to avoid the friend trees.
  // It appears that it can only handle one * per call.
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
  embeddedChain.ls();

  // Define the reader and process.
  std::string truePrefix = "true";
  std::string hybridPrefix = "hybrid";
  TTreeReader mcReader(&embeddedChain);
  TTreeReaderValue<float> scaleFactor(mcReader, "scale_factor");
  //TTreeReaderValue<float> ptJet(dataReader, "ptJet");
  TTreeReaderValue<float> hybridJetPt(mcReader, ("jet_pt_" + hybridPrefix).c_str());
  TTreeReaderValue<float> hybridKt(mcReader, (grooming_method + "_" + hybridPrefix + "_kt").c_str());
  TTreeReaderValue<float> trueJetPt(mcReader, ("jet_pt_" + truePrefix).c_str());
  TTreeReaderValue<float> trueKt(mcReader, (grooming_method + "_" + truePrefix + "_kt").c_str());

  while (mcReader.Next()) {
    if (*trueJetPt > 160) {
      continue;
    }

    h2fulleff->Fill(*trueKt, *trueJetPt, *scaleFactor);
    h2smearednocuts->Fill(*hybridKt, *hybridJetPt, *scaleFactor);
    responsenotrunc.Fill(*hybridKt, *hybridJetPt, *trueKt, *trueJetPt, *scaleFactor);

    if (*hybridJetPt > 120 || *hybridJetPt < 40) {
      continue;
    }
    // Also cut on hybrid kt
    if (*hybridKt < 2) {
      continue;
    }
    h2smeared->Fill(*hybridKt, *hybridJetPt, *scaleFactor);
    h2true->Fill(*trueKt, *trueJetPt, *scaleFactor);
    response.Fill(*hybridKt, *hybridJetPt, *trueKt, *trueJetPt, *scaleFactor);
  }

  /*while (infile2 >> filename2) {
    int pthardbin = 0;

    TFile* input = TFile::Open(filename2);
    TList* list = (TList*)input->Get("AliAnalysisTaskEmcalEmbeddingHelper_histos");
    TList* list2 = (TList*)list->FindObject("EventCuts");
    TH1D* hcent = (TH1D*)list2->FindObject("Centrality_selected");
    TProfile* hcross = (TProfile*)list->FindObject("fHistXsection");
    TH1D* htrials = (TH1D*)list->FindObject("fHistTrials");
    TH1D* hpthard = (TH1D*)list->FindObject("fHistPtHard");
    TH1D* hnevent = (TH1D*)list->FindObject("fHistEventCount");
    for (Int_t i = 1; i <= htrials->GetNbinsX(); i++) {
      if (htrials->GetBinContent(i) != 0) {
        pthardbin = i;
      }
    }
    double scalefactor =
     (hcross->Integral(pthardbin, pthardbin) * hcross->GetEntries()) / htrials->Integral(pthardbin, pthardbin);

    // Setup to fill Roounfold from MC tree.
    TTreeReader mcReader("AliAnalysisTaskJetDynamicalGrooming_RawTree_EventSub_Incl", input);
    TTreeReaderValue<float> ptJet(mcReader, "ptJet");
    TTreeReaderValue<float> ng(mcReader, "ng");
    TTreeReaderValue<float> zg(mcReader, "zg");
    TTreeReaderValue<float> rg(mcReader, "rg");
    TTreeReaderValue<float> kt(mcReader, "ktg");
    TTreeReaderValue<float> ptJetMatch(mcReader, "ptJetMatch");
    TTreeReaderValue<float> ngMatch(mcReader, "ngMatch");
    TTreeReaderValue<float> zgMatch(mcReader, "zgMatch");
    TTreeReaderValue<float> rgMatch(mcReader, "rgMatch");
    TTreeReaderValue<float> ktMatch(mcReader, "ktgMatch");

    while (mcReader.Next()) {
      if (*ptJetMatch > 160) {
        continue;
      }

      h2fulleff->Fill(*ngMatch, *ptJetMatch, scalefactor);
      h2smearednocuts->Fill(*ng, *ptJet, scalefactor);
      responsenotrunc.Fill(*ng, *ptJet, *rgMatch, *ptJetMatch, scalefactor);

      if (*ptJet > 120 || *ptJet < 40) {
        continue;
      }
      h2smeared->Fill(*ng, *ptJet, scalefactor);
      h2true->Fill(*ngMatch, *ptJetMatch, scalefactor);
      response.Fill(*ng, *ptJet, *ngMatch, *ptJetMatch, scalefactor);
    }
  }*/

  TH1F* htrueptd = (TH1F*)h2fulleff->ProjectionX("trueptd", 1, -1);
  TH1F* htruept = (TH1F*)h2fulleff->ProjectionY("truept", 1, -1);

  //    TH2D* hfold=(TH2D*)h2raw->Clone("hfold");
  //    hfold->Sumw2();

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

  TFile* fout = new TFile(Form("Unfoldng_ebye_sd2_R25_Feb3.root"), "RECREATE");
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

    RooUnfoldBayes unfold(&response, h2raw, iter); // OR
    TH2D* hunf = (TH2D*)unfold.Hreco(errorTreatment);
    // FOLD BACK
    TH1* hfold = response.ApplyToTruth(hunf, "");

    /* for(int i=0;i<11;i++){
 for(int j=0;j<5;j++){
double effects=0;
double error=0;
for(int k=0;k<11;k++){
for(int l=0;l<8;l++){

 int indexm=i+11*j;
 int indext=k+8*l;



 effects=effects+hunf->GetBinContent(k+1,l+1)*response(indexm,indext);
 error=error+hunf->GetBinError(k+1,l+1)*hunf->GetBinError(k+1,l+1)*response(indexm,indext)*response(indexm,indext);

}}
   hfold->SetBinContent(i+1,j+1,effects);
hfold->SetBinError(i+1,j+1,sqrt(error));
}}*/

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
