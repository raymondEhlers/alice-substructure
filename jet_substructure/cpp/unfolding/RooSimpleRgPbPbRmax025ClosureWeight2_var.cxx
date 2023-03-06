//=====================================================================-*-C++-*-
// File and Version Information:
//      $Id: RooUnfoldExample.cxx 279 2011-02-11 18:23:44Z T.J.Adye $
//
// Description:
//      Simple example usage of the RooUnfold package using toy MC.
//
// Authors: Tim Adye <T.J.Adye@rl.ac.uk> and Fergus Wilson <fwilson@slac.stanford.edu>
//
//==============================================================================

#if !(defined(__CINT__) || defined(__CLING__)) || defined(__ACLIC__)
#include <iostream>
using std::cout;
using std::endl;

#include "RooUnfoldBayes.h"
#include "RooUnfoldResponse.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TLegend.h"
#include "TLine.h"
#include "TNtuple.h"
#include "TPostScript.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TRandom.h"
#include "TString.h"
#include "TStyle.h"
#include "TVectorD.h"
//#include "RooUnfoldTestHarness2D.h"
#endif

//==============================================================================
// Global definitions
//==============================================================================

const Double_t cutdummy = -99999.0;

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
  cout << nb << " " << na << endl;
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

void RooSimpleRgPbPbRmax025ClosureWeight2_var(
 TString cFiles2 =
  "/Users/leticiacunqueiro/2019RawData/SmearedReferenceSemiEventWiseHard0.4Rmax025Matching/files1.txt")
{
#ifdef __CINT__
  gSystem->Load("libRooUnfold");
#endif
  Int_t difference = 1;
  Int_t Ppol = 0;
  cout << "==================================== pick up the response matrix for background=========================="
     << endl;
  ///////////////////parameter setting
  RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

  // Get the tree for data
  TString fnamedata;
  // fnamedata="semicentralevent.root";
  fnamedata =
   "/Users/leticiacunqueiro/2019RawData/SmearedReferenceSemiEventWiseHard0.4Rmax025/semihard04rmax025.root";
  // fnamedata="semicentralsjet-wise.root";
  TFile* inputdata;
  inputdata = TFile::Open(fnamedata);
  TTree* data = (TTree*)inputdata->Get(
   "JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_TCRawTree_Data_ConstSub_Incl");

  TFile* RatioFile = new TFile(
   "/Users/leticiacunqueiro/2019RawData/SmearedReferenceSemiEventWiseHard0.4Rmax025Matching/"
   "fileRatioDataMC_var.root");

  //***************************************************
  Double_t xbins[6];

  xbins[0] = 40;
  xbins[1] = 50;
  xbins[2] = 60;
  xbins[3] = 70;
  xbins[4] = 90;
  xbins[5] = 120;

  Double_t xbinsb[8];
  xbinsb[0] = -0.05;
  xbinsb[1] = 0.;
  xbinsb[2] = 0.02;
  xbinsb[3] = 0.04;
  xbinsb[4] = 0.06;
  xbinsb[5] = 0.1;
  xbinsb[6] = 0.2;
  xbinsb[7] = 0.35;

  Double_t xbinsc[9];
  xbinsc[0] = -0.05;
  xbinsc[1] = 0.;
  xbinsc[2] = 0.02;
  xbinsc[3] = 0.04;
  xbinsc[4] = 0.06;
  xbinsc[5] = 0.1;
  xbinsc[6] = 0.2;
  xbinsc[7] = 0.35;
  xbinsc[8] = 0.6;

  // the raw correlation
  TH2D* h2raw(0);
  h2raw = new TH2D("raw", "raw", 7, xbinsb, 5, xbins);
  // detector measure level
  TH2D* h2smeared(0);
  h2smeared = new TH2D("smeared", "smeared", 7, xbinsb, 5, xbins);

  // detector measure level no cuts
  TH2D* h2smearednocuts(0);
  h2smearednocuts = new TH2D("smearednocuts", "smearednocuts", 8, xbinsc, 8, 0, 160);
  // true correlations with measured cuts
  TH2D* h2true(0);
  h2true = new TH2D("true", "true", 8, xbinsc, 8, 0, 160);
  // full true correlation
  TH2D* h2fulleff(0);
  h2fulleff = new TH2D("truef", "truef", 8, xbinsc, 8, 0, 160);

  TH2D* hcovariance(0);
  hcovariance = new TH2D("covariance", "covariance", 10, 0., 1., 10, 0, 1.);
  TH2D* h2trueb(0);
  h2trueb = new TH2D("pseudotrue", "pseudotrue", 8, xbinsc, 8, 0, 160);

  TH2D* h2smearedb(0);
  h2smearedb = new TH2D("pseudodata", "pseudodata", 7, xbinsb, 5, xbins);

  TH2D* effnum = (TH2D*)h2fulleff->Clone("effnum");
  TH2D* effdenom = (TH2D*)h2fulleff->Clone("effdenom");

  effnum->Sumw2();
  effdenom->Sumw2();
  h2smeared->Sumw2();
  h2true->Sumw2();
  h2raw->Sumw2();
  h2fulleff->Sumw2();
  h2trueb->Sumw2();
  h2smearedb->Sumw2();

  Float_t ptJet, ptJetMatch, zg, zgMatch, rg, rgMatch, ktg, ktgMatch, ng, ngMatch, LeadingTrackPt, LeadingTrackPtDet,
   ngDet, subjet1, subjet2;

  Int_t nEv = 0;
  ;
  // so mcr is correctly normalized to one, not the response.
  cout << "cucu" << endl;
  nEv = data->GetEntries();
  cout << "entries" << nEv << endl;
  data->SetBranchAddress("ptJet", &ptJet);
  data->SetBranchAddress("zg", &zg);
  data->SetBranchAddress("rg", &rg);
  data->SetBranchAddress("ktg", &ktg);
  data->SetBranchAddress("ng", &ng);

  for (int iEntry = 0; iEntry < nEv; iEntry++) {
    data->GetEntry(iEntry);
    if (ptJet > 120 || ptJet < 40)
      continue;
    if (rg > 0.35)
      continue;
    if (zg < 0.4)
      rg = -0.025;
    h2raw->Fill(rg, ptJet);
  }

  h2raw->Draw("text");

  TH2D* historatio;
  historatio = (TH2D*)RatioFile->Get("weight");

  ifstream infile2;
  infile2.open(cFiles2.Data());
  char filename2[300];
  RooUnfoldResponse response;
  RooUnfoldResponse responsenotrunc;
  response.Setup(h2smeared, h2true);
  responsenotrunc.Setup(h2smearednocuts, h2fulleff);
  gRandom = new TRandom3(0);

  while (infile2 >> filename2) {
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
      if (htrials->GetBinContent(i) != 0)
        pthardbin = i;
    }
    double scalefactor = hcross->Integral(pthardbin, pthardbin) / htrials->Integral(pthardbin, pthardbin);

    TFile* input2 = TFile::Open(filename2);
    TTree* mc = (TTree*)input2->Get(
     "JetSubstructure_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_TCRawTree_EventSub_Incl");
    Int_t nEv = mc->GetEntries();
    // Int_t nEv=10000;
    mc->SetBranchAddress("ptJet", &ptJet);
    mc->SetBranchAddress("ptJetMatch", &ptJetMatch);
    mc->SetBranchAddress("zg", &zg);
    mc->SetBranchAddress("ngDet", &ngDet);
    mc->SetBranchAddress("zgMatch", &zgMatch);
    mc->SetBranchAddress("rg", &rg);
    mc->SetBranchAddress("rgMatch", &rgMatch);
    mc->SetBranchAddress("ktg", &ktg);
    mc->SetBranchAddress("ktgMatch", &ktgMatch);
    mc->SetBranchAddress("ng", &ng);
    mc->SetBranchAddress("ngMatch", &ngMatch);
    mc->SetBranchAddress("LeadingTrackPt", &LeadingTrackPt);
    mc->SetBranchAddress("LeadingTrackPtDet", &LeadingTrackPtDet);
    mc->SetBranchAddress("subjet1", &subjet1);
    mc->SetBranchAddress("subjet2", &subjet2);
    Int_t countm = 0;
    for (int iEntry = 0; iEntry < nEv; iEntry++) {
      mc->GetEntry(iEntry);

      if (ptJetMatch > 160)
        continue;
      if (LeadingTrackPt > LeadingTrackPtDet)
        continue;
      if (rgMatch > 0.6)
        continue;
      if (zgMatch < 0.4)
        rgMatch = -0.025;
      h2fulleff->Fill(rgMatch, ptJetMatch, scalefactor);
      h2smearednocuts->Fill(rg, ptJet, scalefactor);
      responsenotrunc.Fill(rg, ptJet, rgMatch, ptJetMatch, scalefactor);

      if (ptJet > 120 || ptJet < 40)
        continue;
      if (rg > 0.4)
        continue;
      if (zg < 0.4)
        rg = -0.025;
      h2smeared->Fill(rg, ptJet, scalefactor);
      h2true->Fill(rgMatch, ptJetMatch, scalefactor);

      Double_t rnd = gRandom->Rndm();
      Double_t myw = 1;

      if (ptJetMatch < 40) {
        if (rgMatch >= -0.05 && rgMatch < 0)
          myw = historatio->GetBinContent(1, 1);
        if (rgMatch >= 0 && rgMatch < 0.02)
          myw = historatio->GetBinContent(2, 1);
        if (rgMatch >= 0.02 && rgMatch < 0.04)
          myw = historatio->GetBinContent(3, 1);
        if (rgMatch >= 0.04 && rgMatch < 0.06)
          myw = historatio->GetBinContent(4, 1);
        if (rgMatch >= 0.06 && rgMatch < 0.1)
          myw = historatio->GetBinContent(5, 1);
        if (rgMatch >= 0.1 && rgMatch < 0.2)
          myw = historatio->GetBinContent(6, 1);
        if (rgMatch >= 0.2)
          myw = historatio->GetBinContent(7, 1);
      }

      if (ptJetMatch >= 40 && ptJetMatch < 50) {
        if (rgMatch >= -0.05 && rgMatch < 0)
          myw = historatio->GetBinContent(1, 1);
        if (rgMatch >= 0 && rgMatch < 0.02)
          myw = historatio->GetBinContent(2, 1);
        if (rgMatch >= 0.02 && rgMatch < 0.04)
          myw = historatio->GetBinContent(3, 1);
        if (rgMatch >= 0.04 && rgMatch < 0.06)
          myw = historatio->GetBinContent(4, 1);
        if (rgMatch >= 0.06 && rgMatch < 0.1)
          myw = historatio->GetBinContent(5, 1);
        if (rgMatch >= 0.1 && rgMatch < 0.2)
          myw = historatio->GetBinContent(6, 1);
        if (rgMatch >= 0.2)
          myw = historatio->GetBinContent(7, 1);
      }

      if (ptJetMatch >= 50 && ptJetMatch < 60) {
        if (rgMatch >= -0.05 && rgMatch < 0)
          myw = historatio->GetBinContent(1, 2);
        if (rgMatch >= 0 && rgMatch < 0.02)
          myw = historatio->GetBinContent(2, 2);
        if (rgMatch >= 0.02 && rgMatch < 0.04)
          myw = historatio->GetBinContent(3, 2);
        if (rgMatch >= 0.04 && rgMatch < 0.06)
          myw = historatio->GetBinContent(4, 2);
        if (rgMatch >= 0.06 && rgMatch < 0.1)
          myw = historatio->GetBinContent(5, 2);
        if (rgMatch >= 0.1 && rgMatch < 0.2)
          myw = historatio->GetBinContent(6, 2);
        if (rgMatch >= 0.2)
          myw = historatio->GetBinContent(7, 2);
      }

      if (ptJetMatch >= 60 && ptJetMatch < 70) {
        if (rgMatch >= -0.05 && rgMatch < 0)
          myw = historatio->GetBinContent(1, 3);
        if (rgMatch >= 0 && rgMatch < 0.02)
          myw = historatio->GetBinContent(2, 3);
        if (rgMatch >= 0.02 && rgMatch < 0.04)
          myw = historatio->GetBinContent(3, 3);
        if (rgMatch >= 0.04 && rgMatch < 0.06)
          myw = historatio->GetBinContent(4, 3);
        if (rgMatch >= 0.06 && rgMatch < 0.1)
          myw = historatio->GetBinContent(5, 3);
        if (rgMatch >= 0.1 && rgMatch < 0.2)
          myw = historatio->GetBinContent(6, 3);
        if (rgMatch >= 0.2)
          myw = historatio->GetBinContent(7, 3);
      }

      if (ptJetMatch >= 70 && ptJetMatch < 90) {
        if (rgMatch >= -0.05 && rgMatch < 0)
          myw = historatio->GetBinContent(1, 4);
        if (rgMatch >= 0 && rgMatch < 0.02)
          myw = historatio->GetBinContent(2, 4);
        if (rgMatch >= 0.02 && rgMatch < 0.04)
          myw = historatio->GetBinContent(3, 4);
        if (rgMatch >= 0.04 && rgMatch < 0.06)
          myw = historatio->GetBinContent(4, 4);
        if (rgMatch >= 0.06 && rgMatch < 0.1)
          myw = historatio->GetBinContent(5, 4);
        if (rgMatch >= 0.1 && rgMatch < 0.2)
          myw = historatio->GetBinContent(6, 4);
        if (rgMatch >= 0.2)
          myw = historatio->GetBinContent(7, 4);
      }

      if (ptJetMatch >= 90 && ptJetMatch < 120) {
        if (rgMatch >= -0.05 && rgMatch < 0)
          myw = historatio->GetBinContent(1, 5);
        if (rgMatch >= 0 && rgMatch < 0.02)
          myw = historatio->GetBinContent(2, 5);
        if (rgMatch >= 0.02 && rgMatch < 0.04)
          myw = historatio->GetBinContent(3, 5);
        if (rgMatch >= 0.04 && rgMatch < 0.06)
          myw = historatio->GetBinContent(4, 5);
        if (rgMatch >= 0.06 && rgMatch < 0.1)
          myw = historatio->GetBinContent(5, 5);
        if (rgMatch >= 0.1 && rgMatch < 0.2)
          myw = historatio->GetBinContent(6, 5);
        if (rgMatch >= 0.2)
          myw = historatio->GetBinContent(7, 5);
      }

      if (ptJetMatch >= 120) {
        if (rgMatch >= -0.05 && rgMatch < 0)
          myw = historatio->GetBinContent(1, 5);
        if (rgMatch >= 0 && rgMatch < 0.02)
          myw = historatio->GetBinContent(2, 5);
        if (rgMatch >= 0.02 && rgMatch < 0.04)
          myw = historatio->GetBinContent(3, 5);
        if (rgMatch >= 0.04 && rgMatch < 0.06)
          myw = historatio->GetBinContent(4, 5);
        if (rgMatch >= 0.06 && rgMatch < 0.1)
          myw = historatio->GetBinContent(5, 5);
        if (rgMatch >= 0.1 && rgMatch < 0.2)
          myw = historatio->GetBinContent(6, 5);
        if (rgMatch >= 0.2)
          myw = historatio->GetBinContent(7, 5);
      }

      if (rnd > 0.9) {
        h2smearedb->Fill(rg, ptJet, scalefactor);
        h2trueb->Fill(rgMatch, ptJetMatch, scalefactor);
        continue;
      }

      response.Fill(rg, ptJet, rgMatch, ptJetMatch, scalefactor * myw);
    }
  }

  TH1D* htrueptd = (TH1D*)h2fulleff->ProjectionX("trueptd", 1, -1);
  TH1D* htruept = (TH1D*)h2fulleff->ProjectionY("truept", 1, -1);

  TH2D* hfold = (TH2D*)h2raw->Clone("hfold");
  hfold->Sumw2();

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

  TH1D* effok9 = (TH1D*)h2true->ProjectionX("effok7", 6, 6);
  TH1D* effok10 = (TH1D*)h2fulleff->ProjectionX("effok8", 6, 6);
  effok9->Divide(effok10);
  effok9->SetName("correff100-120");

  TFile* fout = new TFile(Form("UnfoldRgzg0.4EventWiseDefaultRmax025ClosureWeight2_var.root"), "RECREATE");
  fout->cd();
  effok->Write();
  effok3->Write();
  effok5->Write();
  effok7->Write();
  effok9->Write();
  h2raw->SetName("raw");
  h2raw->Write();
  h2smeared->SetName("smeared");
  h2smeared->Write();
  h2fulleff->Write();
  htrueptd->Write();
  h2true->Write();
  h2trueb->Write();
  h2smearedb->Write();
  for (int jar = 1; jar < 16; jar++) {
    Int_t iter = jar;
    cout << "iteration" << iter << endl;
    cout << "==============Unfold h1=====================" << endl;

    RooUnfoldBayes unfold(&response, h2smearedb, iter); // OR
    TH2D* hunf = (TH2D*)unfold.Hreco(errorTreatment);
    // FOLD BACK
    TH2D* hfold = (TH2D*)response.ApplyToTruth(hunf, "");

    TH2D* htempUnf = (TH2D*)hunf->Clone("htempUnf");
    htempUnf->SetName(Form("Bayesian_Unfoldediter%d.root", iter));

    TH2D* htempFold = (TH2D*)hfold->Clone("htempFold");
    htempFold->SetName(Form("Bayesian_Foldediter%d.root", iter));

    htempUnf->Write();
    htempFold->Write();
  }
}

#ifndef __CINT__
int main()
{
  RooSimpleRgPbPbRmax025ClosureWeight2_var();
  return 0;
} // Main program when run stand-alone
#endif
