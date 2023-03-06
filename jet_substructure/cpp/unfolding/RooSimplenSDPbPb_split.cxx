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
#include "TRandom3.h"

#include "RooUnfoldBayes.h"
#include "RooUnfoldResponse.h"
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

void RooSimplenSDPbPb_split(TString cFiles2 = "files1.txt", std::string type = "nsd", std::string cut = "",
              std::string date = "")
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

  TRandom3* rand = new TRandom3(0);

  //***************************************************

  Double_t xbins[6];
  xbins[0] = 40;
  xbins[1] = 50;
  xbins[2] = 60;
  xbins[3] = 80;
  xbins[4] = 100;
  xbins[5] = 120;

  int Nn = 5;
  int Nz = 6;
  if (cut == "wide")
    Nz = 5;
  int Nr = 7;
  int N = -1;
  if (type == "nsd")
    N = Nn;
  else if (type == "zg")
    N = Nz;
  else if (type == "rg")
    N = Nr;
  Double_t nbins[N + 1];
  if (type != "rg")
    nbins[0] = 0.;
  else
    nbins[0] = -0.05;
  if (type == "nsd")
    nbins[1] = 1;
  else if (type == "zg")
    nbins[1] = 0.2;
  else
    nbins[1] = 0.;
  if (type == "nsd")
    nbins[2] = 2;
  else if (type == "zg") {
    if (cut != "")
      nbins[2] = 0.24;
    else
      nbins[2] = 0.25;
  } else
    nbins[2] = 0.02;
  if (type == "nsd")
    nbins[3] = 3;
  else if (type == "zg")
    nbins[3] = 0.3;
  else
    nbins[3] = 0.04;
  if (type == "nsd")
    nbins[4] = 4;
  else if (type == "zg") {
    if (cut != "")
      nbins[4] = 0.4;
    else
      nbins[4] = 0.35;
  } else
    nbins[4] = 0.06;
  if (type == "nsd")
    nbins[5] = 8;
  else if (type == "rg")
    nbins[5] = 0.1;
  else {
    if (cut != "")
      nbins[5] = 0.5;
    else
      nbins[5] = 0.4;
  }
  if (type == "rg") {
    nbins[6] = 0.2;
    nbins[7] = 0.35;
  }
  if ((type == "zg") && (cut == "")) {
    nbins[6] = 0.5;
  }

  Double_t nbins_true[9];
  nbins_true[0] = -0.05;
  nbins_true[1] = 0.;
  nbins_true[2] = 0.02;
  nbins_true[3] = 0.04;
  nbins_true[4] = 0.06;
  nbins_true[5] = 0.1;
  nbins_true[6] = 0.2;
  nbins_true[7] = 0.35;
  nbins_true[8] = 0.6;

  // the raw correlation
  TH2D* h2raw(0);
  h2raw = new TH2D("r", "raw", N, nbins, 5, xbins);
  // detector measure level
  TH2D* h2smeared(0);
  //  h2smeared=new TH2D("smeared","smeared",N,nbins,5,xbins);
  h2smeared = new TH2D("smeared", "smeared", N, nbins, 80, 0, 160);

  TH1D* h1smeared(0);
  h1smeared = new TH1D("h1smeared", "h1smeared", 5, xbins);
  TH1D* h1raw(0);
  h1raw = new TH1D("h1raw", "h1raw", 5, xbins);
  TH1D* h1true(0);
  //  h1true = new TH1D("h1true", "h1true", 8, 0, 160);
  h1true = new TH1D("h1true", "h1true", 80, 0, 160);
  TH1D* h1truefull(0);
  h1truefull = new TH1D("h1truef", "h1truef", 8, 0, 160);

  // detector measure level no cuts
  TH2D* h2smearednocuts(0);
  h2smearednocuts = new TH2D("smearednocuts", "smearednocuts", N, nbins, 8, 0, 160);
  // true correlations with measured cuts
  TH2D* h2true(0);
  //  if (type != "rg") h2true=new TH2D("true","true", N,nbins,8,0,160);
  //  else h2true=new TH2D("true","true", 8,nbins_true,8,0,160);
  if (type != "rg")
    h2true = new TH2D("true", "true", N, nbins, 80, 0, 160);
  else
    h2true = new TH2D("true", "true", 8, nbins_true, 80, 0, 160);
  TH2D* h2truefrac(0);
  if (type != "rg")
    h2truefrac = new TH2D("truefrac", "truefrac", N, nbins, 8, 0, 160);
  else
    h2truefrac = new TH2D("truefrac", "truefrac", 8, nbins_true, 8, 0, 160);
  // full true correlation
  TH2D* h2fulleff(0);
  if (type != "rg")
    h2fulleff = new TH2D("truef", "truef", N, nbins, 8, 0, 160);
  else
    h2fulleff = new TH2D("truef", "truef", 8, nbins_true, 8, 0, 160);

  TH2D* hcovariance(0);
  hcovariance = new TH2D("covariance", "covariance", 10, 0., 1., 10, 0, 1.);

  TH2D* effnum = (TH2D*)h2fulleff->Clone("effnum");
  TH2D* effdenom = (TH2D*)h2fulleff->Clone("effdenom");

  effnum->Sumw2();
  effdenom->Sumw2();
  h2smeared->Sumw2();
  h2true->Sumw2();
  h2raw->Sumw2();
  h2fulleff->Sumw2();

  Float_t ptJet, ptJetMatch, zg, zgMatch, rg, rgMatch, ktg, ktgMatch, ng, ngMatch, leadptMatch, leadpt, leadptDet;

  Int_t nEv = 0;
  ;
  // so mcr is correctly normalized to one, not the response.
  cout << "cucu" << endl;

  ifstream infile2;
  infile2.open(cFiles2.Data());
  char filename2[300];
  RooUnfoldResponse response;
  RooUnfoldResponse responsenotrunc;
  response.Setup(h2smeared, h2true);
  responsenotrunc.Setup(h2smearednocuts, h2fulleff);
  RooUnfoldResponse response1D;
  response1D.Setup(h1smeared, h1true);

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
    double scalefactor =
     (hcross->Integral(pthardbin, pthardbin) * hcross->GetEntries()) / htrials->Integral(pthardbin, pthardbin);

    TFile* input2 = TFile::Open(filename2);
    TTree* mc = (TTree*)input2->Get(
     "JetSubstructure_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_TCRawTree_EventSub_Incl");
    Int_t nEv = mc->GetEntries();

    mc->SetBranchAddress("ptJet", &ptJet);
    mc->SetBranchAddress("ptJetMatch", &ptJetMatch);
    mc->SetBranchAddress("zg", &zg);
    mc->SetBranchAddress("zgMatch", &zgMatch);
    mc->SetBranchAddress("rg", &rg);
    mc->SetBranchAddress("rgMatch", &rgMatch);
    mc->SetBranchAddress("ktg", &ktg);
    mc->SetBranchAddress("ktgMatch", &ktgMatch);
    mc->SetBranchAddress("ng", &ng);
    mc->SetBranchAddress("ngMatch", &ngMatch);
    mc->SetBranchAddress("LeadingTrackPtMatch", &leadptMatch);
    mc->SetBranchAddress("LeadingTrackPtDet", &leadptDet);
    mc->SetBranchAddress("LeadingTrackPt", &leadpt);
    Int_t countm = 0;
    for (int iEntry = 0; iEntry < nEv; iEntry++) {
      mc->GetEntry(iEntry);
      if (leadpt > leadptDet)
        continue;

      if (zg < 0.2)
        rg = -0.02;
      if (zgMatch < 0.2)
        rgMatch = -0.02;

      if (ptJetMatch > 160)
        continue;
      // if (ptJetMatch < 20) continue;
      if ((ngMatch >= 8) && (type == "nsd"))
        ngMatch = 7;
      //      if (ngMatch == 0) ngMatch = 0.5;
      if (cut == "wide") {
        if (rgMatch < 0.1)
          zgMatch = 0.05;
        if (rg < 0.1)
          zg = 0.05;
      }
      if (cut == "narrow") {
        if (rgMatch > 0.2)
          zgMatch = 0.05;
        if (rg > 0.2)
          zg = 0.05;
      }
      float varMatch = -1;
      float var = -1;
      if (type == "nsd") {
        varMatch = ngMatch;
        var = ng;
      } else if (type == "zg") {
        varMatch = zgMatch;
        var = zg;
      } else {
        varMatch = rgMatch;
        var = rg;
      }
      h2fulleff->Fill(varMatch, ptJetMatch, scalefactor);

      h2smearednocuts->Fill(var, ptJet, scalefactor);
      responsenotrunc.Fill(var, ptJet, varMatch, ptJetMatch, scalefactor);
      // response.Miss(ngMatch, ptJetMatch, scalefactor);
      h1truefull->Fill(ptJetMatch, scalefactor);

      response1D.Miss(ptJetMatch, scalefactor);

      /*      if (h2fulleff->Integral() < h1truefull->Integral() )
       {
     std::cout << "full eff int: " << h2fulleff->Integral() << std::endl;
     std::cout << "single eff int: " << h1truefull->Integral() << std::endl;
     std::cout << "pt: " << ptJetMatch << std::endl;
     std::cout << "scale: " << scalefactor << std::endl;
     std::cout << "ng: " << ngMatch << std::endl;
     }*/

      if (ptJet > 120 || ptJet < 40)
        continue;
      if (ptJetMatch < 10.)
        continue;
      // if (ptJet < 40) continue;
      if ((ng >= 8) && (type == "nsd"))
        ng = 7;
      // if(zg<0.2) rg=0.55;
      h2truefrac->Fill(varMatch, ptJetMatch, scalefactor);
      double split = rand->Rndm();
      if (split < 0.98) {
        h2smeared->Fill(var, ptJet, scalefactor);
        //	h2true->Fill(ngMatch,ptJetMatch,scalefactor);
        response.Fill(var, ptJet, varMatch, ptJetMatch, scalefactor);
        response1D.Fill(ptJet, ptJetMatch, scalefactor);
      } else {
        h2raw->Fill(var, ptJet, scalefactor);
        h2true->Fill(varMatch, ptJetMatch, scalefactor);
        h1true->Fill(ptJetMatch, scalefactor);
        h1raw->Fill(ptJet, scalefactor);
      }
    }
  }

  TH1D* htrueptd = (TH1D*)h2fulleff->ProjectionX("trueptd", 1, -1);
  TH1D* htruept = (TH1D*)h2fulleff->ProjectionY("truept", 1, -1);

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

  std::stringstream ss;
  TFile* fout =
   new TFile(Form("Unfold%s%sSplit_ebye_sd2_R25_%s.root", type.c_str(), cut.c_str(), date.c_str()), "RECREATE");

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
  // h2true->SetName("true");
  h2true->Write();
  h2truefrac->Write();
  h2fulleff->Write();
  h1truefull->Write();
  h1true->Write();
  h1raw->Write();
  response.Write();
  response1D.Write();
  for (int jar = 1; jar < 10; jar++) {
    Int_t iter = jar;
    cout << "iteration" << iter << endl;
    cout << "==============Unfold h1=====================" << endl;

    RooUnfoldBayes unfold(&response, h2raw, iter, true); // OR
    TH2D* hunf = (TH2D*)unfold.Hreco(errorTreatment);
    // FOLD BACK
    TH1* hfold = response.ApplyToTruth(hunf, "");

    RooUnfoldBayes unfold1D(&response1D, h1raw, iter, false);
    TH1D* hunf1D = (TH1D*)unfold1D.Hreco(errorTreatment);
    TH1* hfold1D = response1D.ApplyToTruth(hunf1D, "");

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
    TH1D* htempUnf1D = (TH1D*)hunf1D->Clone("htempUnf1D");
    htempUnf1D->SetName(Form("Bayesian_Unfolded1Diter%d", iter));

    TH2D* htempFold = (TH2D*)hfold->Clone("htempFold");
    htempFold->SetName(Form("Bayesian_Foldediter%d", iter));
    TH1D* htempFold1D = (TH1D*)hfold1D->Clone("htempFold1D");
    htempFold1D->SetName(Form("Bayesian_Folded1Diter%d", iter));

    htempUnf->Write();
    htempFold->Write();
    htempUnf1D->Write();
    htempFold1D->Write();

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
  fout->Close();
}

#ifndef __CINT__
int main()
{
  RooSimplenSDPbPb_split();
  return 0;
} // Main program when run stand-alone
#endif
