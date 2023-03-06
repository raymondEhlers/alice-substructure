#include <Riostream.h>
#include <TChain.h>
#include <TClonesArray.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TKey.h>
#include <TList.h>
#include <TLorentzVector.h>
#include <TProfile.h>
#include <TRefArray.h>
#include <math.h>

#include "TF1.h"
#include "TFile.h"
#include "TGraph.h"
#include "TRandom3.h"
#include "TTree.h"
// flow components
Double_t background(Double_t* x, Double_t* par)
{
  return par[0] + 2. * par[1] * TMath::Cos(x[0]) + 2. * par[2] * TMath::Cos(2. * x[0]);
}

void DrawLatex(Float_t x, Float_t y, Int_t color, const char* text, Float_t textSize = 0.06)
{
  TLatex* latex = new TLatex(x, y, text);
  latex->SetNDC();
  latex->SetTextSize(textSize);
  latex->SetTextColor(color);
  latex->SetTextFont(42);
  latex->Draw();
}

TGraphAsymmErrors* RatioDataPythia(TH1D* gr, TH1D* true1, TGraphAsymmErrors* shapeuncorr)
{
  Int_t bins = gr->GetNbinsX();
  TGraphAsymmErrors* graphRatio = new TGraphAsymmErrors(bins);
  for (Int_t iBin = 1; iBin <= gr->GetNbinsX(); iBin++) {
    Double_t yErrCorrHigh = 0., yErrCorrLow = 0., xErrCorrHigh = 0., xErrCorrLow = 0., yErrCorrHighRel = 0.,
         yErrCorrLowRel = 0., yErrUncorrHigh = 0., yErrUncorrLow = 0., yErrUncorrHighRel = 0.,
         yErrUncorrLowRel = 0.;
    Double_t xPoint = 0., yPoint = 0., xCorr = 0., yCorr = 0., xUncorr = 0., yUncorr = 0., yratio = 0.,
         yratioPer11 = 0;

    yPoint = true1->GetBinContent(iBin);
    xPoint = true1->GetBinCenter(iBin);

    // shapeuncorr->GetPoint(iBin-1, xUncorr, yUncorr);

    yErrUncorrHigh = shapeuncorr->GetErrorYhigh(iBin - 1);
    yErrUncorrLow = shapeuncorr->GetErrorYlow(iBin - 1);
    yErrUncorrHighRel = yErrUncorrHigh / yPoint;
    yErrUncorrLowRel = yErrUncorrLow / yPoint;

    xErrCorrHigh = gr->GetBinWidth(iBin);
    yratio = gr->GetBinContent(iBin);
    graphRatio->SetPoint(iBin - 1, xPoint, yratio);
    cout << yErrUncorrHighRel << " " << iBin << endl;
    graphRatio->SetPointError(iBin - 1, xErrCorrHigh * 0.5, xErrCorrHigh * 0.5, yErrUncorrLowRel,
                 yErrUncorrHighRel);
  }
  return graphRatio;
}

TGraphAsymmErrors* DivideTGraphWithTGraph(TGraphAsymmErrors* gr1, TGraphAsymmErrors* gr2)
{
  Int_t nPoints = gr1->GetN();
  Double_t x1, y1, x2, y2;

  TGraphAsymmErrors* gnew = (TGraphAsymmErrors*)gr1->Clone("gnew");

  for (Int_t i = 0; i < nPoints; i++) {
    gr1->GetPoint(i, x1, y1);
    gr2->GetPoint(i, x2, y2);
    Double_t e1 = gr1->GetErrorYhigh(i);
    Double_t e2 = gr2->GetErrorYhigh(i);
    Double_t e1b = gr1->GetErrorYlow(i);
    Double_t e2b = gr2->GetErrorYlow(i);
    Double_t value = y1 / y2;
    Double_t error = TMath::Sqrt((e1 * e1 / y1 / y1 + e2 * e2 / y2 / y2) * value * value);
    Double_t errorb = TMath::Sqrt((e1b * e1b / y1 / y1 + e2b * e2b / y2 / y2) * value * value);
    gnew->SetPoint(i, x1, value);

    gnew->SetPointError(i, gr1->GetErrorXlow(i), gr1->GetErrorXhigh(i), errorb, error);
  }
  return gnew;
}

void plotchooseiter60_var()
{
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  Int_t bincorr = 0;
  Int_t bin1 = 3;
  Int_t bin2 = 3;
  TH1D* shaperatiopp;
  TH1D* pptrue;

  TFile* test[20];
  TFile* ppres;
  TH1D* eff;
  TH1D* ppshape;
  TCanvas* canv2;
  TCanvas* canv3;
  TCanvas* canv4;
  TCanvas* canv5;
  TCanvas* ca;
  TPad* pad;
  TPad* pad2;

  TH1D* h1_ratio;
  TH1D* shape[20];
  TH1D* shapeun[20];
  TH2D* htrue;
  TH1D* htrue1;
  TH1D* htrue1un;
  TH1D *def1, *def2, *def3, *def4;
  TH2D *itera, *iterad, *iterau, *iterp;
  Double_t errprior, errreg1, errreg2, errreg, errstat, errtot;

  // output of the unfolding
  test[0] = new TFile("UnfoldRgzg0.4EventWiseDefaultRmax025_var.root");
  test[4] = new TFile("UnfoldRgzg0.4EventWiseDefaultRmax025Prior_var.root");
  TH1D* histotot(0);
  TH1D* historeg(0);
  TH1D* histoprior(0);
  TH1D* histostat(0);
  histotot = new TH1D("histotot", "histot", 12, 0, 15);
  historeg = new TH1D("historeg", "historeg", 12, 0, 15);
  histoprior = new TH1D("histoprior", "histoprior", 12, 0, 15);
  histostat = new TH1D("histostat", "histostat", 12, 0, 15);

  for (Int_t k = 2; k < 13; k++) {
    itera = (TH2D*)test[0]->Get(Form("Bayesian_Unfoldediter%d.root", k));
    iterad = (TH2D*)test[0]->Get(Form("Bayesian_Unfoldediter%d.root", k - 1));
    iterau = (TH2D*)test[0]->Get(Form("Bayesian_Unfoldediter%d.root", k + 2));
    iterp = (TH2D*)test[4]->Get(Form("Bayesian_Unfoldediter%d.root", k));

    def1 = (TH1D*)itera->ProjectionX(Form("def1_%i", k), 4, 4);
    def2 = (TH1D*)iterad->ProjectionX(Form("def2_%i", k), 4, 4);
    def3 = (TH1D*)iterau->ProjectionX(Form("def3_%i", k), 4, 4);
    def4 = (TH1D*)iterp->ProjectionX(Form("def4_%i", k), 4, 4);

    errprior = 0;
    errreg1 = 0;
    errreg2 = 0;
    errreg = 0;
    errstat = 0;
    errtot = 0;

    for (Int_t i = 1; i <= def1->GetNbinsX(); i++) {
      errprior = errprior + TMath::Abs(def4->GetBinContent(i) - def1->GetBinContent(i)) / def1->GetBinContent(i);
      errreg1 = TMath::Abs(def2->GetBinContent(i) - def1->GetBinContent(i));
      errreg2 = TMath::Abs(def3->GetBinContent(i) - def1->GetBinContent(i));
      errreg = errreg + TMath::Max(errreg1, errreg2) / def1->GetBinContent(i);
      errstat = errstat + def1->GetBinError(i) / def1->GetBinContent(i);
      errtot = TMath::Sqrt(errprior * errprior + errreg * errreg + errstat * errstat);
    }

    cout << k << " " << errtot << " " << errreg << " " << errprior << " " << errstat << endl;
    histotot->SetBinContent(k, errtot);
    historeg->SetBinContent(k, errreg);
    histoprior->SetBinContent(k, errprior);
    histostat->SetBinContent(k, errstat);
  }

  canv3 = new TCanvas(Form("canvas3"), Form("canvas3"), 1100, 1100);
  canv3->SetTicks();
  canv3->cd();
  pad = new TPad("pad0", "this is pad", 0, 0, 1, 1);
  pad->SetFillColor(0);

  pad->SetMargin(0.15, 0.12, 0.25, 0.9);
  pad->Draw();
  pad->SetTicks(1, 1);
  pad->cd();

  histotot->GetYaxis()->SetTitleOffset(0.9);
  histotot->GetXaxis()->SetTitleOffset(0.9);

  histotot->GetXaxis()->SetLabelFont(42);
  histotot->GetYaxis()->SetLabelFont(42);
  histotot->GetXaxis()->SetLabelSize(0.04);
  histotot->GetYaxis()->SetLabelSize(0.04);

  histotot->GetXaxis()->SetTitleFont(42);
  histotot->GetYaxis()->SetTitleFont(42);

  histotot->GetXaxis()->SetTitleSize(0.065);
  histotot->GetYaxis()->SetTitleSize(0.065);

  histotot->GetXaxis()->SetTitle("Iterations");
  histotot->GetYaxis()->SetTitle("Summed errors");

  histotot->SetMarkerSize(1.3);
  histotot->SetMarkerStyle(21);
  histotot->SetMarkerColor(kBlack);
  histotot->SetLineColor(1);
  histotot->Draw("");
  historeg->SetLineColor(4);
  historeg->Draw("same");
  histostat->SetLineColor(3);
  histostat->Draw("same");
  histoprior->SetLineColor(2);
  histoprior->Draw("same");

  TLegend* lego = new TLegend(0.6, 0.7, 0.75, 0.87);
  lego->SetBorderSize(0);
  lego->SetTextSize(0.025);
  lego->SetTextFont(42);
  lego->AddEntry(histotot, "total", "L");
  lego->AddEntry(historeg, "Regularization", "L");
  lego->AddEntry(histostat, "Statistical", "L");
  lego->AddEntry(histoprior, "Prior", "L");
  //  lego->AddEntry(shapeuncorr14[0],"pure matches and swaps response", "F");
  lego->Draw();
  lego->SetFillColor(0);

  DrawLatex(0.23, 0.4, 1, "PbPb #sqrt{#it{s_{NN}}} = 5.02 TeV, 30-50\% central", 0.03);
  DrawLatex(0.23, 0.35, 1, Form("Anti-#it{k}_{T}  charged jets, #it{R} = 0.4, SD zcut = 0.4"), 0.03);
  // DrawLatex(0.22,0.88,1,"ALICE Preliminary",0.05);
  DrawLatex(0.23, 0.3, 1, "60 < p_{T}^{jet,ch} < 80 GeV/#it{c}", 0.03);
  canv3->SaveAs("IterChoice_rg_zcut04_PbPb_var.pdf");
}
