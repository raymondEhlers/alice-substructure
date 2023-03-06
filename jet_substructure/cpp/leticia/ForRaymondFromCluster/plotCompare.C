#include <Riostream.h>
#include "TFile.h"
#include "TTree.h"
#include <TClonesArray.h>
#include <TChain.h>
#include <TRefArray.h>
#include <TFile.h>
#include <TKey.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include "TGraph.h"
#include <TProfile.h>
#include <TList.h>
#include <TLorentzVector.h>
#include <math.h>
#include "TF1.h"
#include "TRandom3.h"
// flow components
 Double_t background(Double_t *x, Double_t *par) {
 return par[0] +2.*par[1]*TMath::Cos(x[0])+2.*par[2]*TMath::Cos(2.*x[0]);
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

TGraphAsymmErrors *DivideTGraphWithTGraph(TGraphAsymmErrors *gr1, TGraphAsymmErrors *gr2)
{
  Int_t nPoints = gr1->GetN();
  Double_t x1,y1,x2,y2;
 
  TGraphAsymmErrors *gnew=(TGraphAsymmErrors*)gr1->Clone("gnew");
 
      for(Int_t i=0; i<nPoints; i++)
    {
      gr1->GetPoint(i,x1,y1);
      gr2->GetPoint(i,x2,y2);
      Double_t e1 = gr1->GetErrorYhigh(i);
      Double_t e2 = gr2->GetErrorYhigh(i);
      Double_t e1b = gr1->GetErrorYlow(i);
      Double_t e2b = gr2->GetErrorYlow(i);
      Double_t value = y1/y2;
      Double_t error = TMath::Sqrt((e1*e1/y1/y1 + e2*e2/y2/y2)*value*value);
      Double_t errorb = TMath::Sqrt((e1b*e1b/y1/y1 + e2b*e2b/y2/y2)*value*value);
      gnew->SetPoint(i,x1,value);
     
      gnew->SetPointError(i,gr1->GetErrorXlow(i),gr1->GetErrorXhigh(i),errorb,error);
    }
      return gnew;

}



TGraphAsymmErrors *RatioDataPythia(TH1D *gr,TH1D *true1, TGraphAsymmErrors *shapeuncorr){

         Int_t bins=gr->GetNbinsX();
	 TGraphAsymmErrors *graphRatio = new TGraphAsymmErrors(bins);
         for (Int_t iBin=1; iBin<=gr->GetNbinsX(); iBin++){
        
        Double_t yErrCorrHigh=0.,yErrCorrLow=0.,xErrCorrHigh=0.,xErrCorrLow=0.,yErrCorrHighRel=0., yErrCorrLowRel=0., yErrUncorrHigh=0., yErrUncorrLow=0., yErrUncorrHighRel=0., yErrUncorrLowRel=0.;
        Double_t xPoint=0., yPoint=0., xCorr=0., yCorr=0.,xUncorr=0., yUncorr=0., yratio=0., yratioPer11=0;
        
      
        yPoint = true1->GetBinContent(iBin);
        xPoint = true1->GetBinCenter(iBin);
        
        //shapeuncorr->GetPoint(iBin-1, xUncorr, yUncorr);
	
        yErrUncorrHigh = shapeuncorr->GetErrorYhigh(iBin-1);
        yErrUncorrLow = shapeuncorr->GetErrorYlow(iBin-1);
        yErrUncorrHighRel=yErrUncorrHigh/yPoint;
        yErrUncorrLowRel=yErrUncorrLow/yPoint;


	xErrCorrHigh = gr->GetBinWidth(iBin); 
        yratio = gr->GetBinContent(iBin);
        graphRatio->SetPoint(iBin-1, xPoint, yratio);
	cout<<yErrUncorrHighRel<<" "<<iBin<<endl;
        graphRatio->SetPointError(iBin-1, xErrCorrHigh*0.5, xErrCorrHigh*0.5, yErrUncorrLowRel,yErrUncorrHighRel);}
        return graphRatio;
    
      
      }


void plotCompare(){

    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    Int_t bincorr=0;
     Int_t  bin1=3;
     Int_t bin2=3;

     Int_t mine=2;
   TFile *test[20];
   TH1D *eff;
   TCanvas *canv2;
    TCanvas *canv3;
     TCanvas *canv4;
      TCanvas *canv5;
       TCanvas *canv6;
      TCanvas *ca;
      TPad *pad;
      TPad *pad2;
      TGraphAsymmErrors *err1, *err2, *err3, *err4, *err5;
      TH1D *histo1, *histo2, *histo3, *histo4, *histo5;
      TH1D *histo1_ratio, *histo2_ratio, *histo3_ratio, *histo4_ratio, *histo5_ratio;
    TString fname[5];
    TFile *input[5];
    fname[0]="result_leadingktzcut04.root";
    fname[1]="result_ledingktzcut02.root";
    fname[2]="result_dynamickt.root";
    fname[3]="result_dynamictf.root";
      fname[4]="result_leadingktnocut.root";
      
input[0]=TFile::Open(fname[0]);
 input[1]=TFile::Open(fname[1]);
  input[2]=TFile::Open(fname[2]);
   input[3]=TFile::Open(fname[3]);
  input[4]=TFile::Open(fname[4]);

  histo1=(TH1D*)input[0]->Get("shapeR");
     histo2=(TH1D*)input[1]->Get("shapeR");
      histo3=(TH1D*)input[2]->Get("shapeR");
        histo4=(TH1D*)input[3]->Get("shapeR");
	 histo5=(TH1D*)input[4]->Get("shapeR");

       err1=(TGraphAsymmErrors*)input[0]->Get("Graph");
       err2=(TGraphAsymmErrors*)input[1]->Get("Graph");
       err3=(TGraphAsymmErrors*)input[2]->Get("Graph");
        err4=(TGraphAsymmErrors*)input[3]->Get("Graph");
          err5=(TGraphAsymmErrors*)input[4]->Get("Graph");






	  canv4= new TCanvas(Form("canvas4"),Form("canvas4") ,700,600);
    

   auto *p1 = new TPad("p1","p1",0.,0.,1.,1.);  
    p1->SetMargin(0.15,0.05,0.4,0.1);
   //p1->Draw();

   auto *p2 = new TPad("p2","p3",0.,0.,1.,1);
      p2->SetMargin(0.15,0.05,0.1,0.6);
       p2->SetFillStyle(4000);
    
     TLegend *legi = new TLegend(0.2, 0.4, 0.3, 0.6);
   legi->SetBorderSize(0);
  
  
   legi->SetFillColor(0);
  legi->SetBorderSize(0);
  legi->SetTextFont(42);
  legi->SetTextSize(0.035);
  // leg->SetNDC(true);
  TLatex* lat = new TLatex();
  lat->SetTextFont(42);
  lat->SetNDC(true);
  lat->SetTextSize(0.04);
  lat->SetTextAlign(33);


   
   p1->cd();
   
   gPad->SetTickx();
   gPad->SetTicky(); 
   gPad->SetLogy();
  histo1->GetXaxis()->SetTitleSize(0.05);
  histo1->GetYaxis()->SetTitleOffset(1.3);
  histo1->GetXaxis()->SetLabelSize(0.05);					   
  histo1->GetXaxis()->SetLabelOffset(1.0);					   
  histo1->GetYaxis()->SetLabelSize(0.05);					   
  histo1->GetXaxis()->SetTitle("#it{k}_{T}");
  histo1->GetYaxis()->SetTitleSize(0.05);
  histo1->GetYaxis()->SetTitle("(1/N_{jet}) dN_{jet}/d#it{k}_{T}");
  
  histo1->SetLineColor(kBlack);
  histo1->SetMarkerColor(kBlack);
  histo1->SetMarkerStyle(20);
  histo1->SetMarkerSize(1);
  histo1->SetTitle("");
    histo1->GetYaxis()->SetRangeUser(5.e-4,2);
  histo1->GetXaxis()->SetRangeUser(0.25,8);
  
  histo1->Draw("pe");

  histo2->SetLineColor(kRed);
  histo2->SetMarkerColor(kRed);
  histo2->SetMarkerStyle(20);
  histo2->SetMarkerSize(1);
  histo2->Draw("pe same");


   histo3->SetLineColor(kGreen);
  histo3->SetMarkerColor(kGreen);
  histo3->SetMarkerStyle(20);
  histo3->SetMarkerSize(1);
  
  histo3->Draw("pe same");

 histo4->SetLineColor(kBlue);
  histo4->SetMarkerColor(kBlue);
  histo4->SetMarkerStyle(20);
  histo4->SetMarkerSize(1);
  histo4->Draw("pe same");


  histo5->SetLineColor(kOrange);
  histo5->SetMarkerColor(kOrange);
  histo5->SetMarkerStyle(20);
  histo5->SetMarkerSize(1);
  histo5->Draw("pe same");
 
  histo4_ratio=(TH1D*)histo4->Clone("histo4_ratio");
   histo3_ratio=(TH1D*)histo3->Clone("histo3_ratio");
    histo1_ratio=(TH1D*)histo1->Clone("histo1_ratio");
histo2_ratio=(TH1D*)histo2->Clone("histo5_ratio");

   histo4_ratio->Divide(histo5);
    histo3_ratio->Divide(histo5);
    histo1_ratio->Divide(histo5);
    histo2_ratio->Divide(histo5);
    
  
  err1->SetFillColorAlpha(kBlack, 0.4);
  err1->SetLineColor(kBlack);
  err1->SetMarkerColor(kBlack);
  err1->SetMarkerStyle(20);
  err1->SetMarkerSize(1);
  err1->SetLineWidth(1);
  err1->Draw("2 same");

   err2->SetFillColorAlpha(kRed, 0.4);
  err2->SetLineColor(kRed);
  err2->SetMarkerColor(kRed);
  err2->SetMarkerStyle(20);
  err2->SetMarkerSize(1);
  err2->SetLineWidth(1);
  err2->Draw("2 same");

   err3->SetFillColorAlpha(kGreen, 0.4);
  err3->SetLineColor(kGreen);
  err3->SetMarkerColor(kGreen);
  err3->SetMarkerStyle(20);
  err3->SetMarkerSize(1);
  err3->SetLineWidth(1);
  err3->Draw("2 same");


   err4->SetFillColorAlpha(kBlue, 0.4);
  err4->SetLineColor(kBlue);
  err4->SetMarkerColor(kBlue);
  err4->SetMarkerStyle(20);
  err4->SetMarkerSize(1);
  err4->SetLineWidth(1);
  err4->Draw("2 same");


   err5->SetFillColorAlpha(kOrange, 0.4);
  err5->SetLineColor(kOrange);
  err5->SetMarkerColor(kOrange);
  err5->SetMarkerStyle(20);
  err5->SetMarkerSize(1);
  err5->SetLineWidth(1);
  err5->Draw("2 same");

  
 TGraphAsymmErrors *g_ratio1 = DivideTGraphWithTGraph(err1,err5);
  TGraphAsymmErrors *g_ratio3 = DivideTGraphWithTGraph(err3,err5);
  TGraphAsymmErrors *g_ratio4 = DivideTGraphWithTGraph(err4,err5);
  TGraphAsymmErrors *g_ratio2 = DivideTGraphWithTGraph(err2,err5);
  histo1->Draw("same");
  
  legi->AddEntry(err1, "leading k_{T} && zcut=0.4");
 legi->AddEntry(err2, "leading k_{T} && zcut=0.2");
  legi->AddEntry(err3, "Dynamical k_{T}");
   legi->AddEntry(err4, "Dynamical t_{f}");
    legi->AddEntry(err5, "leading k_{T}"); 

   
  legi->Draw("same");
  
  lat->DrawLatex(0.93, 0.88, "ALICE Preliminary");
  lat->DrawLatex(0.93, 0.83, "2018 pp #sqrt{s} = 5.02 TeV");
  lat->DrawLatex(0.93, 0.78, "anti-k_{T} charged jets R = 0.4");
  
  lat->DrawLatex(0.93, 0.73, "60 < #it{p}_{T,jet}^{ch} < 80 GeV/c");
  
  lat->SetTextAlign(31);

   p2->cd();
  gPad->SetTickx();
  gPad->SetTicky();
  
  histo1_ratio->GetXaxis()->SetLabelSize(0.05);
  histo1_ratio->GetYaxis()->SetLabelSize(0.03);
  histo1_ratio->GetXaxis()->SetTitleSize(0.05);
  histo1_ratio->GetYaxis()->SetTitleSize(0.05);
  histo1_ratio->GetYaxis()->SetTitleOffset(1.3);					     
  histo1_ratio->GetYaxis()->SetTitle("Ratio");
  
  histo1_ratio->GetXaxis()->SetTitle("#it{k}_{T}");
  
  histo1_ratio->GetXaxis()->SetRangeUser(0.25, 8);

  histo1_ratio->SetTitle("");
  histo1_ratio->GetYaxis()->SetRangeUser(0., 2.6);
  histo1_ratio->GetXaxis()->SetLabelOffset(0.005);
  histo1_ratio->GetXaxis()->SetNdivisions(8);
  histo1_ratio->GetYaxis()->SetNdivisions(606);
  histo1_ratio->SetMarkerStyle(20);
  histo1_ratio->SetMarkerSize(1);
  histo1_ratio->Draw("");
  histo3_ratio->Draw("same"); 
  histo4_ratio->Draw("same"); 
   histo2_ratio->Draw("same"); 
  g_ratio1->SetFillColorAlpha(kBlack, 0.4);
  g_ratio1->SetLineColor(kBlack);
  g_ratio1->SetMarkerColor(kBlack);
  g_ratio1->SetMarkerStyle(20);
  g_ratio1->SetMarkerSize(1);
  g_ratio1->SetLineWidth(1);
  g_ratio1->Draw("2 same");


   g_ratio3->SetFillColorAlpha(kGreen, 0.4);
  g_ratio3->SetLineColor(kGreen);
  g_ratio3->SetMarkerColor(kGreen);
  g_ratio3->SetMarkerStyle(20);
  g_ratio3->SetMarkerSize(1);
  g_ratio3->SetLineWidth(1);
  g_ratio3->Draw("2 same");


   g_ratio4->SetFillColorAlpha(kBlue, 0.4);
  g_ratio4->SetLineColor(kBlue);
  g_ratio4->SetMarkerColor(kBlue);
  g_ratio4->SetMarkerStyle(20);
  g_ratio4->SetMarkerSize(1);
  g_ratio4->SetLineWidth(1);
  g_ratio4->Draw("2 same");


     g_ratio2->SetFillColorAlpha(kRed, 0.4);
  g_ratio2->SetLineColor(kRed);
  g_ratio2->SetMarkerColor(kRed);
  g_ratio2->SetMarkerStyle(20);
  g_ratio2->SetMarkerSize(1);
  g_ratio2->SetLineWidth(1);
  g_ratio2->Draw("2 same");

    histo1_ratio->Draw("same");
  histo3_ratio->Draw("same"); 
  histo4_ratio->Draw("same"); 
   histo2_ratio->Draw("same"); 
  TLine* line = new TLine(0, 1, 8, 1);
  line->SetLineStyle(2);
  line->Draw("same");

  canv4->cd();
  p1->Draw();
  p2->Draw();
 
 

  canv4->cd();
  p1->Draw();
   p2->Draw();


  canv4->SaveAs("GroomingComparison.pdf");
  








}    
