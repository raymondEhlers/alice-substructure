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


TGraphAsymmErrors *GraphRebin(TH1D *gr,TGraphAsymmErrors *shapeuncorr){

         Int_t bins=gr->GetNbinsX();
	 TGraphAsymmErrors *graphRatio = new TGraphAsymmErrors(bins-1);

	 for (Int_t iBin=2; iBin<=gr->GetNbinsX(); iBin++){
        
        Double_t yErrCorrHigh=0.,yErrCorrLow=0.,xErrCorrHigh=0.,xErrCorrLow=0.,yErrCorrHighRel=0., yErrCorrLowRel=0., yErrUncorrHigh=0., yErrUncorrLow=0., yErrUncorrHighRel=0., yErrUncorrLowRel=0.;
        Double_t xPoint=0., yPoint=0., xCorr=0., yCorr=0.,xUncorr=0., yUncorr=0., yratio=0., yratioPer11=0;
        
      

	
        yErrUncorrHigh = shapeuncorr->GetErrorYhigh(iBin-1);
        yErrUncorrLow = shapeuncorr->GetErrorYlow(iBin-1);
      	xErrCorrHigh = gr->GetBinWidth(iBin); 
        
	
        shapeuncorr->GetPoint(iBin-1, xPoint,yPoint);
	
	graphRatio->SetPoint(iBin-2, xPoint, yPoint);
        graphRatio->SetPointError(iBin-2, xErrCorrHigh*0.5, xErrCorrHigh*0.5, yErrUncorrLow,yErrUncorrHigh);}
        return graphRatio;
    
      
      }







void plotfinalUncertPanels(){

    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    Int_t bincorr=0;
     Int_t  bin1=3;
     Int_t bin2=3;

     Int_t mine=1;
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
    
     TGraphAsymmErrors *shapeuncorr[20];
     TGraphAsymmErrors *shapeuncorr1[20];
     TGraphAsymmErrors *shapeuncorr2[20];
     TGraphAsymmErrors *shapeuncorr3[20];
     TGraphAsymmErrors *shapeuncorr4[20];
     TGraphAsymmErrors *shapeuncorr5[20];
     TGraphAsymmErrors *shapeuncorr6[20];
     TGraphAsymmErrors *shapeuncorr7[20];
     TGraphAsymmErrors *shapeuncorr8[20];

      TGraphAsymmErrors *shapeuncorr9[20];
      TGraphAsymmErrors *shapeuncorr10[20];
      TGraphAsymmErrors *shapeuncorr11[20];
      TGraphAsymmErrors *shapeuncorr12[20];
      TGraphAsymmErrors *shapeuncorr13[20];
      TGraphAsymmErrors *shapeuncorr14[20];
      TH1D *h1_ratio;
      TH2D *itera;
    TH1D *shape[20];
     TH1D *shapeun[20];
   TH2D* htrue;
   TH1D* htrue1;
TH1D* htrue1un;

 if(mine==0){
    //output of the unfolding
       test[0] = new TFile("UnfoldKgzg0.4ppDefaultRmax025.root");
       //binning
       test[1] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Bin.root");
       //for iterations//
       test[2] = new TFile("UnfoldKgzg0.4ppDefaultRmax025.root");
       test[3] = new TFile("UnfoldKgzg0.4ppDefaultRmax025.root");
       //prior
       test[4] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Prior.root"); 
       //truncation
       test[5] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncHigh.root"); 
       test[6] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncLow.root");
       test[7] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncLow.root");
       
       //effi, rmax, displacement
       test[8] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Eff.root"); 
      
       test[9] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Displaced.root");
       test[10] = new TFile("UnfoldKgzg0.4ppDefaultRmax025.root");

       test[11] = new TFile("UnfoldKgzg0.4ppDefaultRmax025.root");}



    if(mine==1){
    //output of the unfolding
       test[0] = new TFile("UnfoldKgzg0.2ppDefaultRmax025.root");
       //binning
       test[1] = new TFile("UnfoldKgzg0.2ppDefaultRmax025Bin.root");
       //for iterations//
       test[2] = new TFile("UnfoldKgzg0.2ppDefaultRmax025.root");
       test[3] = new TFile("UnfoldKgzg0.2ppDefaultRmax025.root");
       //prior
       test[4] = new TFile("UnfoldKgzg0.2ppDefaultRmax025Prior.root"); 
       //truncation
       test[5] = new TFile("UnfoldKgzg0.2ppDefaultRmax025TruncHigh.root"); 
       test[6] = new TFile("UnfoldKgzg0.2ppDefaultRmax025TruncLow.root");
       test[7] = new TFile("UnfoldKgzg0.2ppDefaultRmax025TruncLow.root");
       
       //effi, rmax, displacement
       test[8] = new TFile("UnfoldKgzg0.2ppDefaultRmax025Eff.root"); 
      
       test[9] = new TFile("UnfoldKgzg0.2ppDefaultRmax025Displaced.root");
       test[10] = new TFile("UnfoldKgzg0.2ppDefaultRmax025.root");

       test[11] = new TFile("UnfoldKgzg0.2ppDefaultRmax025.root");}


    if(mine==2){
    //output of the unfolding
       test[0] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Dyn.root");
       //binning
       test[1] = new TFile("UnfoldKgzg0.4ppDefaultRmax025BinDyn.root");
       //for iterations//
       test[2] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Dyn.root");
       test[3] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Dyn.root");
       //prior
       test[4] = new TFile("UnfoldKgzg0.4ppDefaultRmax025PriorDyn.root"); 
       //truncation
       test[5] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncHighDyn.root"); 
       test[6] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncLowDyn.root");
       test[7] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncLowDyn.root");
       
       //effi, rmax, displacement
       test[8] = new TFile("UnfoldKgzg0.4ppDefaultRmax025EffDyn.root"); 
      
       test[9] = new TFile("UnfoldKgzg0.4ppDefaultRmax025DisplacedDyn.root");
       test[10] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Dyn.root");

       test[11] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Dyn.root");}



    if(mine==3){
    //output of the unfolding
       test[0] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Time.root");
       //binning
       test[1] = new TFile("UnfoldKgzg0.4ppDefaultRmax025BinTime.root");
       //for iterations//
       test[2] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Time.root");
       test[3] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Time.root");
       //prior
       test[4] = new TFile("UnfoldKgzg0.4ppDefaultRmax025PriorTime.root"); 
       //truncation
       test[5] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncHighTime.root"); 
       test[6] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncLowTime.root");
       test[7] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncLowTime.root");
       
       //effi, rmax, displacement
       test[8] = new TFile("UnfoldKgzg0.4ppDefaultRmax025EffTime.root"); 
      
       test[9] = new TFile("UnfoldKgzg0.4ppDefaultRmax025DisplacedTime.root");
       test[10] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Time.root");

       test[11] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Time.root");}


     if(mine==4){
    //output of the unfolding
       test[0] = new TFile("UnfoldKgzg0.4ppDefaultRmax025nocut.root");
       //binning
       test[1] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Binnocut.root");
       //for iterations//
       test[2] = new TFile("UnfoldKgzg0.4ppDefaultRmax025nocut.root");
       test[3] = new TFile("UnfoldKgzg0.4ppDefaultRmax025nocut.root");
       //prior
       test[4] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Priornocut.root"); 
       //truncation
       test[5] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncHighnocut.root"); 
       test[6] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncLownocut.root");
       test[7] = new TFile("UnfoldKgzg0.4ppDefaultRmax025TruncLownocut.root");
       
       //effi, rmax, displacement
       test[8] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Effnocut.root"); 
      
       test[9] = new TFile("UnfoldKgzg0.4ppDefaultRmax025Displacednocut.root");
       test[10] = new TFile("UnfoldKgzg0.4ppDefaultRmax025nocut.root");

       test[11] = new TFile("UnfoldKgzg0.4ppDefaultRmax025nocut.root");}


    
	
         htrue=(TH2D*)test[0]->Get("truef");
	 htrue1=(TH1D*)htrue->ProjectionX("true1",4,4);
	 htrue1un=(TH1D*)htrue1->Clone("htrue1un");
	 htrue1->Scale(1./htrue1->Integral(1,-1));
	 htrue1->Scale(1,"width");
	  for(Int_t k=0;k<12;k++){
	    eff=(TH1D*)test[k]->Get("correff60-80");
	    if(!eff) cout<<k<<endl;
	    itera=(TH2D*)test[k]->Get(Form("Bayesian_Unfoldediter%d.root",5));
	  if(k==2)itera=(TH2D*)test[k]->Get(Form("Bayesian_Unfoldediter%d.root",4));
	  if(k==3)itera=(TH2D*)test[k]->Get(Form("Bayesian_Unfoldediter%d.root",7));
	 
          shape[k]=(TH1D*)itera->ProjectionX(Form("shape_%i",k),4,4);
          shape[k]->Divide(eff); 
      	 
	   shapeun[k]=(TH1D*)shape[k]->Clone(Form("shapeun[%d]",k));
	    shape[k]->Scale(1./shape[k]->Integral(1,-1));
           shape[k]->Scale(1,"width");
	   
	}
	 
  //////////////////////////////////////////////////////////////////////////

     Double_t vectorx[10];
     Double_t relative[10];
     Double_t vectory[10];
     Double_t vectory2[10];
     Double_t errxa[10];
     Double_t errxb[10];
     Double_t errya[10];
     Double_t erryb[10];
     Double_t erryatrunc[10];
     Double_t errybtrunc[10];
     Double_t erryaiter[10];
     Double_t errybiter[10];
     Double_t erryaprior[10];
     Double_t errybprior[10];

     Double_t erryarmax[10];
     Double_t errybrmax[10];

     Double_t erryaeffi[10];
     Double_t errybeffi[10];
     Double_t erryaresp[10];
     Double_t errybresp[10];
     Double_t erryaresp2[10];
     Double_t errybresp2[10];
     Double_t erryaclose[10];
     Double_t errybclose[10];
     
     Double_t erruncorrup[10];
     Double_t erruncorrdo[10];
     Double_t erryabin[10];
     Double_t errybbin[10];

      Double_t erryabin2[10];
     Double_t errybbin2[10];
      Double_t erryabin3[10];
     Double_t errybbin3[10];
      Double_t erryabin4[10];
     Double_t errybbin4[10];
      Double_t erryabin5[10];
     Double_t errybbin5[10];
     Double_t erryadisplace[10];
     Double_t errybdisplace[10];

     Double_t erryadisplace2[10];
     Double_t errybdisplace2[10];
     Double_t erruncorrup1[10];
     Double_t erruncorrdo1[10];
     Double_t erruncorrup2[10];
     Double_t erruncorrdo2[10];
     Double_t erruncorrup3[10];
     Double_t erruncorrdo3[10];
     Double_t erruncorrup4[10];
     Double_t erruncorrdo4[10];
     Double_t erruncorrup5[10];
     Double_t erruncorrdo5[10];
     Double_t erruncorrup6[10];
     Double_t erruncorrdo6[10];
    Double_t erruncorrup7[10];
     Double_t erruncorrdo7[10];
      Double_t erruncorrup8[10];
     Double_t erruncorrdo8[10];

        Double_t erruncorrup9[10];
     Double_t erruncorrdo9[10];

        Double_t erruncorrup10[10];
     Double_t erruncorrdo10[10];

     Double_t erruncorrup11[10];
     Double_t erruncorrdo11[10];
     Double_t erruncorrup12[10];
     Double_t erruncorrdo12[10];

     Double_t erruncorrup13[10];
     Double_t erruncorrdo13[10];

     Double_t erruncorrup14[10];
     Double_t erruncorrdo14[10];
     
     Double_t vectox[10];
     Double_t vectoy[10];
     Double_t exa[10];
     Double_t exb[10];
     Double_t eya[10];
     Double_t eyb[10];
     
     

      for(Int_t kk=0;kk<shape[0]->GetNbinsX();kk++){
      vectorx[kk]=0;
      vectory[kk]=0; 
      relative[kk]=0;
      errxa[kk]=0;
      errxb[kk]=0;
      errya[kk]=0;
      erryb[kk]=0;
      errybtrunc[kk]=0;
      erryatrunc[kk]=0;
      errybclose[kk]=0;
      erryaclose[kk]=0;
      erruncorrup[kk]=0;
      erruncorrdo[kk]=0;
      erruncorrup1[kk]=0;
      erruncorrdo1[kk]=0;
      erruncorrup2[kk]=0;
      erruncorrdo2[kk]=0;
       erruncorrup3[kk]=0;
      erruncorrdo3[kk]=0;

       erruncorrup4[kk]=0;
      erruncorrdo4[kk]=0;
      erruncorrup5[kk]=0;
      erruncorrdo5[kk]=0;
       erruncorrup6[kk]=0;
      erruncorrdo6[kk]=0;

       erruncorrup7[kk]=0;
      erruncorrdo7[kk]=0; 

      erruncorrup8[kk]=0;
      erruncorrdo8[kk]=0; 

      erruncorrup9[kk]=0;
      erruncorrdo9[kk]=0;
      erruncorrup10[kk]=0;
      erruncorrdo10[kk]=0;
      erruncorrup11[kk]=0;
      erruncorrdo11[kk]=0;
      erruncorrup12[kk]=0;
      erruncorrdo12[kk]=0;
      erruncorrup13[kk]=0;
      erruncorrdo13[kk]=0; 
      errybiter[kk]=0;
      erryaiter[kk]=0;

      errybprior[kk]=0;
      erryaprior[kk]=0;

      errybrmax[kk]=0;
      erryarmax[kk]=0;

      errybdisplace[kk]=0;
      erryadisplace[kk]=0;

      errybdisplace2[kk]=0;
      erryadisplace2[kk]=0;

 errybresp[kk]=0;
      erryaresp[kk]=0;

       errybresp2[kk]=0;
      erryaresp2[kk]=0;
       }
      
      Double_t errydisplace=0;
      Double_t errydisplace2=0;
    for(Int_t j=0;j<shape[0]->GetNbinsX();j++){
      
      Double_t erry=0;
      
      Double_t erryiter=shape[2]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
      Double_t erryitermenos=shape[3]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
      Double_t errybin=shape[1]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
      
      Double_t erryprior=shape[4]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
      Double_t errytrunc=shape[6]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
      Double_t errytruncmenos=shape[5]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
      Double_t errytrunctrue=shape[7]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
      errydisplace=shape[9]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
        errydisplace2=shape[9]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
      
      Double_t erryeffi=shape[8]->GetBinContent(j+1)-shape[0]->GetBinContent(j+1);
     
    

        
      
      if(erryprior>=0){ erryaprior[j]=erryprior;
        errybprior[j]=0;}
      if(erryprior<0){errybprior[j]=-1*erryprior;
        erryaprior[j]=0;}

    
  
     if(errybin>=0){ erryabin[j]=errybin;
        errybbin[j]=0;}
      if(errybin<0){errybbin[j]=-1*errybin;
        erryabin[j]=0;}

    
       

         if(erryiter>0 && erryitermenos>0){ erryaiter[j]=TMath::Max(erryiter,erryitermenos);
          errybiter[j]=0;}
        if(erryiter<0 && erryitermenos<0){ erryaiter[j]=0;
          errybiter[j]=-1*TMath::Min(erryiter,erryitermenos);}
        if(erryiter>0 && erryitermenos<0){ erryaiter[j]=erryiter;
          errybiter[j]=-1*erryitermenos;}
        if(erryiter<0 && erryitermenos>0){ erryaiter[j]=erryitermenos;
          errybiter[j]=-1*erryiter;}
     

          if(errytrunc>0 && errytruncmenos>0){ erryatrunc[j]=TMath::Max(errytrunc,errytruncmenos);
          errybtrunc[j]=0;}
        if(errytrunc<0 && errytruncmenos<0){ erryatrunc[j]=0;
          errybtrunc[j]=-1*TMath::Min(errytrunc,errytruncmenos);}
        if(errytrunc>0 && errytruncmenos<0){ erryatrunc[j]=errytrunc;
          errybtrunc[j]=-1*errytruncmenos;}
        if(errytrunc<0 && errytruncmenos>0){ erryatrunc[j]=errytruncmenos;
          errybtrunc[j]=-1*errytrunc;}


 	if(errydisplace>0 && errydisplace2>0){ erryadisplace[j]=TMath::Max(errydisplace,errydisplace2);
          errybdisplace[j]=0;}
        if(errydisplace<0 && errydisplace2<0){ erryadisplace[j]=0;
          errybdisplace[j]=-1*TMath::Min(errydisplace,errydisplace2);}
        if(errydisplace>0 && errydisplace2<0){ erryadisplace[j]=errydisplace;
          errybdisplace[j]=-1*errydisplace2;}
        if(errydisplace<0 && errydisplace2>0){ erryadisplace[j]=errydisplace2;
          errybdisplace[j]=-1*errydisplace;}


	


       if(erryeffi>=0){ erryaeffi[j]=erryeffi;
        errybeffi[j]=erryeffi;}
      if(erryeffi<0){errybeffi[j]=-1*erryeffi;
        erryaeffi[j]=-1*erryeffi;}


    

       


    
      
     
      erruncorrup[j]=TMath::Sqrt(erryaiter[j]*erryaiter[j]+erryatrunc[j]*erryatrunc[j]+erryaeffi[j]*erryaeffi[j]+erryadisplace[j]*erryadisplace[j]+erryabin[j]*erryabin[j]+erryaprior[j]*erryaprior[j]);
      erruncorrdo[j]=TMath::Sqrt(errybiter[j]*errybiter[j]+errybtrunc[j]*errybtrunc[j]+errybeffi[j]*errybeffi[j]+errybdisplace[j]*errybdisplace[j]+errybbin[j]*errybbin[j]+errybprior[j]*errybprior[j]);

       
  

      // vectory[j]=1;
      vectorx[j]=shape[0]->GetBinCenter(j+1);
      vectory[j]=shape[0]->GetBinContent(j+1);
      vectory2[j]=1;
      errxa[j]=0.5*shape[0]->GetBinWidth(j+1);
      errxb[j]=errxa[j];
      relative[j]=1;

    

       erruncorrup2[j]=TMath::Sqrt(erryatrunc[j]*erryatrunc[j])/vectory[j];
       erruncorrdo2[j]=TMath::Sqrt(errybtrunc[j]*errybtrunc[j])/vectory[j];
        
      erruncorrup3[j]=TMath::Sqrt(erryaiter[j]*erryaiter[j])/vectory[j];
      erruncorrdo3[j]=TMath::Sqrt(errybiter[j]*errybiter[j])/vectory[j];

      erruncorrup4[j]=TMath::Sqrt(erryabin[j]*erryabin[j])/vectory[j];
      erruncorrdo4[j]=TMath::Sqrt(errybbin[j]*errybbin[j])/vectory[j];

      erruncorrup5[j]=TMath::Sqrt(erryaeffi[j]*erryaeffi[j])/vectory[j];
      erruncorrdo5[j]=TMath::Sqrt(errybeffi[j]*errybeffi[j])/vectory[j];

      erruncorrup6[j]=TMath::Sqrt(erryadisplace[j]*erryadisplace[j])/vectory[j];
      erruncorrdo6[j]=TMath::Sqrt(errybdisplace[j]*errybdisplace[j])/vectory[j];

       erruncorrup8[j]=TMath::Sqrt(erryaprior[j]*erryaprior[j])/vectory[j];
      erruncorrdo8[j]=TMath::Sqrt(errybprior[j]*errybprior[j])/vectory[j];     


   

         
    }
       
     
    

 
       shapeuncorr[0]=new TGraphAsymmErrors(shape[0]->GetNbinsX(),vectorx,vectory,errxa,errxb,erruncorrdo,erruncorrup);
       shapeuncorr[0]->SetFillColorAlpha(kBlue-2,0.25);
       shapeuncorr[0]->SetLineWidth(2);
       shapeuncorr[0]->SetFillStyle(1001);
       shapeuncorr[0]->SetMarkerSize(0);

     

    

      
       shapeuncorr2[0]=new TGraphAsymmErrors(shape[0]->GetNbinsX(),vectorx,relative,errxa,errxb,erruncorrdo2,erruncorrup2);
       shapeuncorr2[0]->SetFillColorAlpha(kRed,0.25);
       shapeuncorr2[0]->SetLineColor(kRed);
       shapeuncorr2[0]->SetLineWidth(2);
       shapeuncorr2[0]->SetFillStyle(0);
       shapeuncorr2[0]->SetMarkerSize(0);


       shapeuncorr3[0]=new TGraphAsymmErrors(shape[0]->GetNbinsX(),vectorx,relative,errxa,errxb,erruncorrdo3,erruncorrup3);
       shapeuncorr3[0]->SetFillColorAlpha(kMagenta-2,0.25);
       shapeuncorr3[0]->SetLineColor(kMagenta-2);
       shapeuncorr3[0]->SetLineWidth(2);
       shapeuncorr3[0]->SetFillStyle(0);
       shapeuncorr3[0]->SetMarkerSize(0);



       shapeuncorr4[0]=new TGraphAsymmErrors(shape[0]->GetNbinsX(),vectorx,relative,errxa,errxb,erruncorrdo4,erruncorrup4);
       shapeuncorr4[0]->SetFillColorAlpha(kYellow,0.25);
       shapeuncorr4[0]->SetLineColor(kYellow-4);
       shapeuncorr4[0]->SetLineWidth(2);
       shapeuncorr4[0]->SetFillStyle(0);
       shapeuncorr4[0]->SetMarkerSize(0);

    
       shapeuncorr5[0]=new TGraphAsymmErrors(shape[0]->GetNbinsX(),vectorx,relative,errxa,errxb,erruncorrdo5,erruncorrup5);
       shapeuncorr5[0]->SetFillColorAlpha(kBlue+2,0.25);
       shapeuncorr5[0]->SetLineColor(kBlue+2);
       shapeuncorr5[0]->SetLineWidth(2);
       shapeuncorr5[0]->SetFillStyle(0);
       shapeuncorr5[0]->SetMarkerSize(0);



       shapeuncorr6[0]=new TGraphAsymmErrors(shape[0]->GetNbinsX(),vectorx,relative,errxa,errxb,erruncorrdo6,erruncorrup6);
       shapeuncorr6[0]->SetFillColorAlpha(kPink+1,0.25);
       shapeuncorr6[0]->SetLineColor(kPink+1);
       shapeuncorr6[0]->SetLineWidth(2);
       shapeuncorr6[0]->SetFillStyle(0);
       shapeuncorr6[0]->SetMarkerSize(0);
 
      
 
    
       shapeuncorr8[0]=new TGraphAsymmErrors(shape[0]->GetNbinsX(),vectorx,relative,errxa,errxb,erruncorrdo8,erruncorrup8);
       shapeuncorr8[0]->SetFillColorAlpha(kBlack+2,0.25);
       shapeuncorr8[0]->SetLineColor(kBlack+2+1);
       shapeuncorr8[0]->SetLineWidth(2);
       shapeuncorr8[0]->SetFillStyle(0);
       shapeuncorr8[0]->SetMarkerSize(0);
     
      

    


  



       
  canv2= new TCanvas(Form("canvas2"),Form("canvas2") ,1100,1100);
          canv2->SetTicks();
  	  canv2->cd();
  	  pad=new TPad("pad0","this is pad",0,0,1,1);
  	  pad->SetFillColor(0);
	 
  	  pad->SetMargin(0.15,0.12,0.25,0.9);
  	  pad->Draw();
  	  pad->SetTicks(1,1);
  	  pad->cd();
  	     gPad->SetLogy();       
    
       shape[0]->GetYaxis()->SetTitleOffset(0.9);
     shape[0]->GetXaxis()->SetTitleOffset(0.9);
  
   shape[0]->GetXaxis()->SetLabelFont(42);
   shape[0]->GetYaxis()->SetLabelFont(42);
   shape[0]->GetXaxis()->SetLabelSize(0.04);
   shape[0]->GetYaxis()->SetLabelSize(0.04); 
    
  shape[0]->GetXaxis()->SetTitleFont(42);
  shape[0]->GetYaxis()->SetTitleFont(42);
 
  shape[0]->GetXaxis()->SetTitleSize(0.065);
  shape[0]->GetYaxis()->SetTitleSize(0.065);
 
  	  shape[0]->GetXaxis()->SetTitle("k_{T,leading}^{zcut=0.4}");
  	  shape[0]->GetYaxis()->SetTitle("1/N_{jets} dN/k_{T,leading}^{zcut=0.4}");
  	  //shape[0]->GetYaxis()->SetRangeUser(1.e-3,50);
	  //shape[0]->GetXaxis()->SetRangeUser(-0.1,0.35);
      
      shape[0]->SetMarkerSize(1.3);
      shape[0]->SetMarkerStyle(21);
      shape[0]->SetMarkerColor(kBlack);
      shape[0]->SetLineWidth(2);
      shape[0]->Draw("P2");
     
      shapeuncorr[0]->Draw("P2SAME");
      htrue1->SetLineColor(2);
      htrue1->SetLineWidth(3);
      htrue1->Draw("same");

     TLegend *leg = new TLegend(0.6, 0.65, 0.75, 0.75);
   leg->SetBorderSize(0);
   leg->SetTextSize(0.03);
   leg->SetTextFont(42);
   leg->AddEntry(shape[0],"ALICE data unfolded", "PEL");
   leg->AddEntry(shapeuncorr[0],"sys uncertainty", "F");
    leg->AddEntry(htrue1,"pythia 8", "PEL");
   leg->Draw();
   leg->SetFillColor(0);  
     leg->Draw("same");
   DrawLatex(0.33, 0.45, 1,"pp #sqrt{#it{s_{NN}}} = 5.02 TeV ",0.03); 
   DrawLatex(0.33, 0.4, 1,Form("Anti-#it{k}_{T}  charged jets, #it{R} = 0.4, SD zcut = 0.4"),0.03); 
   // DrawLatex(0.22,0.88,1,"ALICE Preliminary",0.05);
   DrawLatex(0.33, 0.35, 1,"60 < p_{T}^{jet,ch} < 80 GeV/#it{c}",0.03); 
    
    
   canv2->SaveAs("rgunfodled.pdf");
    
      
      
 
   if(mine<2) cout<<"integral"<<shape[0]->Integral(1,8)<<" "<<1-shape[0]->Integral(1,1)/shape[0]->Integral(1,-1)<<" "<<shape[0]->GetBinContent(1)<<" "<<shape[0]->GetBinError(1)<<" "<<shape[0]->GetBinCenter(1)<<endl;




    canv3= new TCanvas(Form("canvas3"),Form("canvas3") ,1100,1100);
          canv3->SetTicks();
  	  canv3->cd();
  	  pad=new TPad("pad0","this is pad",0,0,1,1);
  	  pad->SetFillColor(0);
	 
  	  pad->SetMargin(0.15,0.12,0.25,0.9);
  	  pad->Draw();
  	  pad->SetTicks(1,1);
  	  pad->cd();
  	   
    
       shapeuncorr2[0]->GetYaxis()->SetTitleOffset(0.9);
     shapeuncorr2[0]->GetXaxis()->SetTitleOffset(0.9);
  
   shapeuncorr2[0]->GetXaxis()->SetLabelFont(42);
   shapeuncorr2[0]->GetYaxis()->SetLabelFont(42);
   shapeuncorr2[0]->GetXaxis()->SetLabelSize(0.04);
   shapeuncorr2[0]->GetYaxis()->SetLabelSize(0.04); 
    
  shapeuncorr2[0]->GetXaxis()->SetTitleFont(42);
  shapeuncorr2[0]->GetYaxis()->SetTitleFont(42);
 
  shapeuncorr2[0]->GetXaxis()->SetTitleSize(0.065);
  shapeuncorr2[0]->GetYaxis()->SetTitleSize(0.065);
 
  if(mine<2){shapeuncorr2[0]->GetXaxis()->SetTitle("k_{T}");
    shapeuncorr2[0]->GetYaxis()->SetTitle("1/N_{jets} dN/dk_{T}");}

   if(mine==2){shapeuncorr2[0]->GetXaxis()->SetTitle("k_{T}");
    shapeuncorr2[0]->GetYaxis()->SetTitle("1/N_{jets} dN/dk_{T}");}
  	 shapeuncorr2[0]->GetYaxis()->SetRangeUser(0.5,1.5);
  	 shapeuncorr2[0]->GetXaxis()->SetRangeUser(0.5,8);
      
      shapeuncorr2[0]->SetMarkerSize(1.3);
      shapeuncorr2[0]->SetMarkerStyle(21);
      shapeuncorr2[0]->SetMarkerColor(kBlack);
      shapeuncorr2[0]->SetLineWidth(2);
     
      shapeuncorr2[0]->Draw("AP2");
      shapeuncorr3[0]->Draw("P2SAME");
        shapeuncorr4[0]->Draw("P2SAME");
	 shapeuncorr5[0]->Draw("P2SAME");
	shapeuncorr6[0]->Draw("P2SAME");
	shapeuncorr8[0]->Draw("P2SAME");
    
  
     TLegend *lego = new TLegend(0.6, 0.7, 0.75, 0.87);
   lego->SetBorderSize(0);
   lego->SetTextSize(0.025);
   lego->SetTextFont(42);
  
   lego->AddEntry(shapeuncorr2[0],"Truncation", "F");
   lego->AddEntry(shapeuncorr3[0],"Iteration choice", "F");
   lego->AddEntry(shapeuncorr4[0],"Bin variation", "F");
   lego->AddEntry(shapeuncorr5[0],"Trackingefficiency", "F");
   lego->AddEntry(shapeuncorr6[0],"displacement", "F");
   lego->AddEntry(shapeuncorr8[0],"prior", "F");
 
   //  lego->AddEntry(shapeuncorr14[0],"pure matches and swaps response", "F");
   lego->Draw();
   lego->SetFillColor(0);  

    
   DrawLatex(0.23, 0.4, 1,"pp #sqrt{#it{s}} = 5.02 TeV",0.03); 
   DrawLatex(0.23, 0.35, 1,Form("Anti-#it{k}_{T}  charged jets, #it{R} = 0.4, SD zcut = 0.4"),0.03); 
   // DrawLatex(0.22,0.88,1,"ALICE Preliminary",0.05);
   DrawLatex(0.23, 0.3, 1,"60 < p_{T}^{jet,ch} < 80 GeV/#it{c}",0.03); 
   if(mine==0) canv3->SaveAs("uncert_zcut4.pdf");
   if(mine==1) canv3->SaveAs("uncert_zcut2.pdf");
     if(mine==2) canv3->SaveAs("uncert_dyn.pdf");
      if(mine==3) canv3->SaveAs("uncert_dyntime.pdf");
       if(mine==4) canv3->SaveAs("uncert_nocut.pdf");


  
   h1_ratio=(TH1D*)shape[0]->Clone("h1_ratio");
   h1_ratio->Divide(htrue1);
   TGraphAsymmErrors *g_ratio = RatioDataPythia(h1_ratio,htrue1,shapeuncorr[0]);
      

     canv4= new TCanvas(Form("canvas4"),Form("canvas4") ,700,600);
      auto *p2 = new TPad("p2","p3",0.,0.,1.,1);
      p2->SetMargin(0.15,0.05,0.1,0.6);
       p2->SetFillStyle(4000);
       // p2->Draw();
      
  
      //p2->SetGrid();

   auto *p1 = new TPad("p1","p1",0.,0.,1.,1.);  
   p1->SetMargin(0.15,0.05,0.4,0.1);
   //p1->Draw();
     TLegend *legi = new TLegend(0.2, 0.5, 0.3, 0.6);
   legi->SetBorderSize(0);
  
  
   leg->SetFillColor(0);
  legi->SetBorderSize(0);
  legi->SetTextFont(42);
  legi->SetTextSize(0.045);
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
   shape[0]->GetXaxis()->SetTitleSize(0.05);
  shape[0]->GetYaxis()->SetTitleOffset(1.3);
  shape[0]->GetXaxis()->SetLabelSize(0.05);					   
  shape[0]->GetXaxis()->SetLabelOffset(1.0);					   
  shape[0]->GetYaxis()->SetLabelSize(0.05);					   
  shape[0]->GetXaxis()->SetTitle("#it{k}_{T}");
  shape[0]->GetYaxis()->SetTitleSize(0.05);
  if(mine==0) shape[0]->GetYaxis()->SetTitle("(1/N_{jet}) dN_{jet}/d#it{k}_{T,leading}");
  if(mine==1)shape[0]->GetYaxis()->SetTitle("(1/N_{jet}) dN_{jet}/d#it{k}_{T,leading}");
  if(mine==2)shape[0]->GetYaxis()->SetTitle("(1/N_{jet}) dN_{jet}/d#it{k}_{T,dynamical-k_{T}}");
   if(mine==3)shape[0]->GetYaxis()->SetTitle("(1/N_{jet}) dN_{jet}/d#it{k}_{T,dynamical-t_{f}}");
    if(mine==4) shape[0]->GetYaxis()->SetTitle("(1/N_{jet}) dN_{jet}/d#it{k}_{T,leading}");
  shape[0]->SetLineColor(kBlack);
  shape[0]->SetMarkerColor(kBlack);
  shape[0]->SetMarkerStyle(20);
  shape[0]->SetMarkerSize(1);
  shape[0]->SetTitle("");
  shape[0]->GetXaxis()->SetRangeUser(0.25,8);
  shape[0]->GetXaxis()->SetNdivisions(8);
  //shape[0]->GetYaxis()->SetRangeUser(0, 10);
  shape[0]->Draw("pe");

  htrue1->SetLineColor(kRed-4);
  htrue1->SetMarkerColor(kRed-4);
  htrue1->SetMarkerStyle(21);
  htrue1->SetMarkerSize(1);
  htrue1->Draw("same");
  
  shapeuncorr[0]->SetFillColorAlpha(kBlack, 0.4);
  shapeuncorr[0]->SetLineColor(kBlack);
  shapeuncorr[0]->SetMarkerColor(kBlack);
  shapeuncorr[0]->SetMarkerStyle(20);
  shapeuncorr[0]->SetMarkerSize(1);
  shapeuncorr[0]->SetLineWidth(1);
  shapeuncorr[0]->Draw("2 same");
  
  shape[0]->Draw("same");
  htrue1->Draw("same");
  legi->AddEntry(shapeuncorr[0], "Unfolded Data");
  legi->AddEntry(htrue1, "Pythia8");
  legi->Draw("same");
  
  lat->DrawLatex(0.93, 0.88, "ALICE Preliminary");
  lat->DrawLatex(0.93, 0.83, "2018 pp #sqrt{s} = 5.02 TeV");
  lat->DrawLatex(0.93, 0.78, "anti-k_{T} charged jets R = 0.4");
  if(mine==0) lat->DrawLatex(0.93, 0.73, "Soft drop z_{cut} = 0.4 #beta = 0");
   if(mine==1) lat->DrawLatex(0.93, 0.73, "Soft drop z_{cut} = 0.2 #beta = 0");
    if(mine==2) lat->DrawLatex(0.93, 0.73, "Dynamical k_{T}");
     if(mine==3) lat->DrawLatex(0.93, 0.73, "Dynamical t_{f}");
     if(mine==0)  lat->DrawLatex(0.93, 0.63, "f_{tagged}=0.35");
      if(mine==1)  lat->DrawLatex(0.93, 0.63, "f_{tagged}=0.84");
  lat->DrawLatex(0.93, 0.68, "60 < #it{p}_{T} < 80 GeV/c");
  
  lat->SetTextAlign(31);

 
  p2->cd();
  gPad->SetTickx();
  gPad->SetTicky();
  
  h1_ratio->GetXaxis()->SetLabelSize(0.05);
  h1_ratio->GetYaxis()->SetLabelSize(0.05);
  h1_ratio->GetXaxis()->SetTitleSize(0.05);
  h1_ratio->GetYaxis()->SetTitleSize(0.05);
  h1_ratio->GetYaxis()->SetTitleOffset(1.3);					     
  h1_ratio->GetYaxis()->SetTitle("Data/MC");
  
  if(mine==0) h1_ratio->GetXaxis()->SetTitle("#it{k}_{T}");
   if(mine==1) h1_ratio->GetXaxis()->SetTitle("#it{k}_{T}");
    if(mine==2) h1_ratio->GetXaxis()->SetTitle("#it{k}_{T}");
     if(mine==3) h1_ratio->GetXaxis()->SetTitle("#it{k}_{T}");
     if(mine==4) h1_ratio->GetXaxis()->SetTitle("#it{k}_{T}");
     h1_ratio->GetXaxis()->SetRangeUser(0.25, 8);
  h1_ratio->SetTitle("");
  h1_ratio->GetYaxis()->SetRangeUser(0.6, 1.6);
  h1_ratio->GetXaxis()->SetLabelOffset(0.005);
  h1_ratio->GetXaxis()->SetNdivisions(8);
  h1_ratio->GetYaxis()->SetNdivisions(606);
  h1_ratio->SetMarkerStyle(20);
  h1_ratio->SetMarkerSize(1);
  h1_ratio->Draw();
  g_ratio->SetFillColorAlpha(kBlack, 0.4);
  g_ratio->SetLineColor(kBlack);
  g_ratio->SetMarkerColor(kBlack);
  g_ratio->SetMarkerStyle(20);
  g_ratio->SetMarkerSize(1);
  g_ratio->SetLineWidth(1);
  g_ratio->GetXaxis()->SetLimits(1, 13);
  g_ratio->Draw("2 same");
  TLine* line = new TLine(0, 1, 8, 1);
  line->SetLineStyle(2);
  line->Draw("same");

  canv4->cd();
  p1->Draw();
  p2->Draw();


  if(mine==0) canv4->SaveAs("ktunfolded_zcut4.pdf");
   if(mine==1) canv4->SaveAs("ktunfolded_zcut2.pdf");
     if(mine==2) canv4->SaveAs("ktunfolded_dyn.pdf");
      if(mine==3) canv4->SaveAs("ktunfolded_dyntime.pdf");
       if(mine==4) canv4->SaveAs("ktunfolded_nocut.pdf");

        TH1D *shapeR;
        Double_t binsNew[8]={0,0.5,1,2,4,6,8,15};
        shapeR=(TH1D*)shape[0]->Rebin(7,"shapeR",binsNew);
         
       TGraphAsymmErrors *shapeuncorrR = GraphRebin(shape[0],shapeuncorr[0]);
       // shapeuncorrR->Draw("");

            TString cOutFile; 
	   if(mine==0) cOutFile=Form("result_leadingktzcut04.root");
	   if(mine==1) cOutFile=Form("result_ledingktzcut02.root");
	   if(mine==2) cOutFile=Form("result_dynamickt.root");
	   if(mine==3) cOutFile=Form("result_dynamictf.root");
	   if(mine==4) cOutFile=Form("result_leadingktnocut.root");
	   TFile *fOut=0;
	   fOut=new TFile(cOutFile,"RECREATE");
	   fOut->cd();
	   //if(mine>=2){
	   //shape[0]->Write();
	   //shapeuncorr[0]->Write();}

	   // if(mine<2){
	      shapeR->Write();
	      shapeuncorrR->Write();
	      // }
	   fOut->Close();








}
