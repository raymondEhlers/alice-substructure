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


// RooSimplepTPbPb_XML.cxx: Script to unfold the ML corrected data
// Hannah Bossi <hannah.bossi@yale.edu>
// 3/4/2020




#if !(defined(__CINT__) || defined(__CLING__)) || defined(__ACLIC__)
#include <iostream>
using std::cout;
using std::endl;

#include "TRandom.h"
#include "TH1D.h"

#include "TFile.h"
#include "TVectorD.h"

#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TRandom.h"
#include "TPostScript.h"
#include "TH2D.h"
#include "TFile.h"
#include "TLine.h"
#include "TNtuple.h"
#include "TProfile.h"
#inlcude "TRandom3.h"

#include "RooUnfoldResponse.h"
#include "RooUnfoldBayes.h"
//#include "RooUnfoldTestHarness2D.h"
#endif

//==============================================================================
// Global definitions
//==============================================================================

const Double_t cutdummy= -99999.0;

//==============================================================================
// Gaussian smearing, systematic translation, and variable inefficiency
//==============================================================================

TH2D* CorrelationHistShape (const TMatrixD& cov,const char* name, const char* title,
		       Int_t na, Int_t nb, Int_t kbin)
{
 
   TH2D* h= new TH2D (name, title, nb, 0, nb, nb, 0, nb);
 
     	  for(int l=0;l<nb;l++){
                 for(int n=0;n<nb;n++){
                int index1=kbin+na*l;
                int index2=kbin+na*n;
   	        Double_t Vv=cov(index1,index1)*cov(index2,index2);
   		 if (Vv>0.0) h->SetBinContent(l+1,n+1,cov(index1,index2)/sqrt(Vv));


  		  }}
  return h;
}

TH2D* CorrelationHistPt (const TMatrixD& cov,const char* name, const char* title,
		       Int_t na, Int_t nb, Int_t kbin)
{
 
   TH2D* h= new TH2D (name, title, na, 0, na, na, 0, na);
 
     	  for(int l=0;l<na;l++){
                 for(int n=0;n<na;n++){

                int index1=l+na*kbin;
                int index2=n+na*kbin;
   	        Double_t Vv=cov(index1,index1)*cov(index2,index2);
   		 if (Vv>0.0)h->SetBinContent(l+1,n+1,cov(index1,index2)/sqrt(Vv));


  		  }}
  return h;
}


   





void Normalize2D(TH2* h)
{
   Int_t nbinsYtmp = h->GetNbinsY();
   const Int_t nbinsY = nbinsYtmp;
   Double_t norm[nbinsY];
   for(Int_t biny=1; biny<=nbinsY; biny++)
     {
       norm[biny-1] = 0;
       for(Int_t binx=1; binx<=h->GetNbinsX(); binx++)
     {
       norm[biny-1] += h->GetBinContent(binx,biny);
     }
     }

   for(Int_t biny=1; biny<=nbinsY; biny++)
     {
       for(Int_t binx=1; binx<=h->GetNbinsX(); binx++)
     {
       if(norm[biny-1]==0)  continue;
       else
         {
  h->SetBinContent(binx,biny,h->GetBinContent(binx,biny)/norm[biny-1]);
  h->SetBinError(binx,biny,h->GetBinError(binx,biny)/norm[biny-1]);
         }
     }
     }
}






TH2D* CorrelationHist (const TMatrixD& cov,const char* name, const char* title,
		       Double_t lo, Double_t hi,Double_t lon,Double_t hin)
{
  Int_t nb= cov.GetNrows();
  Int_t na= cov.GetNcols();
  cout<<nb<<" "<<na<<endl;
  TH2D* h= new TH2D (name, title, nb, 0, nb, na, 0, na);
  h->SetAxisRange (-1.0, 1.0, "Z");
  for(int i=0; i < na; i++)
  for(int j=0; j < nb; j++) {
  Double_t Viijj= cov(i,i)*cov(j,j);
      if (Viijj>0.0) h->SetBinContent (i+1, j+1, cov(i,j)/sqrt(Viijj));
    }
  return h;
}

//==============================================================================
// Example Unfolding
//==============================================================================

void RooSimplepTPbPb_ML(TString cFiles2="filesML.txt")
{
#ifdef __CINT__
  gSystem->Load("libRooUnfold");
#endif
  Int_t difference=1;
  Int_t Ppol=0;
  cout << "==================================== pick up the response matrix for background==========================" << endl;
  ///////////////////parameter setting
  RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;   
  
  TRandom3* rand = new TRandom3();

   //***************************************************

  Double_t recbins[9];                                                                                                                                                       
  recbins[0]=25;                                                                                                                                                         
  recbins[1]=30;                                                                                                                                                              
  recbins[2]=40;                                                                                                                                                            
  recbins[3]=50;                                                                                                                                                       
  recbins[4]=60;                                                                                                                                                   
  recbins[5]=70;                                                                                                                                                      
  recbins[6]=85;                                                                                                                                                        
  recbins[7]=100;                                                                                                                                                    
  recbins[8]=120;                                                                                                                                                               


  Double_t xbins[13];
  xbins[0]=10;
  xbins[1]=20;
  xbins[2]=30;
  xbins[3]=40;
  xbins[4]=50;
  xbins[5]=60;
  xbins[6]=70;
  xbins[7]=80;
  xbins[8]=100;
  xbins[9]=120;
  xbins[10]=140;
  xbins[11]=190;
  xbins[12]=250;
  //xbins[13]=250;


  //std::vector<Double_t> kBinsMeasured = {25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120};
  std::vector<Double_t> kBinsUnfolded = {10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 190,250};
  std::vector<Double_t> kBinsMeasured = {25, 30, 40, 50, 60, 70, 80, 85, 100, 120};   

  //the raw correlation (data or psuedodata)
  TH1D *h1raw(0);
  h1raw=new TH1D("r","raw",  kBinsMeasured.size()-1, kBinsMeasured.data());

  //detector measure level (reco or hybrid MC)
  TH1D *h1smeared(0);
  h1smeared=new TH1D("smeared","smeared", kBinsMeasured.size()-1, kBinsMeasured.data());

  // full range of reco
  TH1D *h1smearedFullRange(0);
  h1smearedFullRange = new TH1D("smearedFull", "smearedFull", 220, -20,200);
    
  //detector measure level no cuts
  TH1D *h1smearednocuts(0);
  h1smearednocuts=new TH1D("smearednocuts","smearednocuts",  kBinsMeasured.size()-1, kBinsMeasured.data());
  //true correlations with measured cuts
  TH1D *h1true(0);
  h1true=new TH1D("true","true",  kBinsUnfolded.size()-1, kBinsUnfolded.data());
  //full true correlation
  TH1D *h1fulleff(0);
  h1fulleff=new TH1D("truef","truef",  kBinsUnfolded.size()-1, kBinsUnfolded.data()); 
  
  TH2D *hcovariance(0);
  hcovariance=new TH2D("covariance","covariance",10,0.,1.,10,0,1.);

  TH2D *effnum=(TH2D*)h1fulleff->Clone("effnum");
  TH2D *effdenom=(TH2D*)h1fulleff->Clone("effdenom");
 
   effnum->Sumw2();
   effdenom->Sumw2();
   h1smeared->Sumw2();
   h1smearedFullRange->Sumw2(); 
   h1true->Sumw2();
   h1raw->Sumw2();
   h1fulleff->Sumw2();

   //branches in the tree that you need in this analysis
   // we need the hybrid Pt to determine what EB scaling factor
   Float_t ptJetMatch, hybridPt, hybridPtData, ptPart, truePtFrac, dist, raw;
   Double_t  leadingTrackPtData, leadingTrackPt;
   Long64_t pTHardBin; // we are getting the pt hard bin from the tree this time
   Float_t cent, centData;
   Double_t pTRec, ptJet; // ml corrected data 
   Int_t motherParton;
   
   //so mcr is correctly normalized to one, not the response.       
   cout<<"cucu"<<endl;
   RooUnfoldResponse response;
   RooUnfoldResponse responsenotrunc;
   response.Setup(h1smeared,h1true);
   responsenotrunc.Setup(h1smearednocuts,h1fulleff);

   
   TFile *input1=TFile::Open("/home/alidock/PredictionTrees/predictionTree_NeuralNetwork_For_LHC15o_R040_050220_ConstList.root");
   TTree *data=(TTree*)input1->Get("NeuralNetwork_For_LHC15o_R040");
   Int_t nEvents=data->GetEntries();
   std::cout << nEvents << std::endl;
   data->SetBranchAddress("Jet_Pt", &hybridPtData); 
   data->SetBranchAddress("Predicted_Jet_Pt", &pTRec);
   data->SetBranchAddress("Jet_TrackPt0", &leadingTrackPtData); 
   data->SetBranchAddress("Event_Centrality", &centData);
   for(Int_t i = 0; i < nEvents; i++){
     data->GetEntry(i);
     if(centData > 10) continue;
     //if(leadingTrackPtData > 100) continue; 
     //if(leadingTrackPtData < 7)continue; 
     double EBscale = 1.;
     if(hybridPtData >= -20. && hybridPtData < 10.)       EBscale = 10.;
     else if(hybridPtData >= 10. && hybridPtData < 20.)   EBscale = 10.;
     else if(hybridPtData >= 20. && hybridPtData < 40.)   EBscale = 2.5;
     else if(hybridPtData >= 40. && hybridPtData < 60.)   EBscale = 1.25;
     else if(hybridPtData >= 60. && hybridPtData < 80.)   EBscale = 1.111;
     else if(hybridPtData >= 80. && hybridPtData < 100.)  EBscale = 1.111;
     else if(hybridPtData >= 100. && hybridPtData < 500.) EBscale = 1.0;
     if(pTRec >120 || pTRec <25) continue;
     h1raw->Fill(pTRec, EBscale); 
   }
   // previously derived pT hard bin scaling factors INCLUDING rk. index with PtHardBin Branch
   Double_t scalingFactors[20] = {0.484481, 0.412187,  0.398519, 0.262053, 0.132004, 0.0634448, 0.0223615, 0.00815064, 0.00349327,  0.00120855, 0.00048521, 0.000174209, 8.63207e-05, 4.0769e-05, 1.98943e-05, 1.29441e-05, 5.53066e-06,  3.71667e-06, 1.56311e-06, 2.04679e-06};
   //Double_t scalingFactors[20] = {0.3471,0.2978,0.2888,0.1894,0.9596e-02,4.613e-02,1.619e-02,5.920e-03,2.547e-03,8.795e-04,3.517e-04,1.272e-04,6.276e-05,2.956e-05,1.452e-05,7.395e-06,3.955e-06,2.111e-06,1.141e-06,1.472e-06};
   TFile *input2=TFile::Open("/home/alidock/PredictionTrees/predictionTree_NeuralNetwork_For_LHC16j5_Embedded_R040_050620_Quarks.root");
   TTree *mc=(TTree*)input2->Get("NeuralNetwork_For_LHC16j5_Embedded_R040"); 
   Int_t nEv=mc->GetEntries(); 
   // get the jet pT predicted by the ml
   mc->SetBranchAddress("Predicted_Jet_Pt", &ptJet); 
   mc->SetBranchAddress("Jet_MC_MatchedPartLevelJet_Pt", &ptJetMatch);
   mc->SetBranchAddress("Jet_Pt_Raw", &raw);
   //mc->SetBranchAddress("Jet_MC_MatchedPartLevelJet_Pt", &ptPart);
   mc->SetBranchAddress("PtHardBin", &pTHardBin);
   mc->SetBranchAddress("Jet_Pt", &hybridPt);
   //mc->SetBranchAddress("Event_Centrality", &cent);
   mc->SetBranchAddress("Jet_TrackPt0", &leadingTrackPt); 
   mc->SetBranchAddress("Jet_MC_MotherParton", &motherParton);
   mc->SetBranchAddress("Jet_MC_MatchedDetLevelJet_Distance", &dist);
   mc->SetBranchAddress("Jet_MC_TruePtFraction", &truePtFrac); 
   Int_t countm=0;
   for(int iEntry=0; iEntry< nEv; iEntry++){
     mc->GetEntry(iEntry);
     //if (cent > 10) continue;
     if (leadingTrackPt > 100) continue;
     //if (leadingTrackPt < 7) continue; 
     //if((truePtFrac) < 0.5*ptJetMatch || dist > 0.3) continue;
     if(pTHardBin < 4) continue;
     // select on quarks
     //if(motherParton > 10) continue; 
     double scalefactor = scalingFactors[pTHardBin-1];
     double EBscale = 1.;
     // put in if/else statements on the ptJet 
     if(hybridPt >= -20. && hybridPt < 10.)       EBscale = 10.;
     else if(hybridPt >= 10. && hybridPt < 20.)   EBscale = 10.;
     else if(hybridPt >= 20. && hybridPt < 40.)   EBscale = 2.5;
     else if(hybridPt >= 40. && hybridPt < 60.)   EBscale = 1.25;
     else if(hybridPt >= 60. && hybridPt < 80.)   EBscale = 1.111;
     else if(hybridPt >= 80. && hybridPt < 100.)  EBscale = 1.111;
     else if(hybridPt >= 100. && hybridPt < 500.) EBscale = 1.0;
     scalefactor*=EBscale;


     if(ptJetMatch < 10 || ptJetMatch > 250) continue;

     h1fulleff->Fill(ptJetMatch,scalefactor);  
     //h1smearedFullRange->Fill(ptJet, scalefactor);
     h1smearednocuts->Fill(ptJet,scalefactor);  
     responsenotrunc.Fill(ptJet,ptJetMatch,scalefactor);
     if(ptJet>120 || ptJet<25) continue;
     h1smeared->Fill(ptJet,scalefactor);
     //this is the half split to be the response 
     response.Fill(ptJet, ptJetMatch,scalefactor);
     //this is the generator level distribution for the pseudo data or our answer :)
     h1true->Fill(ptJetMatch,scalefactor);
      
   }
 

    
    TH1D *htrueptd=(TH1D*) h1fulleff->Clone("trueptd");
    TH1D *htruept=(TH1D*) h1fulleff->Clone( "truept");
    





 
    //////////efficiencies done////////////////////////////////////
 
    TFile *fout=new TFile (Form("Unfolding_NeuralNetwork_May9th_Part_Quarks.root"),"RECREATE");
    fout->cd();
    h1raw->SetName("raw");
    h1raw->Write();
    h1smeared->SetName("smeared");
    h1smeared->Write();
    h1smearedFullRange->Write();
    htrueptd->Write();
    h1true->SetName("true");
    h1true->Write();
    response.Write();
    responsenotrunc.Write();
    TH1D* htruthNoTrunc = (TH1D*)responsenotrunc.Htruth();
    htruthNoTrunc->SetName("htruthnotrunc");
    htruthNoTrunc->Write();
    TH1D* htruth = (TH1D*)response.Htruth();
    htruth->SetName("htruth");
    htruth->Write();
    for(int jar=1;jar<16;jar++){
      Int_t iter=jar;
      cout<<"iteration"<<iter<<endl;
      cout<<"==============Unfold h1====================="<<endl;

      RooUnfoldBayes unfold(&response, h1raw, iter, false);    // OR
      TH1D* hunf= (TH1D*) unfold.Hreco(errorTreatment);
      //FOLD BACK
      TH1* hfold = response.ApplyToTruth (hunf, "");

      TH2D *htempUnf=(TH2D*)hunf->Clone("htempUnf");          
      htempUnf->SetName(Form("Bayesian_Unfoldediter%d",iter));
      
      TH2D *htempFold=(TH2D*)hfold->Clone("htempFold");          
      htempFold->SetName(Form("Bayesian_Foldediter%d",iter));        

      htempUnf->Write();
      htempFold->Write();
	  
      ///HERE I GET THE COVARIANCE MATRIX/////
      /*
      if(iter==8){
	TMatrixD covmat= unfold.Ereco((RooUnfold::ErrorTreatment)RooUnfold::kCovariance);
	for(Int_t k=0;k<h2true->GetNbinsX();k++){
	  TH2D *hCorr= (TH2D*) CorrelationHistShape(covmat, Form("corr%d",k), "Covariance matrix",h2true->GetNbinsX(),h2true->GetNbinsY(),k);
	  TH2D *covshape=(TH2D*)hCorr->Clone("covshape");      
	  covshape->SetName(Form("pearsonmatrix_iter%d_binshape%d",iter,k));
	  covshape->SetDrawOption("colz");
	  covshape->Write();
	}
	
	for(Int_t k=0;k<h2true->GetNbinsY();k++){
	  TH2D *hCorr= (TH2D*) CorrelationHistPt(covmat, Form("corr%d",k), "Covariance matrix",h2true->GetNbinsX(),h2true->GetNbinsY(),k);
	  TH2D *covpt=(TH2D*)hCorr->Clone("covpt");      
	  covpt->SetName(Form("pearsonmatrix_iter%d_binpt%d",iter,k));
	  covpt->SetDrawOption("colz");
	  covpt->Write();
	  }
	  }*/
    }
	  
}
#ifndef __CINT__
int main () { RooSimplepTPbPb_ML(); return 0; }  // Main program when run stand-alone
#endif
