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

void RooSimpleRgpp(TString cFiles2="/Users/leticiacunqueiro/2019RawData/MCHard0.4/files.txt")
{
#ifdef __CINT__
  gSystem->Load("libRooUnfold");
#endif
  Int_t difference=1;
  Int_t Ppol=0;
  cout << "==================================== pick up the response matrix for background==========================" << endl;
  ///////////////////parameter setting
  RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;   
 
  //Get the tree for data
  TString fnamedata;
  //fnamedata="semicentralevent.root";
  fnamedata="/Users/leticiacunqueiro/2019RawData/ppdataHard0.4/ppdatazcut04.root";
  //fnamedata="semicentralsjet-wise.root";
  TFile *inputdata;
  inputdata=TFile::Open(fnamedata);
  TTree *data=(TTree*)inputdata->Get("JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRawTree_Data_NoSub_Incl"); 

  

   //***************************************************

       Double_t xbins[6];
       xbins[0]=20;
       xbins[1]=30;
         xbins[2]=40;
         xbins[3]=50;
	 xbins[4]=60;
	 //	 xbins[5]=70;
         xbins[5]=85;
        
       
	
      
   
         Double_t xbinsb[8];
	 xbinsb[0]=-0.05;
	 xbinsb[1]=0.;
	  xbinsb[2]=0.02;
         xbinsb[3]=0.04;
	  xbinsb[4]=0.06;
 	 xbinsb[5]=0.1;
	 xbinsb[6]=0.2;
         xbinsb[7]=0.35;
	
	 
         
        
        


	 Double_t xbinsc[9];
         xbinsc[0]=-0.05;
	 xbinsc[1]=0.;
	 xbinsc[2]=0.02;
         xbinsc[3]=0.04;
	 xbinsc[4]=0.06;
 	 xbinsc[5]=0.1;
	 xbinsc[6]=0.2;
         xbinsc[7]=0.35;
	 xbinsc[8]=0.6;
	 
   //the raw correlation
   TH2D *h2raw(0);
   h2raw=new TH2D("raw","raw",7,xbinsb,5,xbins);
   //detector measure level
   TH2D *h2smeared(0);
   h2smeared=new TH2D("smeared","smeared",7,xbinsb,5,xbins);

     //detector measure level no cuts
   TH2D *h2smearednocuts(0);
   h2smearednocuts=new TH2D("smearednocuts","smearednocuts",8,xbinsc,8,0,160);
   //true correlations with measured cuts
    TH2D *h2true(0);
    h2true=new TH2D("true","true",8,xbinsc,8,0,160);
    //full true correlation
    TH2D *h2fulleff(0);
    h2fulleff=new TH2D("truef","truef",8,xbinsc,8,0,160);
   

 
  
   TH2D *hcovariance(0);
  hcovariance=new TH2D("covariance","covariance",10,0.,1.,10,0,1.);

  

  TH2D *effnum=(TH2D*)h2fulleff->Clone("effnum");
  TH2D *effdenom=(TH2D*)h2fulleff->Clone("effdenom");
 
   effnum->Sumw2();
   effdenom->Sumw2();
   h2smeared->Sumw2();
   h2true->Sumw2();
   h2raw->Sumw2();
   h2fulleff->Sumw2();
  
   Float_t ptJet,ptJetMatch,zg,zgMatch,rg,rgMatch,ktg,ktgMatch,ng,ngMatch,LeadingTrackPt,LeadingTrackPtDet,ngDet,subjet1,subjet2;
   Double_t counttot=0;
   Double_t ccount=0;
  Int_t nEv=0;; 
  //so mcr is correctly normalized to one, not the response.       
  cout<<"cucu"<<endl;
  nEv=data->GetEntries(); 
  cout<<"entries"<<nEv<<endl;
  data->SetBranchAddress("ptJet", &ptJet); 
  data->SetBranchAddress("zg", &zg); 
  data->SetBranchAddress("rg", &rg); 
  data->SetBranchAddress("ktg", &ktg);
  data->SetBranchAddress("ng", &ng);  
  
 
   for(int iEntry=0; iEntry< nEv; iEntry++){
   data->GetEntry(iEntry); 
   if(ptJet>85 || ptJet<20) continue;
   if(rg>0.35) continue;
   if(zg<0.4) rg=-0.025;
   
   if(rg<0) ccount=ccount+1;
   counttot=counttot+1;
   
   h2raw->Fill(rg,ptJet);}
 
   cout<<ccount/counttot<<"counts"<<endl;
   
   h2raw->Draw("text");
  

  

  ifstream infile2;
  infile2.open(cFiles2.Data());
  char filename2[300];
    RooUnfoldResponse response;
   RooUnfoldResponse responsenotrunc;
   response.Setup(h2smeared,h2true);
   responsenotrunc.Setup(h2smearednocuts,h2fulleff);
  
  int count=0;
    while(infile2>>filename2){
    int pthardbin=0;
    
    TFile *input=TFile::Open(filename2);
    TList *list=(TList*) input->Get("JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRaw_PythiaDef_NoSub_Incl");
   
  
     TProfile *hcross=(TProfile*)list->FindObject("fHistXsection");
     TH1D *htrials=(TH1D*)list->FindObject("fHistTrials");
         
        for(Int_t i=1;i<=htrials->GetNbinsX();i++){
	if(htrials->GetBinContent(i)!=0) pthardbin=i;}
	double scalefactor=hcross->Integral(pthardbin,pthardbin)/htrials->Integral(pthardbin,pthardbin);

    
       TTree *mc=(TTree*)input->Get("JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRawTree_PythiaDef_NoSub_Incl"); 
       Int_t nEv=mc->GetEntries(); 
   
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
      
      
     
      for(int iEntry=0; iEntry< nEv; iEntry++){
       mc->GetEntry(iEntry);
       
       if(ptJetMatch>160 ) continue;
      
       if(rgMatch>0.6) continue;
       if(zgMatch<0.4) rgMatch=-0.025;
        h2fulleff->Fill(rgMatch,ptJetMatch,scalefactor);  
        h2smearednocuts->Fill(rg,ptJet,scalefactor);  
        responsenotrunc.Fill(rg,ptJet,rgMatch,ptJetMatch,scalefactor);

       if(ptJet>85 || ptJet<20) continue;
          if(rg>0.35) continue;
          if(zg<0.4) rg=-0.025; 
        h2smeared->Fill(rg,ptJet,scalefactor);
      	h2true->Fill(rgMatch,ptJetMatch,scalefactor);
      	response.Fill(rg,ptJet,rgMatch,ptJetMatch,scalefactor);
      
      }
      count=count+1;
      cout<<"closing "<<count<<endl;
      input->Close();
      
     
    }
 

     
       TH1F *htrueptd=(TH1F*) h2fulleff->ProjectionX("trueptd",1,-1);
       TH1F *htruept=(TH1F*) h2fulleff->ProjectionY( "truept",1,-1); 
 
     


  
 TH2D* hfold=(TH2D*)h2raw->Clone("hfold");
 hfold->Sumw2();
 
 //////////efficiencies done////////////////////////////////////
 TH1D * effok=(TH1D *)h2true->ProjectionX("effok",2,2);
 TH1D * effok1=(TH1D *)h2fulleff->ProjectionX("effok2",2,2);
 effok->Divide(effok1);
 effok->SetName("correff20-40");
 
 TH1D * effok3=(TH1D *)h2true->ProjectionX("effok3",3,3);
  TH1D * effok4=(TH1D *)h2fulleff->ProjectionX("effok4",3,3);
 effok3->Divide(effok4);
  effok3->SetName("correff40-60"); 

  TH1D * effok5=(TH1D *)h2true->ProjectionX("effok5",4,4);
  TH1D * effok6=(TH1D *)h2fulleff->ProjectionX("effok6",4,4);
 effok5->Divide(effok6);
 effok5->SetName("correff60-80"); 

  
 

 
 TFile *fout=new TFile (Form("UnfoldRgzg0.4ppDefaultRmax025.root"),"RECREATE");
 fout->cd();
 effok->Write(); 
 effok3->Write();
 effok5->Write();
  
  h2raw->SetName("raw");
  h2raw->Write();
  h2smeared->SetName("smeared");
  h2smeared->Write();
  h2fulleff->Write();
  h2true->Write();
  htrueptd->Write();
  
     for(int jar=1;jar<16;jar++){
      Int_t iter=jar;
      cout<<"iteration"<<iter<<endl;
      cout<<"==============Unfold h1====================="<<endl;

      RooUnfoldBayes   unfold(&response, h2raw, iter);    // OR
      TH2D* hunf= (TH2D*) unfold.Hreco(errorTreatment);
       TH2D* hfold= (TH2D*)response.ApplyToTruth(hunf,"");
      //FOLD BACK
   
     




  
          TH2D *htempUnf=(TH2D*)hunf->Clone("htempUnf");          
	  htempUnf->SetName(Form("Bayesian_Unfoldediter%d.root",iter));
           
	    TH2D *htempFold=(TH2D*)hfold->Clone("htempFold");          
	  htempFold->SetName(Form("Bayesian_Foldediter%d.root",iter));        


           






     
      	  htempUnf->Write();
	  htempFold->Write();
	  



//         	  ///HERE I GET THE COVARIANCE MATRIX/////

//             if(iter==8){

//           TMatrixD covmat= unfold.Ereco((RooUnfold::ErrorTreatment)RooUnfold::kCovariance);
//             for(Int_t k=0;k<h2true->GetNbinsX();k++){
// 	    TH2D *hCorr= (TH2D*) CorrelationHistShape(covmat, Form("corr%d",k), "Covariance matrix",h2true->GetNbinsX(),h2true->GetNbinsY(),k);
//             TH2D *covshape=(TH2D*)hCorr->Clone("covshape");      
// 	    covshape->SetName(Form("pearsonmatrix_iter%d_binshape%d",iter,k));
// 	    covshape->SetDrawOption("colz");
//             covshape->Write();
// 	  }

//             for(Int_t k=0;k<h2true->GetNbinsY();k++){
// 	    TH2D *hCorr= (TH2D*) CorrelationHistPt(covmat, Form("corr%d",k), "Covariance matrix",h2true->GetNbinsX(),h2true->GetNbinsY(),k);
//             TH2D *covpt=(TH2D*)hCorr->Clone("covpt");      
// 	    covpt->SetName(Form("pearsonmatrix_iter%d_binpt%d",iter,k));
// 	    covpt->SetDrawOption("colz");
//             covpt->Write();
// 	  }

// 	  }


     

	  
     


     
     }}



   
#ifndef __CINT__
int main () { RooSimpleRgpp(); return 0; }  // Main program when run stand-alone
#endif
