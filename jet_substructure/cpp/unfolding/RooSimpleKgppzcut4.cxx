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

void RooSimpleKgppzcut4(Int_t kindex=0)
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
  fnamedata="/Users/leticiacunqueiro/2019RawData/RaymondLeadKt/pp/ppraymond.root";
  //fnamedata="semicentralsjet-wise.root";
  TFile *inputdata;
  inputdata=TFile::Open(fnamedata);
  TTree *data=(TTree*)inputdata->Get("tree"); 


  TString cFiles2;
   cFiles2="/Users/leticiacunqueiro/2019RawData/RaymondLeadKt/pythia/2110/skim/files.txt";
   if(kindex==2)  cFiles2="/Users/leticiacunqueiro/2019RawData/RaymondLeadKt/pythia/2161_Eff/skim/files.txt";
  ifstream infile2;
  infile2.open(cFiles2.Data());
  char filename2[300];

  TFile *RatioFile =new TFile("/Users/leticiacunqueiro/2019RawData/RaymondLeadKt/pythia/2110/skim/fileRatioDataMCppzcut4.root"); 
   TH2D* historatio; 
   historatio=(TH2D*)RatioFile->Get("weight");
  
  //k=0 default
  //k=1 bin
  //k=2 efficiency
  //kindex==3 truncation low
  //k=4 truncation high
  //kindex==5 prior
   //kindex==6 displacement

   //***************************************************

       Double_t xbins[6];
         xbins[0]=20;
         xbins[1]=30;
         xbins[2]=40;
         xbins[3]=50;
	 xbins[4]=60;
 	 xbins[5]=85;
         
	 if(kindex==3) xbins[0]=17;
	 if(kindex==4) xbins[0]=23;
       
         Double_t xbinsb[9];

         xbinsb[0]=0.25;
 	 xbinsb[1]=0.5;
	 xbinsb[2]=1.;
         xbinsb[3]=1.5;
	 xbinsb[4]=2.5;
	 xbinsb[5]=4;
         xbinsb[6]=6;
	 xbinsb[7]=8;
         xbinsb[8]=9;

	 if(kindex==6){
      
         xbinsb[0]=0;
	 xbinsb[1]=0.25;
 	 xbinsb[2]=0.5;
	 xbinsb[3]=1.;
         xbinsb[4]=1.5;
	 xbinsb[5]=2.5;
	 xbinsb[6]=4;
         xbinsb[7]=6;
	 xbinsb[8]=8;
        
	 
	 }
	 
        Double_t xbinsc[9];
	 xbinsc[0]=-0.5;
	 xbinsc[1]=0.;
         xbinsc[2]=0.5;
 	 xbinsc[3]=1;
	 xbinsc[4]=2;
         xbinsc[5]=4;
	 xbinsc[6]=6;
	 xbinsc[7]=8;
	 xbinsc[8]=15;


	 if(kindex==1){
     
         xbinsb[0]=0.25;
 	 xbinsb[1]=0.4;
	 xbinsb[2]=0.8;
         xbinsb[3]=1.3;
	 xbinsb[4]=2.4;
	 xbinsb[5]=3.9;
         xbinsb[6]=6.1;
	 xbinsb[7]=8;
	 xbinsb[8]=9;
        	 }

	 
	 
	 
	 
   //the raw correlation
   TH2D *h2raw(0);
   h2raw=new TH2D("raw","raw",8,xbinsb,5,xbins);
   //detector measure level
   TH2D *h2smeared(0);
   h2smeared=new TH2D("smeared","smeared",8,xbinsb,5,xbins);

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
  
   Float_t jet_pt_data,leading_kt_z_cut_04_kt;
 
  Int_t nEv=0;; 
  //so mcr is correctly normalized to one, not the response.       
  cout<<"cucu"<<endl;
  nEv=data->GetEntries(); 
  cout<<"entries"<<nEv<<endl;
  data->SetBranchAddress("jet_pt_data", &jet_pt_data); 
  data->SetBranchAddress("leading_kt_z_cut_04_data_kt", &leading_kt_z_cut_04_kt); 
  
  
 
   for(int iEntry=0; iEntry< nEv; iEntry++){
   data->GetEntry(iEntry); 
   if(jet_pt_data>xbins[5] || jet_pt_data<xbins[0]) continue;
   if(kindex==6){
   if(leading_kt_z_cut_04_kt>8) continue;
   if(leading_kt_z_cut_04_kt<0.25 && leading_kt_z_cut_04_kt>=0) continue;
   if(leading_kt_z_cut_04_kt<0) leading_kt_z_cut_04_kt=0.12;}

   if(kindex!=6){
   if(leading_kt_z_cut_04_kt>8) continue;
   if(leading_kt_z_cut_04_kt<0.25 && leading_kt_z_cut_04_kt>=0) continue;
   if(leading_kt_z_cut_04_kt<0) leading_kt_z_cut_04_kt=8.5;}

   
   
   h2raw->Fill(leading_kt_z_cut_04_kt,jet_pt_data);}
 
 
   
   h2raw->Draw("text");
  

   Float_t leading_kt_z_cut_04_data_kt, leading_kt_z_cut_04_matched_kt, jet_pt_matched,scale_factor;
   

  
    RooUnfoldResponse response;
   RooUnfoldResponse responsenotrunc;
   response.Setup(h2smeared,h2true);
   responsenotrunc.Setup(h2smearednocuts,h2fulleff);
  
  int count=0;
    while(infile2>>filename2){
      //  int pthardbin=0;
    
      TFile *input=TFile::Open(filename2);
    
    
       TTree *mc=(TTree*)input->Get("tree"); 
       Int_t nEv=mc->GetEntries(); 
   
      mc->SetBranchAddress("jet_pt_data", &jet_pt_data); 
      mc->SetBranchAddress("jet_pt_matched", &jet_pt_matched);
      mc->SetBranchAddress("scale_factor", &scale_factor);
      mc->SetBranchAddress("leading_kt_z_cut_04_data_kt", &leading_kt_z_cut_04_data_kt); 
      mc->SetBranchAddress("leading_kt_z_cut_04_matched_kt", &leading_kt_z_cut_04_matched_kt); 
      
     
     
      for(int iEntry=0; iEntry< nEv; iEntry++){
       mc->GetEntry(iEntry);
       
       if(jet_pt_matched>160 ) continue;
      
       if(leading_kt_z_cut_04_matched_kt>15) continue;
       if(leading_kt_z_cut_04_matched_kt<0) leading_kt_z_cut_04_matched_kt=-0.25;
      
        h2fulleff->Fill(leading_kt_z_cut_04_matched_kt,jet_pt_matched,scale_factor);  
        h2smearednocuts->Fill(leading_kt_z_cut_04_data_kt,jet_pt_data,scale_factor);  
        responsenotrunc.Fill(leading_kt_z_cut_04_data_kt,jet_pt_data,leading_kt_z_cut_04_matched_kt,jet_pt_matched,scale_factor);

       if(jet_pt_data>xbins[5] || jet_pt_data<xbins[0]) continue;

   if(kindex==6){
   if(leading_kt_z_cut_04_data_kt>8) continue;
   if(leading_kt_z_cut_04_data_kt<0.25 && leading_kt_z_cut_04_data_kt>=0) continue;
   if(leading_kt_z_cut_04_data_kt<0) leading_kt_z_cut_04_data_kt=0.12;}

   if(kindex!=6){
   if(leading_kt_z_cut_04_data_kt>8) continue;
   if(leading_kt_z_cut_04_data_kt<0.25 && leading_kt_z_cut_04_data_kt>=0) continue;
   if(leading_kt_z_cut_04_data_kt<0) leading_kt_z_cut_04_data_kt=8.5;}
       
           
      
	  h2smeared->Fill(leading_kt_z_cut_04_data_kt,jet_pt_data,scale_factor);
      	h2true->Fill(leading_kt_z_cut_04_matched_kt,jet_pt_matched,scale_factor);

         Double_t myw=1;
	 if(kindex==5){

              if(jet_pt_matched<20){
		  if(leading_kt_z_cut_04_matched_kt<0.)  myw=historatio->GetBinContent(8,1);
		  if(leading_kt_z_cut_04_matched_kt>=0. &&leading_kt_z_cut_04_matched_kt<0.5) myw=historatio->GetBinContent(1,1);
		   if(leading_kt_z_cut_04_matched_kt>=0.5 &&leading_kt_z_cut_04_matched_kt<1) myw=historatio->GetBinContent(2,1);
		    if(leading_kt_z_cut_04_matched_kt>=1 &&leading_kt_z_cut_04_matched_kt<2) myw=historatio->GetBinContent(3,1);
		      if(leading_kt_z_cut_04_matched_kt>=2 &&leading_kt_z_cut_04_matched_kt<4) myw=historatio->GetBinContent(5,1);
		       if(leading_kt_z_cut_04_matched_kt>=4 &&leading_kt_z_cut_04_matched_kt<6) myw=historatio->GetBinContent(6,1);
		        if(leading_kt_z_cut_04_matched_kt>=6 &&leading_kt_z_cut_04_matched_kt<8) myw=historatio->GetBinContent(7,1);
			 if(leading_kt_z_cut_04_matched_kt>=8) myw=historatio->GetBinContent(7,1);
		}

                 	if(jet_pt_matched>=20 && jet_pt_matched<30){
                if(leading_kt_z_cut_04_matched_kt<0.)  myw=historatio->GetBinContent(8,1);
		  if(leading_kt_z_cut_04_matched_kt>=0. &&leading_kt_z_cut_04_matched_kt<0.5) myw=historatio->GetBinContent(1,1);
		   if(leading_kt_z_cut_04_matched_kt>=0.5 &&leading_kt_z_cut_04_matched_kt<1) myw=historatio->GetBinContent(2,1);
		    if(leading_kt_z_cut_04_matched_kt>=1 &&leading_kt_z_cut_04_matched_kt<2) myw=historatio->GetBinContent(3,1);
		      if(leading_kt_z_cut_04_matched_kt>=2 &&leading_kt_z_cut_04_matched_kt<4) myw=historatio->GetBinContent(5,1);
		       if(leading_kt_z_cut_04_matched_kt>=4 &&leading_kt_z_cut_04_matched_kt<6) myw=historatio->GetBinContent(6,1);
		        if(leading_kt_z_cut_04_matched_kt>=6 &&leading_kt_z_cut_04_matched_kt<8) myw=historatio->GetBinContent(7,1);
			 if(leading_kt_z_cut_04_matched_kt>=8) myw=historatio->GetBinContent(7,1);
                 
			  
			      
		}


                  	if(jet_pt_matched>=30 && jet_pt_matched<40){
		  if(leading_kt_z_cut_04_matched_kt<0.)  myw=historatio->GetBinContent(8,2);
		  if(leading_kt_z_cut_04_matched_kt>=0. &&leading_kt_z_cut_04_matched_kt<0.5) myw=historatio->GetBinContent(1,2);
		   if(leading_kt_z_cut_04_matched_kt>=0.5 &&leading_kt_z_cut_04_matched_kt<1) myw=historatio->GetBinContent(2,2);
		    if(leading_kt_z_cut_04_matched_kt>=1 &&leading_kt_z_cut_04_matched_kt<2) myw=historatio->GetBinContent(3,2);
		      if(leading_kt_z_cut_04_matched_kt>=2 &&leading_kt_z_cut_04_matched_kt<4) myw=historatio->GetBinContent(5,2);
		       if(leading_kt_z_cut_04_matched_kt>=4 &&leading_kt_z_cut_04_matched_kt<6) myw=historatio->GetBinContent(6,2);
		        if(leading_kt_z_cut_04_matched_kt>=6 &&leading_kt_z_cut_04_matched_kt<8) myw=historatio->GetBinContent(7,2);
			 if(leading_kt_z_cut_04_matched_kt>=8) myw=historatio->GetBinContent(7,2);
			      
		}

			
		
		
			if(jet_pt_matched>=40 && jet_pt_matched<50){
                  
                   if(leading_kt_z_cut_04_matched_kt<0.)  myw=historatio->GetBinContent(8,3);
		  if(leading_kt_z_cut_04_matched_kt>=0. &&leading_kt_z_cut_04_matched_kt<0.5) myw=historatio->GetBinContent(1,3);
		   if(leading_kt_z_cut_04_matched_kt>=0.5 &&leading_kt_z_cut_04_matched_kt<1) myw=historatio->GetBinContent(2,3);
		    if(leading_kt_z_cut_04_matched_kt>=1 &&leading_kt_z_cut_04_matched_kt<2) myw=historatio->GetBinContent(3,3);
		      if(leading_kt_z_cut_04_matched_kt>=2 &&leading_kt_z_cut_04_matched_kt<4) myw=historatio->GetBinContent(5,3);
		       if(leading_kt_z_cut_04_matched_kt>=4 &&leading_kt_z_cut_04_matched_kt<6) myw=historatio->GetBinContent(6,3);
		        if(leading_kt_z_cut_04_matched_kt>=6 &&leading_kt_z_cut_04_matched_kt<8) myw=historatio->GetBinContent(7,3);
			 if(leading_kt_z_cut_04_matched_kt>=8) myw=historatio->GetBinContent(7,3);
			  
				  }

			
				if(jet_pt_matched>=50 && jet_pt_matched<60){
	
		 if(leading_kt_z_cut_04_matched_kt<0.)  myw=historatio->GetBinContent(8,4);
		  if(leading_kt_z_cut_04_matched_kt>=0. &&leading_kt_z_cut_04_matched_kt<0.5) myw=historatio->GetBinContent(1,4);
		   if(leading_kt_z_cut_04_matched_kt>=0.5 &&leading_kt_z_cut_04_matched_kt<1) myw=historatio->GetBinContent(2,4);
		    if(leading_kt_z_cut_04_matched_kt>=1 &&leading_kt_z_cut_04_matched_kt<2) myw=historatio->GetBinContent(3,4);
		      if(leading_kt_z_cut_04_matched_kt>=2 &&leading_kt_z_cut_04_matched_kt<4) myw=historatio->GetBinContent(5,4);
		       if(leading_kt_z_cut_04_matched_kt>=4 &&leading_kt_z_cut_04_matched_kt<6) myw=historatio->GetBinContent(6,4);
		        if(leading_kt_z_cut_04_matched_kt>=6 &&leading_kt_z_cut_04_matched_kt<8) myw=historatio->GetBinContent(7,4);
			 if(leading_kt_z_cut_04_matched_kt>=8) myw=historatio->GetBinContent(7,4);
				  }

				if(jet_pt_matched>=60 && jet_pt_matched<85){

		   if(leading_kt_z_cut_04_matched_kt<0.)  myw=historatio->GetBinContent(8,5);
		  if(leading_kt_z_cut_04_matched_kt>=0. &&leading_kt_z_cut_04_matched_kt<0.5) myw=historatio->GetBinContent(1,5);
		   if(leading_kt_z_cut_04_matched_kt>=0.5 &&leading_kt_z_cut_04_matched_kt<1) myw=historatio->GetBinContent(2,5);
		    if(leading_kt_z_cut_04_matched_kt>=1 &&leading_kt_z_cut_04_matched_kt<2) myw=historatio->GetBinContent(3,5);
		      if(leading_kt_z_cut_04_matched_kt>=2 &&leading_kt_z_cut_04_matched_kt<4) myw=historatio->GetBinContent(5,5);
		       if(leading_kt_z_cut_04_matched_kt>=4 &&leading_kt_z_cut_04_matched_kt<6) myw=historatio->GetBinContent(6,5);
		        if(leading_kt_z_cut_04_matched_kt>=6 &&leading_kt_z_cut_04_matched_kt<8) myw=historatio->GetBinContent(7,5);
			 if(leading_kt_z_cut_04_matched_kt>=8) myw=historatio->GetBinContent(7,5);
				     }


	



		 if(jet_pt_matched>=85){

		  if(leading_kt_z_cut_04_matched_kt<0.)  myw=historatio->GetBinContent(8,5);
		  if(leading_kt_z_cut_04_matched_kt>=0. &&leading_kt_z_cut_04_matched_kt<0.5) myw=historatio->GetBinContent(1,5);
		   if(leading_kt_z_cut_04_matched_kt>=0.5 &&leading_kt_z_cut_04_matched_kt<1) myw=historatio->GetBinContent(2,5);
		    if(leading_kt_z_cut_04_matched_kt>=1 &&leading_kt_z_cut_04_matched_kt<2) myw=historatio->GetBinContent(3,5);
		      if(leading_kt_z_cut_04_matched_kt>=2 &&leading_kt_z_cut_04_matched_kt<4) myw=historatio->GetBinContent(5,5);
		       if(leading_kt_z_cut_04_matched_kt>=4 &&leading_kt_z_cut_04_matched_kt<6) myw=historatio->GetBinContent(6,5);
		        if(leading_kt_z_cut_04_matched_kt>=6 &&leading_kt_z_cut_04_matched_kt<8) myw=historatio->GetBinContent(7,5);
			 if(leading_kt_z_cut_04_matched_kt>=8) myw=historatio->GetBinContent(7,5);
				     }
	
				





	   
	 }
      	response.Fill(leading_kt_z_cut_04_data_kt,jet_pt_data,leading_kt_z_cut_04_matched_kt,jet_pt_matched,scale_factor*myw);
      
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

  
 

 
 TFile *fout;
 if(kindex==0) fout=new TFile (Form("UnfoldKgzg0.4ppDefaultRmax025.root"),"RECREATE");
 if(kindex==1) fout=new TFile (Form("UnfoldKgzg0.4ppDefaultRmax025Bin.root"),"RECREATE");
 if(kindex==2) fout=new TFile (Form("UnfoldKgzg0.4ppDefaultRmax025Eff.root"),"RECREATE");
 if(kindex==3) fout=new TFile (Form("UnfoldKgzg0.4ppDefaultRmax025TruncLow.root"),"RECREATE");
 if(kindex==4) fout=new TFile (Form("UnfoldKgzg0.4ppDefaultRmax025TruncHigh.root"),"RECREATE");
 if(kindex==5) fout=new TFile (Form("UnfoldKgzg0.4ppDefaultRmax025Prior.root"),"RECREATE");
  if(kindex==6) fout=new TFile (Form("UnfoldKgzg0.4ppDefaultRmax025Displaced.root"),"RECREATE");
 fout->cd();
 effok->Write(); 
 effok3->Write();
 effok5->Write();
  
  h2raw->SetName("raw");
  h2raw->Write();
  h2smeared->SetName("smeared");
  h2smeared->Write();
  h2fulleff->Write();
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
int main () { RooSimpleKgppzcut4(); return 0; }  // Main program when run stand-alone
#endif
