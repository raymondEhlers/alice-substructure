
#include "fastjet/Selector.hh" //.......... Background Sutraction event by event
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"//.......... Background Sutraction event by event
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/ClusterSequenceAreaBase.hh"
#include "fastjet/tools/Subtractor.hh"
#include "fastjet/contrib/ConstituentSubtractor.hh"
#include "fastjet/contrib/Nsubjettiness.hh"
#include "fastjet/contrib/SoftDrop.hh"
#include "fastjet/contrib/ModifiedMassDropTagger.hh"
#include "fastjet/contrib/Recluster.hh"
#include <algorithm>
#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include <string>
#include <cstring>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include "Pythia8/Pythia.h"
#include "TTree.h"
#include "THnSparse.h"
#include "TProfile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TFile.h"
#include "TClonesArray.h"
#include "TFile.h"
#include "TList.h"
#include "TVector3.h"
#include "TMath.h"
#include "THnSparse.h"
#include "TNtuple.h"
#include "TString.h"
#include "TRandom3.h"
#include "TH1D.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include <ctime>
#include <iostream> // needed for io
#include <cstdio>   // needed for io
#include <valarray>


using namespace Pythia8;

double Calculate_pX( double pT, double eta, double phi)
{
	return(pT*TMath::Cos(phi));
}

double Calculate_pY( double pT, double eta, double phi)
{
	return(pT*TMath::Sin(phi));
}

double Calculate_pZ( double pT, double eta, double phi)
{
	return( pT*TMath::SinH(eta) );
}

double Calculate_E( double pT, double eta, double phi)
{
	double pZ = Calculate_pZ(pT,eta, phi);

	return( TMath::Sqrt(pT*pT + pZ*pZ) );
}

//__________________________________________________________________________

bool EtaCut(fastjet::PseudoJet fjJet, double etaMin, double etaMax) {
   if(fjJet.eta() > etaMax || fjJet.eta() < etaMin){
      return false;
   }else{
      return true;
   }
}
//_________________________________________________________________________
Double_t RelativePhi(Double_t mphi,Double_t vphi){
   //Get relative azimuthal angle of two particles -pi to pi
   if      (vphi < -TMath::Pi()) vphi += TMath::TwoPi();
   else if (vphi > TMath::Pi())  vphi -= TMath::TwoPi();

   if      (mphi < -TMath::Pi()) mphi += TMath::TwoPi();
   else if (mphi > TMath::Pi())  mphi -= TMath::TwoPi();

   Double_t dphi = mphi - vphi;
   if      (dphi < -TMath::Pi()) dphi += TMath::TwoPi();
   else if (dphi > TMath::Pi())  dphi -= TMath::TwoPi();

   return dphi;//dphi in [-Pi, Pi]
}
//__________________________________________________________________________
float fShapesVar[11];
TTree *fTreeResponse = new TTree("variables", "variables");


Bool_t CompareSubjets(fastjet::PseudoJet subDet,  fastjet::PseudoJet subHyb, std::vector<fastjet::PseudoJet> constDet, std::vector<fastjet::PseudoJet> constHyb)
{
  double pT_det = subDet.perp();
  double sumpT = 0;
  double delta =  0.01;

  for (int i = 0; i < constDet.size(); i++)
      {
	double eta_det = constDet.at(i).eta();
	double phi_det = constDet.at(i).phi();
	for (int j  = 0; j < constHyb.size(); j++)
	  {
	    double eta_hyb = constHyb.at(j).eta();
	    double phi_hyb = constHyb.at(j).phi();
	    double deta = eta_hyb - eta_det;
	    deta = std::sqrt(deta*deta);
	    if (deta > delta) continue;
	    double dphi = phi_hyb - phi_det;
	    dphi = std::sqrt(dphi*dphi);
	    if (dphi > delta) continue;
	    sumpT+=constDet.at(i).perp();
	  }
      }
  if (sumpT/pT_det > 0.5) return true;
  else return false;
}





//
void IterativeDeclustering(fastjet::PseudoJet jet,Double_t zcut,fastjet::PseudoJet jet2, int MATCH )
{
   	double flagSubjet=0;


	double zg=0;
	double rg=0;
	double nsd=0;
		double zgp=0;
	double rgp=0;
	double nsdp=0;
	int subflag1=0;
	int subflag2=0;
	//_________________________________________________________________________________________

   	// Reclustering settings, using generic kt algorithm with p=1/2, which orders in vacuum formation time

	//fastjet::JetAlgorithm jet_algo(fastjet::cambridge_algorithm);
	//double jet_radius_ca = 1.0;
   	//fastjet::JetDefinition jet_def(jet_algo, jet_radius_ca,static_cast<fastjet::RecombinationScheme>(0), fastjet::Best);

   	double jet_radius_ca = 1.0;
   	fastjet::JetDefinition jet_def(fastjet::genkt_algorithm,jet_radius_ca,0,static_cast<fastjet::RecombinationScheme>(0), fastjet::Best);

   	// Reclustering jet constituents with new algorithm
      	try
	{
      		std::vector<fastjet::PseudoJet> particles = jet.constituents();
      		fastjet::ClusterSequence cs_gen(particles, jet_def);
      		std::vector<fastjet::PseudoJet> output_jets = cs_gen.inclusive_jets(0);
      		output_jets = sorted_by_pt(output_jets);

      		// input jet but reclustered with ca
      		fastjet::PseudoJet jj = output_jets[0];

      		// Auxiliar variables
      		fastjet::PseudoJet j1;  // subjet 1 (largest pt)
      		fastjet::PseudoJet j2;  // subjet 2 (smaller pt)

                fastjet::PseudoJet sub1;  // subjet 1 (largest pt)
      		fastjet::PseudoJet sub2;  // subjet 2 (smaller pt)
                fastjet::PseudoJet subp1;  // subjet 1 (largest pt)
      		fastjet::PseudoJet subp2;  // subjet 2 (smaller pt)


      		// Unclustering jet
      		while(jj.has_parents(j1,j2))
		{

        		if(j1.perp() < j2.perp()) std::swap(j1,j2);



        		// Calculate deltaR and Zg between j1 and j2
        		double delta_R = j1.delta_R(j2);


			if( j2.perp()/(j2.perp()+j1.perp()) > zcut)
			  {

				nsd++;
			}
			double z = j2.perp()/(j2.perp()+j1.perp());

			if ((flagSubjet !=1) && (z > zcut)) {flagSubjet = 1; zg = z; rg  = delta_R;
			  sub1=j1;
			  sub2=j2;
			}

        		//continue unclustering
        		jj=j1;
      		}


		flagSubjet=0;



      		std::vector<fastjet::PseudoJet> particlesprobe = jet2.constituents();
      		fastjet::ClusterSequence cs_genp(particlesprobe, jet_def);
      		std::vector<fastjet::PseudoJet> output_jets_probe = cs_genp.inclusive_jets(0);
      		output_jets_probe = sorted_by_pt(output_jets_probe);

      		// input jet but reclustered with ca
      		fastjet::PseudoJet jjp = output_jets_probe[0];

      		// Auxiliar variables
      		fastjet::PseudoJet j1p;  // subjet 1 (largest pt)
      		fastjet::PseudoJet j2p;  // subjet 2 (smaller pt)


      		// Unclustering jet
      		while(jjp.has_parents(j1p,j2p))
		{

        		if(j1p.perp() < j2p.perp()) std::swap(j1p,j2p);



        		// Calculate deltaR and Zg between j1 and j2
        		double delta_Rp = j1p.delta_R(j2p);


			if( j2p.perp()/(j2p.perp()+j1p.perp()) > zcut)
			  {

				nsdp++;
			}
			double zp = j2p.perp()/(j2p.perp()+j1p.perp());

			if ((flagSubjet !=1) && (zp > zcut)) {flagSubjet = 1; zgp = zp; rgp  = delta_Rp;
			  subp1=j1p;
			  subp2=j2p;}

        		//continue unclustering
        		jjp=j1p;
      		}



		//////compare subjets
	  	 if(MATCH>=0){
	           std::vector<fastjet::PseudoJet> const1 = sorted_by_pt(sub1.constituents());
                   std::vector<fastjet::PseudoJet> const2 = sorted_by_pt(sub2.constituents());
		   std::vector<fastjet::PseudoJet> constp1 = sorted_by_pt(subp1.constituents());
		   std::vector<fastjet::PseudoJet> constp2 = sorted_by_pt(subp2.constituents());


		    if(CompareSubjets(subp1,sub1,constp1,const1)==kTRUE) subflag1=1;
		    else if(CompareSubjets(subp1,sub2,constp1,const2)==kTRUE) subflag1=2;
                    else subflag1=3;

		    if(CompareSubjets(subp2,sub2,constp2,const2)==kTRUE) subflag2=2;
		    else if(CompareSubjets(subp2,sub1,constp2,const1)==kTRUE) subflag2=1;
                    else subflag2=3;

		}


		  fShapesVar[0]=output_jets_probe[0].perp();
		  fShapesVar[2]=zgp;
		  fShapesVar[3]=nsdp;
		  fShapesVar[4]=rgp;

		  fShapesVar[1]=output_jets[0].perp();
                  fShapesVar[5]=zg;
                  fShapesVar[6]=nsd;
                  fShapesVar[7]=rg;
		  fShapesVar[8]=MATCH;
	          fShapesVar[9]=subflag1;
		  fShapesVar[10]=subflag2;



	} catch (fastjet::Error) { /*return -1;*/ }
}




//__________________________________________________________________________


int main(int argc, char* argv[])
{


   Int_t cislo = -1;                 //unique number for each file
   Int_t tune  = -1;                 //pythia tune
   Float_t zcut=0.2;
   Float_t pthatmin=5;
   Float_t pthatmax=5020;


 if(argc!=7){
   cout<<"Usage:"<<endl<<"./pygen <PythiaTune> <Number> <nEvts> <zcuts> <pthatmin> <pthatmax>"<<endl;
      return 0;
   }
   tune  = atoi(argv[1]);
   cislo = atoi(argv[2]);
   zcut    = atof(argv[4]);
   Int_t nEvents= atoi(argv[3]);   //(Int_t) 1e3 + 1.0;
   pthatmin=atof(argv[5]);
   pthatmax=atof(argv[6]);

   cout<<nEvents<<" events"<<endl;



   const Int_t nVar = 11;
   TTree *fTreeResposne = new TTree("variables", "variables");
   TString *fShapesVarNames = new TString [nVar];



  fShapesVarNames[0] = "ptJet";
  fShapesVarNames[1] = "ptJetProbe";
  fShapesVarNames[2] = "zg";
  fShapesVarNames[3] = "ng";
  fShapesVarNames[4] = "rg";
  fShapesVarNames[5] = "zgProbe";
  fShapesVarNames[6] = "ngProbe";
  fShapesVarNames[7] = "rgProbe";
  fShapesVarNames[8] = "match";
  fShapesVarNames[9] = "flag1";
  fShapesVarNames[10] = "flag2";
  for (Int_t ivar=0; ivar < nVar; ivar++){
    fTreeResponse->Branch(fShapesVarNames[ivar].Data(), &fShapesVar[ivar],Form("%s/F", fShapesVarNames[ivar].Data()));}



  //__________________________________________________________________________
//ANALYSIS SETTINGS

  double jetParameterR   = 0.4; //jet R
  double trackEtaCut     = 1;
  double trackLowPtCut   = 0.15; //GeV
  double naverage=1000;
  double sigman=40;
  TRandom3 *r3=new TRandom3();
//__________________________________________________________________________
//PYTHIA SETTINGS

  TString name;


 // Float_t ptHatMin=10;
 // Float_t ptHatMax=5020;
   cout<<"hello"<<endl;
  // Generator. Process selection. LHC initialization. Histogram.
  Pythia pythia;
  pythia.readString("Beams:idA = 2212"); //beam 1 proton
  pythia.readString("Beams:idB = 2212"); //beam 2 proton
  pythia.readString("Beams:eCM = 5020.");
  pythia.readString("Tune:pp = 5");  //tune 1-13    5=defaulr TUNE4C,  6=Tune 4Cx, 7=ATLAS MB Tune A2-CTEQ6L1
  cout<<"is it here"<<endl;
   pythia.readString("Random:setSeed = on");
        pythia.readString(Form("Random:seed = %d",cislo));
  cout<<"a ver"<<endl;
  pythia.readString("HardQCD:all = on");
  if(pthatmin<0 || pthatmax <0){
    pythia.readString("PhaseSpace:pTHatMin = 0."); // <<<<<<<<<<<<<<<<<<<<<<<
  }else{
    name = Form("PhaseSpace:pTHatMin = %f", (Float_t) pthatmin);
    pythia.readString(name.Data());
     name = Form("PhaseSpace:pTHatMax = %f", (Float_t) pthatmax);
    pythia.readString(name.Data());
  }



  pythia.readString("310:mayDecay  = off"); //K0s
  pythia.readString("3122:mayDecay = off"); //labda0
  pythia.readString("3112:mayDecay = off"); //sigma-
  pythia.readString("3212:mayDecay = off"); //sigma0
  pythia.readString("3222:mayDecay = off"); //sigma+
  pythia.readString("3312:mayDecay = off"); //xi-
  pythia.readString("3322:mayDecay = off"); //xi+
  pythia.readString("3334:mayDecay = off"); //omega-

  pythia.init();

//_________________________________________________________________________________________________
//FASTJET  SETTINGS

  double etamin_Sig = - trackEtaCut + jetParameterR; //signal jet eta range
  double etamax_Sig = - etamin_Sig;


  fastjet::Strategy strategy = fastjet::Best;
  fastjet::RecombinationScheme recombScheme = fastjet::E_scheme;

  fastjet::JetDefinition *jetDefAKT_Sig = NULL;

  jetDefAKT_Sig = new fastjet::JetDefinition(fastjet::antikt_algorithm, jetParameterR, recombScheme, strategy);

  fastjet::GhostedAreaSpec ghostareaspec(trackEtaCut, 1, 0.05); //ghost
  //max rap, repeat, ghostarea default 0.01
  fastjet::AreaType areaType = fastjet::active_area_explicit_ghosts;
  fastjet::AreaDefinition *areaDef = new fastjet::AreaDefinition(areaType, ghostareaspec);

  // Fastjet input
  std::vector<fastjet::PseudoJet> fjInputs1;

  std::vector<fastjet::PseudoJet> fjInputs2;
  //___________________________________________________
  //HISTOGRAMS
   cout<<"adios"<<endl;
  //After Thermal Particles: "A-Histograms"
  TH1D *hJetPt_A = new TH1D("hJetPt_A","hJetPt_A", 200,0.0,400.0);
  hJetPt_A->Sumw2();

  TH1D* hJetArea_A = new TH1D("hJetArea_A", "hJetArea_A", 400, 0.0, 4.0);
  hJetArea_A->Sumw2();

  TH1D *hJet_deltaR_A = new TH1D("hJet_deltaR_A", "hJet_deltaR_A", 100, 0.0, 1.0);
  hJet_deltaR_A->Sumw2();

  TH1D *hJet_deltaRg_A = new TH1D("hJet_deltaRg_A", "hJet_deltaRg_A", 100, 0.0, 1.0);
  hJet_deltaRg_A->Sumw2();

  TH2D *hJet_deltaRg_pT_A = new TH2D("hJet_deltaRg_pT_A", "hJet_deltaRg_pT_A", 200,0.0,400.0, 100, 0.0, 1.0);

  //Before Thermal Particles "B-Histograms"
  TH1D *hJetPt_B = new TH1D("hJetPt_B","hJetPt_B", 200,0.0,400.0);
  hJetPt_B->Sumw2();

  TH1D* hJetArea_B = new TH1D("hJetArea_B", "hJetArea_B", 400, 0.0, 4.0);
  hJetArea_B->Sumw2();

  TH1D *hJet_deltaR_B = new TH1D("hJet_deltaR_B", "hJet_deltaR_B", 100, 0.0, 1.0);
  hJet_deltaR_B->Sumw2();

  TH1D *hJet_deltaRg_B = new TH1D("hJet_deltaRg_B", "hJet_deltaRg_B", 100, 0.0, 1.0);
  hJet_deltaRg_B->Sumw2();

  TH2D *hJet_deltaRg_pT_B = new TH2D("hJet_deltaRg_pT_B", "hJet_deltaRg_pT_B", 200, 0.0, 400.0, 100, 0.0, 1.0);

  //PARTICLE DISTRIBUITIONS

  TH1D *h_pT = new TH1D("h_pT","",40000,0.0,400);
  TH1D *h_eta = new TH1D("h_eta","",200,-1.0,1.0);
  TH1D *h_phi = new TH1D("h_phi","",700,-3.5,3.5);

  TH1D *hP_pT = new TH1D("hP_pT","",40000,0.0,400);
  TH1D *hP_eta = new TH1D("hP_eta","",200,-1.0,1.0);
  TH1D *hP_phi = new TH1D("hP_phi","",700,-3.5,3.5);

  TH1D *hT_pT = new TH1D("hT_pT","",40000,0.0,400);
  TH1D *hT_eta = new TH1D("hT_eta","",200,-1.0,1.0);
  TH1D *hT_phi = new TH1D("hT_phi","",700,-3.5,3.5);

  TH1D *hS_pT = new TH1D("hS_pT","",40000,0.0,400);
  TH1D *hS_eta = new TH1D("hS_eta","",200,-1.0,1.0);
  TH1D *hS_phi = new TH1D("hS_phi","",700,-3.5,3.5);


   TProfile* fHistXsection = new TProfile("fHistXsection", "fHistXsection", 1, 0, 1);
   fHistXsection->GetYaxis()->SetTitle("xsection");

   TH1F* fHistTrials = new TH1F("fHistTrials", "fHistTrials", 1, 0, 1);
   fHistTrials->GetYaxis()->SetTitle("trials");



//___________________________________________________
//Thermal Particles Distribuitions (toy model)

  TF1* f_pT = new TF1("f_pT","x*exp(-x/0.4)", 0.0, 400.0);
  f_pT->SetNpx(40000);

  TF1* f_eta = new TF1("f_eta", "1", -1.0, 1.0);
  f_eta->SetNpx(200);

  TF1* f_phi = new TF1("f_phi", "1", (-1.0)*TMath::Pi(), TMath::Pi() );
  f_phi->SetNpx(700);
  //___________________________________________________
  cout<<"nEvents"<<nEvents<<endl;
//Begin event loop
  Int_t count=0;
  for(int i = 0; i < nEvents; i++)
    {
      cout<<"here"<<nEvents<<endl;
      double fourvec[4];

      //1st Step: Pythia + FastJet -> Probe Jet := Hardest Jet generated by Pythia.
      if(!pythia.next()) continue;
       count=count+1;
       Double_t weight=pythia.info.sigmaGen();
       fHistXsection->Fill(0.5,weight);
       fHistTrials->Fill(count);
      fjInputs1.resize(0);

      for(int j = 0; j < pythia.event.size(); j++)
	{
	  if(pythia.event[j].isFinal())
	    {
	      //Apply cuts in the particles
	      if(pythia.event[j].pT() < trackLowPtCut) continue;                 //pt cut
	      if(TMath::Abs(pythia.event[j].eta()) > trackEtaCut) continue;      //eta cut


	      fourvec[0]=pythia.event[j].px();
	      fourvec[1]=pythia.event[j].py();
	      fourvec[2]=pythia.event[j].pz();
	      fourvec[3]=pythia.event[j].e();

	      fastjet::PseudoJet PythiaParticle(fourvec);

	      fjInputs1.push_back(PythiaParticle);
	    }
	}

      //Jet Reconstruction:
      std::vector<fastjet::PseudoJet> PythiaJets;//Declaration of vector for Reconstructed Jets

      fastjet::GhostedAreaSpec ghost_spec(1, 1, 0.05);//Ghosts to calculate the Jet Area

      fastjet::AreaDefinition fAreaDef(fastjet::passive_area,ghost_spec);//Area Definition

      fastjet::ClusterSequenceArea clustSeq_Sig(fjInputs1, *jetDefAKT_Sig, fAreaDef);//Cluster Sequence

      PythiaJets = sorted_by_pt(clustSeq_Sig.inclusive_jets(1.));//Vector with the Reconstructed Jets in pT order
      //__

      if(PythiaJets.size()==0) continue;

      if(PythiaJets[0].perp()<10.0) continue;

      fastjet::PseudoJet ProbeJet = PythiaJets[0];//Hardest Pythia Jet

      //Fill "B-Histograms"

      hJetPt_B->Fill(ProbeJet.pt());

      hJetArea_B->Fill(ProbeJet.area());

      //IterativeDeclustering(ProbeJet,true,zcut);

      //2nd Step: Thermal Particles + Probe Particles + CONSTITUENTS SUBTRACTION + FastJet (again) -> New Jets
      fjInputs2.resize(0);
      Int_t nThermalParticles=r3->Gaus(naverage,sigman);
      cout<<nThermalParticles<<"particles"<<endl;
      //Thermal Particles loop
      for(int j = 0; j < nThermalParticles; j++)
	{
	  double pT = f_pT->GetRandom();

	  double eta = f_eta->GetRandom();

	  double phi = f_phi->GetRandom();

	  if(pT < trackLowPtCut) continue;//pt cut

	  hT_pT->Fill(pT);
	  hT_eta->Fill(eta);
	  hT_phi->Fill(phi);

	  h_pT->Fill(pT);
	  h_eta->Fill(eta);
	  h_phi->Fill(phi);

	  fourvec[0] = Calculate_pX( pT, eta, phi );
	  fourvec[1] = Calculate_pY( pT, eta, phi );
	  fourvec[2] = Calculate_pZ( pT, eta, phi );
	  fourvec[3] = Calculate_E( pT, eta, phi );

	  fastjet::PseudoJet ThermalParticle(fourvec);

	  ThermalParticle.set_user_index(0);

	  fjInputs2.push_back(ThermalParticle);
	}

      //Probe Particles loop

      std::vector<fastjet::PseudoJet> ProbeParticles = sorted_by_pt(ProbeJet.constituents());

      for(int j = 0; j < ProbeParticles.size(); j++)
	{
	  hP_pT->Fill(ProbeParticles[j].pt());
	  hP_eta->Fill(ProbeParticles[j].eta());
	  hP_phi->Fill(ProbeParticles[j].phi());

	  h_pT->Fill(ProbeParticles[j].pt());
	  h_eta->Fill(ProbeParticles[j].eta());
	  h_phi->Fill(ProbeParticles[j].phi());

	  ProbeParticles[j].set_user_index(1);

	  fjInputs2.push_back( ProbeParticles[j] );
	}

      //CONSTITUENTS SUBTRACTION:

      fastjet::JetMedianBackgroundEstimator bge;  //.......... Background Sutraction event by event

      fastjet::Selector BGSelector = fastjet::SelectorAbsEtaMax(1.0);

      fastjet::JetDefinition jetDefBG(fastjet::kt_algorithm, jetParameterR, recombScheme, strategy);

      fastjet::AreaDefinition fAreaDefBG(fastjet::active_area_explicit_ghosts,ghost_spec);

      fastjet::ClusterSequenceArea clustSeqBG(fjInputs2, jetDefBG, fAreaDefBG);

      std::vector <fastjet::PseudoJet> BGJets = clustSeqBG.inclusive_jets();

      bge.set_selector(BGSelector);

      bge.set_jets(BGJets);

      fastjet::contrib::ConstituentSubtractor subtractor(&bge);

      subtractor.set_common_bge_for_rho_and_rhom(true);
      // for massless input particles it does not make any difference (rho_m is always zero)

      //	  	ubtractor.set_max_standardDeltaR(jetParameterR);
      subtractor.set_max_standardDeltaR(0.25);

      //SUBTRACTION HERE:
      std::vector<fastjet::PseudoJet> fjInputs3 = subtractor.subtract_event(fjInputs2, 1.0);
      //fjInputs3 := corrected event has the particles, now I need to reconstruct the Jet again

      //Lets look to the particles before the last jet finding:
      for(int j = 0; j < fjInputs3.size(); j++)
	{
	  hS_pT->Fill(fjInputs3[j].pt());

	  hS_eta->Fill(fjInputs3[j].eta());

	  hS_phi->Fill(fjInputs3[j].phi_std());
	}

      //________________________

      //Jet Reconstruction:
      std::vector<fastjet::PseudoJet> NewJets;//Declaration of vector for Reconstructed Jets

      fastjet::GhostedAreaSpec New_ghost_spec(1, 1, 0.05);//Ghosts to calculate the Jet Area

      fastjet::AreaDefinition New_fAreaDef(fastjet::passive_area,New_ghost_spec);//Area Definition

      fastjet::ClusterSequenceArea New_clustSeq_Sig(fjInputs3, *jetDefAKT_Sig, New_fAreaDef);//Cluster Sequence

      NewJets = New_clustSeq_Sig.inclusive_jets(1.);//Vector with the Reconstructed Jets
      //_________________________________________________________________________________

      //3rd Step: MATCHING := Find in NewJets the Jet with the constituents that satisfy the condition
      //#Sigma_{constituents with index=1} p_{T}_{constituent} >= 0.5*p_{T}_{ProbeJet}
      //Fill the tree with pseudodata


      int MATCH = -1;

      for(int j = 0; j < NewJets.size(); j++)
	{
	  bool ifMatch = false;
	  fastjet::PseudoJet NJet = NewJets.at(j);

	  if(!EtaCut(NJet, etamin_Sig, etamax_Sig)) continue;
	  if(NJet.pt()   < 0.5*ProbeJet.pt()) continue;

	  double MATCH_pT = 0.0;

	  std::vector<fastjet::PseudoJet> constituents = sorted_by_pt(NJet.constituents());



	  for(int k = 0; k < constituents.size(); k++)
	    if( constituents.at(k).user_index() == 1 )
	      MATCH_pT += constituents.at(k).pt();

	  if( MATCH_pT >= 0.5*ProbeJet.pt() )
	    {
	      MATCH = j;
	      ifMatch = true;
	      //break;
	    }

	 IterativeDeclustering(NewJets.at(j),zcut,ProbeJet,MATCH);

          fTreeResponse->Fill();
	}


      if(MATCH > 0)
	{
	  fastjet::PseudoJet MatchedJet = NewJets.at(MATCH);

	  hJetPt_A->Fill(MatchedJet.pt());

	  hJetArea_A->Fill(MatchedJet.area());
	}
    }
//End event loop

  TFile* outFile = new TFile(Form("Output_thermal_num%d_z%f_ptmin%2f_ptmax%2f.root",cislo,zcut,pthatmin,pthatmax),"RECREATE");

  outFile->cd();

  hJetPt_A->Write();
  hJetArea_A->Write();
  hJet_deltaR_A->Write();
  hJet_deltaRg_A->Write();
  hJet_deltaRg_pT_A->Write();

  hJetPt_B->Write();
  hJetArea_B->Write();
  hJet_deltaR_B->Write();
  hJet_deltaRg_B->Write();
  hJet_deltaRg_pT_B->Write();
  fHistXsection->Write();
    fHistTrials->Write();

  h_pT->Write();
  h_eta->Write();
  h_phi->Write();

  hP_pT->Write();
  hP_eta->Write();
  hP_phi->Write();

  hT_pT->Write();
  hT_eta->Write();
  hT_phi->Write();

  hS_pT->Write();
  hS_eta->Write();
  hS_phi->Write();

  fTreeResponse->Write();

  outFile->Close();

  return 0;
}
