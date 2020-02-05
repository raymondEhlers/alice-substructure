// main01.cc is a part of the PYTHIA event generator.
// Copyright (C) 2014 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This is a simple test program. It fits on one slide in a talk.
// It studies the charged multiplicity distribution at the LHC.

#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdio> // needed for io
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iostream> // needed for io
#include <sstream>
#include <string>
#include <valarray>
#include <vector>

#include "Pythia8/Pythia.h"
#include "TClonesArray.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "THnSparse.h"
#include "TList.h"
#include "TMath.h"
#include "TNtuple.h"
#include "TProfile.h"
#include "TRandom3.h"
#include "TString.h"
#include "TTree.h"
#include "TVector3.h"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/ClusterSequenceAreaBase.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/Selector.hh"
#include "fastjet/contrib/ConstituentSubtractor.hh"
#include "fastjet/contrib/ModifiedMassDropTagger.hh"
#include "fastjet/contrib/Nsubjettiness.hh"
#include "fastjet/contrib/Recluster.hh"
#include "fastjet/contrib/SoftDrop.hh"
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"
#include "fastjet/tools/Subtractor.hh"

using namespace Pythia8;

#define nThermalParticles 4000

double Calculate_pX(double pT, double eta, double phi) { return (pT * TMath::Cos(phi)); }

double Calculate_pY(double pT, double eta, double phi) { return (pT * TMath::Sin(phi)); }

double Calculate_pZ(double pT, double eta, double phi) { return (pT * TMath::SinH(eta)); }

double Calculate_E(double pT, double eta, double phi)
{
  double pZ = Calculate_pZ(pT, eta, phi);

  return (TMath::Sqrt(pT * pT + pZ * pZ));
}

THnSparse* fHLundIterative;
THnSparse* fHInfo;

bool EtaCut(fastjet::PseudoJet fjJet, double etaMin, double etaMax)
{
  if (fjJet.eta() > etaMax || fjJet.eta() < etaMin) {
    return false;
  } else {
    return true;
  }
}

void ExtractWMass(vector<fastjet::PseudoJet> jet, Int_t type, double etmin, double etmax)
{
  double zg = 0.;
  double njetiness_kt = 0.;
  double sdmass = 0;
  double angle = 0;
  double sdmass2 = 0;

  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, 1, static_cast<fastjet::RecombinationScheme>(0),
                  fastjet::Best);
  for (unsigned int ijet = 0; ijet < jet.size(); ijet++) {
    fastjet::PseudoJet fjJet;
    vector<fastjet::PseudoJet> constit = sorted_by_pt(jet[ijet].constituents());
    if (constit.size() == 0)
      continue;

    fastjet::ClusterSequence cs_gen(constit, jet_def);
    std::vector<fastjet::PseudoJet> output_jets = sorted_by_pt(cs_gen.inclusive_jets(0));
    fjJet = output_jets[0];

    if (!EtaCut(fjJet, etmin, etmax))
      continue;
    if (fjJet.pt() < 0.150)
      continue;
    // fastjet::contrib::Recluster recluster(fastjet::cambridge_aachen_algorithm, 1, true);
    fastjet::contrib::SoftDrop softdrop(0., 0.4);
    // softdrop.set_reclustering(true,&recluster);
    softdrop.set_verbose_structure(kTRUE);
    fastjet::PseudoJet finaljet = softdrop(fjJet);
    zg = finaljet.structure_of<fastjet::contrib::SoftDrop>().symmetry();
    fastjet::contrib::NsubjettinessRatio nSub21_beta2(2, 1, fastjet::contrib::KT_Axes(),
                             fastjet::contrib::NormalizedMeasure(1., 0.4));
    njetiness_kt = nSub21_beta2(finaljet);
    sdmass = finaljet.m();
    sdmass2 = fjJet.m();
    angle = finaljet.structure_of<fastjet::contrib::SoftDrop>().delta_R();
    double var = TMath::Log(sdmass * sdmass / (fjJet.pt() * fjJet.pt()));

    double lundEntries[7] = { finaljet.perp(), (1 - zg) * finaljet.perp(), angle, njetiness_kt, sdmass, var, (double) type };
    fHInfo->Fill(lundEntries);
  }

  return;
}

void ExtractWMassDijet(vector<fastjet::PseudoJet> jet, Int_t type, double etmin, double etmax)
{
  double zg = 0.;
  double njetiness_kt = 0.;
  double sdmass = 0;
  double angle = 0;
  double sdmass2 = 0;

  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, 1, static_cast<fastjet::RecombinationScheme>(0),
                  fastjet::Best);

  for (unsigned int ijet = 0; ijet < jet.size(); ijet++) {
    fastjet::PseudoJet fjJet;
    fastjet::PseudoJet fjJet2;
    vector<fastjet::PseudoJet> constit = sorted_by_pt(jet[ijet].constituents());
    if (constit.size() == 0)
      continue;
    for (unsigned int ijet2 = 0; ijet2 < jet.size(); ijet2++) {
      if (ijet == ijet2)
        continue;
      vector<fastjet::PseudoJet> constit2 = sorted_by_pt(jet[ijet2].constituents());
      if (constit2.size() == 0)
        continue;

      fastjet::ClusterSequence cs_gen(constit, jet_def);
      std::vector<fastjet::PseudoJet> output_jets = sorted_by_pt(cs_gen.inclusive_jets(0));
      fjJet = output_jets[0];

      fastjet::ClusterSequence cs_gen2(constit2, jet_def);
      std::vector<fastjet::PseudoJet> output_jets2 = sorted_by_pt(cs_gen2.inclusive_jets(0));
      fjJet2 = output_jets2[0];

      if (!EtaCut(fjJet, etmin, etmax))
        continue;
      if (fjJet.pt() < 0.150)
        continue;

      if (!EtaCut(fjJet2, etmin, etmax))
        continue;
      if (fjJet2.pt() < 0.150)
        continue;

      fastjet::contrib::SoftDrop softdrop(0., 0.1);

      softdrop.set_verbose_structure(kTRUE);
      fastjet::PseudoJet finaljet1 = softdrop(fjJet);
      fastjet::PseudoJet finaljet2 = softdrop(fjJet2);

      if (TMath::Abs(finaljet1.delta_R(finaljet2)) > TMath::Pi())
        continue;

      double invmass=2*finaljet1.perp()*finaljet2.perp()*(cosh(finaljet1.eta()-finaljet2.eta())-cos(finaljet1.phi()-finaljet2.phi()));
    }
  }

  return;
}

//_________________________________________________________________________
Double_t RelativePhi(Double_t mphi, Double_t vphi)
{
  // Get relative azimuthal angle of two particles -pi to pi
  if (vphi < -TMath::Pi())
    vphi += TMath::TwoPi();
  else if (vphi > TMath::Pi())
    vphi -= TMath::TwoPi();

  if (mphi < -TMath::Pi())
    mphi += TMath::TwoPi();
  else if (mphi > TMath::Pi())
    mphi -= TMath::TwoPi();

  Double_t dphi = mphi - vphi;
  if (dphi < -TMath::Pi())
    dphi += TMath::TwoPi();
  else if (dphi > TMath::Pi())
    dphi -= TMath::TwoPi();

  return dphi; // dphi in [-Pi, Pi]
}

//___________________________________________________________________

int main(int argc, char* argv[])
{
  Int_t cislo = -1;  // unique number for each file
  Int_t tune = -1;   // pythia tune
  Int_t charged = 0; // full or track-based jets
  Int_t unev = 1;    // underlying event (ISR+MPI)

  if (argc != 6) {
    cout << "Usage:" << endl << "./pygen <PythiaTune> <Number> <nEvts> <unev> <jetR>" << endl;
    return 0;
  }
  tune = atoi(argv[1]);
  cislo = atoi(argv[2]);
  unev = atoi(argv[4]);

  Int_t nEvent = atoi(argv[3]); //(Int_t) 1e3 + 1.0;
  TString name;
  //__________________________________________________________________________
  //                        ANALYSIS SETTINGS

  double jetParameterR = (double)atof(argv[5]); // jet R
  double trackLowPtCut = 0.;                    // GeV
  double trackEtaCut = 1;
  Float_t ptHatMin = 200;
  Float_t ptHatMax = 13000;

  //__________________________________________________________________________
  //                        PYTHIA SETTINGS

  // Generator. Process selection. LHC initialization. Histogram.
  Pythia pythia;
  pythia.readString("Beams:idA = 2212"); // beam 1 proton
  pythia.readString("Beams:idB = 2212"); // beam 2 proton
  pythia.readString("Beams:eCM = 13000.");
  pythia.readString(
   Form("Tune:pp = %d", tune)); // tune 1-13    5=defaulr TUNE4C,  6=Tune 4Cx, 7=ATLAS MB Tune A2-CTEQ6L1

  pythia.readString("Random:setSeed = on");
  pythia.readString(Form("Random:seed = %d", cislo));
  // pythia
  pythia.readString("WeakDoubleBoson:ffbar2WW=on");
  // pythia.readString("24:mMin = 100");
  pythia.readString("24:onIfAny = 1 2 3 4 5");
  //# Force production to higher pT
  pythia.readString("PhaseSpace:pTHatMin = 200");
  if (ptHatMin < 0 || ptHatMax < 0) {
    pythia.readString("PhaseSpace:pTHatMin = 0."); // <<<<<<<<<<<<<<<<<<<<<<<
  } else {
    name = Form("PhaseSpace:pTHatMin = %f", (Float_t)ptHatMin);
    pythia.readString(name.Data());
    name = Form("PhaseSpace:pTHatMax = %f", (Float_t)ptHatMax);
    pythia.readString(name.Data());
  }

  if (unev == 0) {
    pythia.readString("PartonLevel:MPI = off");
    pythia.readString("PartonLevel:ISR = off");
  }

  pythia.readString("310:mayDecay  = off"); // K0s
  pythia.readString("3122:mayDecay = off"); // labda0
  pythia.readString("3112:mayDecay = off"); // sigma-
  pythia.readString("3212:mayDecay = off"); // sigma0
  pythia.readString("3222:mayDecay = off"); // sigma+
  pythia.readString("3312:mayDecay = off"); // xi-
  pythia.readString("3322:mayDecay = off"); // xi+
  pythia.readString("3334:mayDecay = off"); // omega-

  pythia.init();

  //___________________________________________________
  //                      FASTJET  SETTINGS

  double etamin_Sig = -trackEtaCut + jetParameterR; // signal jet eta range
  double etamax_Sig = -etamin_Sig;
  fastjet::Strategy strategy = fastjet::Best;
  fastjet::RecombinationScheme recombScheme = fastjet::E_scheme;
  fastjet::JetDefinition* jetDefAKT_Sig = NULL;
  jetDefAKT_Sig = new fastjet::JetDefinition(fastjet::antikt_algorithm, jetParameterR, recombScheme, strategy);

  fastjet::GhostedAreaSpec ghostareaspec(trackEtaCut, 1, 0.05); // ghost
  // max rap, repeat, ghostarea default 0.01
  fastjet::AreaType areaType = fastjet::active_area_explicit_ghosts;
  fastjet::AreaDefinition* areaDef = new fastjet::AreaDefinition(areaType, ghostareaspec);

  // Fastjet inputs
  std::vector<fastjet::PseudoJet> fjInputs;
  std::vector<fastjet::PseudoJet> fjInputs2;

  //_________Thermal Particl_densityes Distribuitions (toy model)

  TH1D* hT_phi = new TH1D("hT_phi", "", 700, -3.5, 3.5);

  TF1* f_pT = new TF1("f_pT", "x*exp(-x/0.3)", 0.0, 400.0);
  f_pT->SetNpx(40000);

  TF1* f_eta = new TF1("f_eta", "1", -1.0, 1.0);
  f_eta->SetNpx(200);

  TF1* f_phi = new TF1("f_phi", "1", (-1.0) * TMath::Pi(), TMath::Pi());
  f_phi->SetNpx(700);

  //___________________________________________________________

  TH1D* histoWMass = new TH1D("histoWMass", "histoWMass", 100, 0.0, 200.0);
  histoWMass->Sumw2();

  TH1D* histoWMassNsub = new TH1D("histoWMassNsub", "histoWMassNsub", 100, 0.0, 200.0);
  histoWMassNsub->Sumw2();

  TH1D* histoAngle = new TH1D("histoAngle", "histoAngle", 100, 0.0, 6.4);
  histoAngle->Sumw2();

  TH2D* histoTest = new TH2D("histoTest", "histoTest", 100, -20, 0, 100, 0, 1);
  histoTest->Sumw2();

  TProfile* fHistXsection = new TProfile("fHistXsection", "fHistXsection", 1, 0, 1);
  fHistXsection->GetYaxis()->SetTitle("xsection");

  TH1F* fHistTrials = new TH1F("fHistTrials", "fHistTrials", 1, 0, 1);
  fHistTrials->GetYaxis()->SetTitle("trials");

  // THnSparse *fHInfo;
  // ptlead,ptsublead,angle,tau2/tau1,mass,log(mass^2/pt^2)
  const Int_t dimSpec = 7;
  const Int_t nBinsSpec[7] = { 200, 200, 100, 20, 200, 20, 2 };
  const Double_t lowBinSpec[7] = { 0, 0, 0, 0, 0, -20, 0 };
  const Double_t hiBinSpec[7] = { 1000, 1000, 6.4, 1.2, 200, 0, 2 };
  fHInfo = new THnSparseF("fHInfo", "fHInfo[jetpt,tform,erad]", dimSpec, nBinsSpec, lowBinSpec, hiBinSpec);

  //___________________________________________________
  // Begin event loop. Generate event. Skip if error. List first one.
  for (int iEvent = 0; iEvent < nEvent; iEvent++) {
    if (!pythia.next())
      continue;

    fjInputs.resize(0);
    fjInputs2.resize(0);
    Double_t index = 0;
    Double_t fourvec[4];
    for (Int_t i = 0; i < pythia.event.size(); ++i) {
      if (pythia.event[i].isFinal()) {
        if (charged == 1)
          if (!pythia.event[i].isCharged())
            continue; // only charged particles
        if (pythia.event[i].pT() < trackLowPtCut)
          continue; // pt cut
        if (TMath::Abs(pythia.event[i].eta()) > trackEtaCut)
          continue; // eta cut
        index = 0;
        fourvec[0] = pythia.event[i].px();
        fourvec[1] = pythia.event[i].py();
        fourvec[2] = pythia.event[i].pz();
        fourvec[3] = pythia.event[i].pAbs();
        fastjet::PseudoJet particle(fourvec);
        particle.set_user_index(index);
        fjInputs.push_back(particle);
        fjInputs2.push_back(particle);
      }
    }
    // Thermal Particles loop
    for (int j = 0; j < nThermalParticles; j++) {
      double pT = f_pT->GetRandom();
      double eta = f_eta->GetRandom();
      double phi = f_phi->GetRandom();
      if (pT < trackLowPtCut)
        continue; // pt cut
      fourvec[0] = Calculate_pX(pT, eta, phi);
      fourvec[1] = Calculate_pY(pT, eta, phi);
      fourvec[2] = Calculate_pZ(pT, eta, phi);
      fourvec[3] = Calculate_E(pT, eta, phi);
      fastjet::PseudoJet ThermalParticle(fourvec);
      ThermalParticle.set_user_index(1);
      fjInputs2.push_back(ThermalParticle);
    }

    //________________signal jets____________________________________________________
    vector<fastjet::PseudoJet> inclusiveJets_Sig;
    fastjet::ClusterSequenceArea clustSeq_Sig(fjInputs, *jetDefAKT_Sig, *areaDef);
    inclusiveJets_Sig = clustSeq_Sig.inclusive_jets(30.);
    ExtractWMass(inclusiveJets_Sig, 0, etamin_Sig, etamax_Sig);
    ExtractWMassDijet(inclusiveJets_Sig, 0, etamin_Sig, etamax_Sig);

    //_________________HI jets_______________________________________________________
    std::vector<fastjet::PseudoJet> NewJets;             // Declaration of vector for Reconstructed Jets
    fastjet::GhostedAreaSpec New_ghost_spec(1, 1, 0.05); // Ghosts to calculate the Jet Area
    fastjet::AreaDefinition New_fAreaDef(fastjet::active_area_explicit_ghosts, New_ghost_spec); // Area Definition
    fastjet::ClusterSequenceArea New_clustSeq_Sig(fjInputs2, *jetDefAKT_Sig, New_fAreaDef);     // Cluster Sequence
    NewJets = New_clustSeq_Sig.inclusive_jets(1.); // Vector with the Reconstructed Jets

    fastjet::JetMedianBackgroundEstimator bge;
    fastjet::Selector BGSelector = fastjet::SelectorAbsEtaMax(1.0);
    fastjet::JetDefinition jetDefBG(fastjet::kt_algorithm, jetParameterR, recombScheme, strategy);
    fastjet::AreaDefinition fAreaDefBG(fastjet::active_area_explicit_ghosts, New_ghost_spec);
    fastjet::ClusterSequenceArea clustSeqBG(fjInputs2, jetDefBG, fAreaDefBG);
    std::vector<fastjet::PseudoJet> BGJets = clustSeqBG.inclusive_jets();
    bge.set_selector(BGSelector);
    bge.set_jets(BGJets);
    fastjet::contrib::ConstituentSubtractor subtractor(&bge);
    //
    subtractor.set_common_bge_for_rho_and_rhom(true);
    subtractor.set_max_standardDeltaR(jetParameterR);
    // subtractor.set_alpha(0.5);

    for (int j = 0; j < NewJets.size(); j++) {
      const fastjet::PseudoJet& jet = NewJets[j];
      fastjet::PseudoJet subtracted_Jet = subtractor(jet);
      NewJets[j] = subtracted_Jet;
    }

    ExtractWMass(NewJets, 1, etamin_Sig, etamax_Sig);
    ExtractWMassDijet(NewJets, 1, etamin_Sig, etamax_Sig);

  } // end of event

  //____________________________________________________
  //          SAVE OUTPUT

  TString tag = Form("PP_2W_ANTIKT%02d", TMath::Nint(jetParameterR * 10));

  TFile* outFile =
   new TFile(Form("%s_tune%d_c%d_charged%d_unev%d.root", tag.Data(), tune, cislo, charged, unev), "RECREATE");

  outFile->cd();
  // fTreeObservables->Write();
  histoWMass->Write();
  histoWMassNsub->Write();
  histoTest->Write();
  histoAngle->Write();
  fHInfo->Write();
  outFile->Close();

  pythia.stat();
  return 0;
}

//_________________________________________________________________________
//_________________________________________________________________________
//_________________________________________________________________________
//_________________________________________________________________________

