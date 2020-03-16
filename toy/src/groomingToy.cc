// Adapted from main01.cc from PYTHIA and from code from Leticia.
// Copyright (C) 2014 Torbjorn Sjostrand.

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

#include "output.h"

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

/*
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
}*/

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

std::map<int, int> PerformGeometricalMatching(std::vector<fastjet::PseudoJet> & outerJets, std::vector<fastjet::PseudoJet> & innerJets)
{
  std::map<int, int> indexMap;
  for (std::size_t outerIndex = 0; outerIndex < outerJets.size(); outerIndex++) {
    fastjet::PseudoJet & outerJet = outerJets[outerIndex];
    // Basic acceptance cuts
    if (outerJet.pt() < 0.1) {
      continue;
    }
    double distance = 1000;
    for (std::size_t innerIndex = 0; innerIndex < innerJets.size(); innerIndex++)
    {
      fastjet::PseudoJet & innerJet = innerJets[innerIndex];
      if (innerJet.pt() < 0.1) {
        continue;
      }
      double deltaR = outerJet.delta_R(innerJet);
      if (deltaR < distance) {
        distance = deltaR;
        indexMap[outerIndex] = innerIndex;
      }
    }
  }
  if (innerJets.size() == 0) {
    std::cout << "indexMap size: " << indexMap.size() << "\n";
  }

  return indexMap;
}

std::tuple<std::map<int, int>, std::map<int, int>> MatchJets(std::vector<fastjet::PseudoJet> & hybridJets, std::vector<fastjet::PseudoJet> & trueJets)
{
  std::map<int, int> trueToHybridIndex = PerformGeometricalMatching(trueJets, hybridJets);
  std::map<int, int> hybridToTrueIndex = PerformGeometricalMatching(hybridJets, trueJets);

  std::cout << "trueToHybridIndex.size(): " << trueToHybridIndex.size() << "\n";

  // Determine matches where one points at the other and vice versa.
  std::map<int, int> trueToHybridIndexVerified;
  std::map<int, int> hybridToTrueIndexVerified;
  for (const auto & hybridIndex : hybridToTrueIndex) {
    if (trueToHybridIndex[hybridIndex.second] == hybridIndex.first) {
      // We have a true match!
      hybridToTrueIndexVerified[hybridIndex.first] = hybridIndex.second;
      trueToHybridIndexVerified[hybridIndex.second] = hybridIndex.first;
    }
    else {
      // Sentinel index to signify no matching
      hybridToTrueIndexVerified[hybridIndex.first] = -1;
      trueToHybridIndexVerified[hybridIndex.second] = -1;
    }
  }

  std::cout << "trueToHybridIndexVerified.size(): " << trueToHybridIndexVerified.size() << "\n";
  std::cout << "hybridToTrueIndexVerified.size(): " << hybridToTrueIndexVerified.size() << "\n";

  return std::make_tuple(trueToHybridIndexVerified, hybridToTrueIndexVerified);
}

double SharedMomentumFraction(fastjet::PseudoJet & hybridJet, fastjet::PseudoJet & trueJet)
{
  double constituentsPt = 0;
  for (const auto & trueConstituent : trueJet.constituents())
  {
    for (const auto & hybridConstituent : hybridJet.constituents())
    {
      // Perform matching based solely on constituent global index.
      if (trueConstituent.user_index() == hybridConstituent.user_index()) {
        constituentsPt += trueConstituent.pt();
      }
    }
  }
  return constituentsPt / trueJet.pt();
}

void ExtractJetSplittings(SubstructureTree::JetSubstructureSplittings & jetSplittings, fastjet::PseudoJet & inputJet, int splittingNodeIndex, bool followingIterativeSplitting, const bool storeRecursiveSplittings = true)
{
  fastjet::PseudoJet j1;
  fastjet::PseudoJet j2;
  if (inputJet.has_parents(j1, j2) == false) {
    // No parents, so we're done - just return.
    return;
  }

  // j1 should always be the harder of the two subjets.
  if (j1.perp() < j2.perp()) {
    swap(j1, j2);
  }

  // We have a splitting. Record the properties.
  double z = j2.perp() / (j2.perp() + j1.perp());
  double delta_R = j1.delta_R(j2);
  double xkt = j2.perp() * sin(delta_R);
  // Add the splitting node.
  jetSplittings.AddSplitting(xkt, delta_R, z, splittingNodeIndex);
  // Increment after storing splitting because the parent is the new one.
  //splittingNodeIndex++;
  // -1 because we want to index the parent splitting that was just stored.
  splittingNodeIndex = jetSplittings.GetNumberOfSplittings() - 1;
  // Store the subjets
  std::vector<unsigned short> j1ConstituentIndices, j2ConstituentIndices;
  for (auto constituent: j1.constituents()) {
    j1ConstituentIndices.emplace_back(constituent.user_index());
  }
  for (auto constituent: j2.constituents()) {
    j2ConstituentIndices.emplace_back(constituent.user_index());
  }
  jetSplittings.AddSubjet(splittingNodeIndex, followingIterativeSplitting, j1ConstituentIndices);
  jetSplittings.AddSubjet(splittingNodeIndex, false, j2ConstituentIndices);

  // Recurse as necessary to get the rest of the splittings.
  ExtractJetSplittings(jetSplittings, j1, splittingNodeIndex, followingIterativeSplitting);
  if (storeRecursiveSplittings == true) {
    ExtractJetSplittings(jetSplittings, j2, splittingNodeIndex, false);
  }
}

void Reclustering(SubstructureTree::JetSubstructureSplittings & jetSplittings, fastjet::PseudoJet & jet,const bool storeRecursiveSplittings = true, const bool isData = false)
{
  // Grab the jet constituents from the jet.
  // They will be storedwith the splitting, and also used for the declustering.
  std::vector<fastjet::PseudoJet> inputVectors;
  fastjet::PseudoJet pseudoTrack;
  unsigned int constituentIndex = 0;
  for (const auto & part : jet.constituents()) {
    //if (isData == true && fDoTwoTrack == kTRUE && CheckClosePartner(jet, part))
    //    continue;
    pseudoTrack.reset(part.px(), part.py(), part.pz(), part.e());
    pseudoTrack.set_user_index(constituentIndex);
    inputVectors.push_back(pseudoTrack);

    // Also store the jet constituents in the output
    jetSplittings.AddJetConstituent(part);

    // Keep track of the number of constituents.
    constituentIndex++;
  }

  try {
    fastjet::JetAlgorithm jetalgo(fastjet::cambridge_algorithm);
    fastjet::JetDefinition jetDef(jetalgo, 1., fastjet::RecombinationScheme::E_scheme, fastjet::BestFJ30);
    // For area calculation (when desired)
    fastjet::GhostedAreaSpec ghost_spec(1, 1, 0.05);
    fastjet::AreaDefinition areaDef(fastjet::passive_area, ghost_spec);
    // We use a pointer for the CS because it has to stay in scope while we explore the splitting history.
    fastjet::ClusterSequence * cs = nullptr;
    if (isData) {
      cs = new fastjet::ClusterSequenceArea(inputVectors, jetDef, areaDef);
    }
    else {
      cs = new fastjet::ClusterSequence(inputVectors, jetDef);
    }
    //std::cout << "About to get jets\n";
    std::vector<fastjet::PseudoJet> outputJets = cs->inclusive_jets(0);
    //std::cout << "Output jets size = " << outputJets.size() << "\n";

    if (outputJets.size() == 0) {
      std::cout << "Not output jets in reclustering! Returning.\n";
      return;
    }

    fastjet::PseudoJet jj;
    jj = outputJets[0];

    // Store the jet splittings.
    int splittingNodeIndex = -1;
    ExtractJetSplittings(jetSplittings, jj, splittingNodeIndex, true, storeRecursiveSplittings);

    // Cleanup the allocated cluster sequence.
    delete cs;
  } catch (const fastjet::Error&) {
    std::cerr << " [w] FJ Exception caught.\n";
  }
}

//___________________________________________________________________

int main(int argc, char* argv[])
{
  Int_t randomSeed = -1;  // unique number for each file
  Int_t tune = -1;   // pythia tune
  Int_t charged = 0; // full or track-based jets
  Int_t underlingEvent = 1;    // underlying event (ISR+MPI)

  if (argc != 6) {
    cout << "Usage:" << endl << "./pygen <PythiaTune> <Number> <nEvts> <underlingEvent> <jetR>" << endl;
    return 0;
  }
  tune = atoi(argv[1]);
  randomSeed = atoi(argv[2]);
  underlingEvent = atoi(argv[4]);

  Int_t nEvent = atoi(argv[3]); //(Int_t) 1e3 + 1.0;
  TString name;
  //__________________________________________________________________________
  //                        ANALYSIS SETTINGS

  double jetParameterR = (double)atof(argv[5]); // jet R
  double trackLowPtCut = 0.;                    // GeV
  double trackEtaCut = 1;
  Float_t ptHatMin = 20;
  Float_t ptHatMax = 200;

  //__________________________________________________________________________
  //                        PYTHIA SETTINGS

  // Generator. Process selection. LHC initialization. Histogram.
  Pythia pythia;
  pythia.readString("Beams:idA = 2212"); // beam 1 proton
  pythia.readString("Beams:idB = 2212"); // beam 2 proton
  pythia.readString("Beams:eCM = 5020.");
  pythia.readString(
   Form("Tune:pp = %d", tune)); // tune 1-13    5=default TUNE4C,  6=Tune 4Cx, 7=ATLAS MB Tune A2-CTEQ6L1

  pythia.readString("Random:setSeed = on");
  pythia.readString(Form("Random:seed = %d", randomSeed));
  // Turn on QCD.
  pythia.readString("HardQCD:all = on");
  // pythia
  //pythia.readString("WeakDoubleBoson:ffbar2WW=on");
  // pythia.readString("24:mMin = 100");
  //pythia.readString("24:onIfAny = 1 2 3 4 5");
  //# Force production to higher pT
  pythia.readString("PhaseSpace:pTHatMin = 40");
  if (ptHatMin < 0 || ptHatMax < 0) {
    pythia.readString("PhaseSpace:pTHatMin = 0."); // <<<<<<<<<<<<<<<<<<<<<<<
  } else {
    name = Form("PhaseSpace:pTHatMin = %f", (Float_t)ptHatMin);
    pythia.readString(name.Data());
    name = Form("PhaseSpace:pTHatMax = %f", (Float_t)ptHatMax);
    pythia.readString(name.Data());
  }

  if (underlingEvent == 0) {
    pythia.readString("PartonLevel:MPI = off");
    pythia.readString("PartonLevel:ISR = off");
  }

  //pythia.readString("310:mayDecay  = off"); // K0s
  //pythia.readString("3122:mayDecay = off"); // labda0
  //pythia.readString("3112:mayDecay = off"); // sigma-
  //pythia.readString("3212:mayDecay = off"); // sigma0
  //pythia.readString("3222:mayDecay = off"); // sigma+
  //pythia.readString("3312:mayDecay = off"); // xi-
  //pythia.readString("3322:mayDecay = off"); // xi+
  //pythia.readString("3334:mayDecay = off"); // omega-

  pythia.init();

  //___________________________________________________
  //                      FASTJET  SETTINGS

  //double etamin_Sig = -trackEtaCut + jetParameterR; // signal jet eta range
  //double etamax_Sig = -etamin_Sig;
  fastjet::Strategy strategy = fastjet::Best;
  fastjet::RecombinationScheme recombScheme = fastjet::E_scheme;
  fastjet::JetDefinition* jetDefAKT_Sig = NULL;
  jetDefAKT_Sig = new fastjet::JetDefinition(fastjet::antikt_algorithm, jetParameterR, recombScheme, strategy);

  fastjet::GhostedAreaSpec ghostareaspec(trackEtaCut, 1, 0.05); // ghost
  // max rap, repeat, ghostarea default 0.01
  fastjet::AreaType areaType = fastjet::active_area_explicit_ghosts;
  fastjet::AreaDefinition* areaDef = new fastjet::AreaDefinition(areaType, ghostareaspec);

  // Fastjet inputs
  std::vector<fastjet::PseudoJet> inputsTrue;
  std::vector<fastjet::PseudoJet> inputsHybrid;

  //_________Thermal Particl_densityes Distribuitions (toy model)

  //TH1D* hT_phi = new TH1D("hT_phi", "", 700, -3.5, 3.5);

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

  // Define output objects.
  int splitLevel = 4;
  int bufferSize = 32000;
  SubstructureTree::JetSubstructureSplittings hybridJetSplittings;
  SubstructureTree::JetSubstructureSplittings trueJetSplittings;
  TTree tree("tree", "tree");
  // data will contain the hybrid jets
  tree.Branch("data.", &hybridJetSplittings, bufferSize, splitLevel);
  // matched will contain the true jets
  tree.Branch("matched.", &trueJetSplittings, bufferSize, splitLevel);

  //___________________________________________________
  // Begin event loop. Generate event. Skip if error. List first one.
  for (int iEvent = 0; iEvent < nEvent; iEvent++) {
    // TEMP
    std::cout << "Event " << iEvent << "\n";
    // ENDTEMP
    if (iEvent % 100 == 0) {
      std::cout << "Event " << iEvent << "\n";
    }
    if (!pythia.next()) {
      continue;
    }

    inputsTrue.clear();
    inputsHybrid.clear();
    inputsTrue.resize(0);
    inputsHybrid.resize(0);
    Double_t globalIndex = 0;
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
        fourvec[0] = pythia.event[i].px();
        fourvec[1] = pythia.event[i].py();
        fourvec[2] = pythia.event[i].pz();
        fourvec[3] = pythia.event[i].pAbs();
        fastjet::PseudoJet particle(fourvec);
        particle.set_user_index(globalIndex);
        inputsTrue.push_back(particle);
        inputsHybrid.push_back(particle);
        ++globalIndex;
      }
    }
    // Add an offset for thermal particles.
    globalIndex = 100000;
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
      ThermalParticle.set_user_index(globalIndex);
      inputsHybrid.push_back(ThermalParticle);
      ++globalIndex;
    }

    //________________signal jets____________________________________________________
    fastjet::ClusterSequenceArea clustSeq_Sig(inputsTrue, *jetDefAKT_Sig, *areaDef);
    std::vector<fastjet::PseudoJet> trueJets = clustSeq_Sig.inclusive_jets(30.);
    //ExtractWMass(trueJets, 0, etamin_Sig, etamax_Sig);
    //ExtractWMassDijet(trueJets, 0, etamin_Sig, etamax_Sig);

    //_________________HI jets_______________________________________________________
    fastjet::GhostedAreaSpec New_ghost_spec(1, 1, 0.05); // Ghosts to calculate the Jet Area
    fastjet::AreaDefinition New_fAreaDef(fastjet::active_area_explicit_ghosts, New_ghost_spec); // Area Definition
    fastjet::ClusterSequenceArea New_clustSeq_Sig(inputsHybrid, *jetDefAKT_Sig, New_fAreaDef);     // Cluster Sequence
    std::vector<fastjet::PseudoJet> hybridJets = New_clustSeq_Sig.inclusive_jets(1.); // Vector with the Reconstructed Jets

    fastjet::JetMedianBackgroundEstimator bge;
    fastjet::Selector BGSelector = fastjet::SelectorAbsEtaMax(1.0);
    fastjet::JetDefinition jetDefBG(fastjet::kt_algorithm, jetParameterR, recombScheme, strategy);
    fastjet::AreaDefinition fAreaDefBG(fastjet::active_area_explicit_ghosts, New_ghost_spec);
    fastjet::ClusterSequenceArea clustSeqBG(inputsHybrid, jetDefBG, fAreaDefBG);
    std::vector<fastjet::PseudoJet> BGJets = clustSeqBG.inclusive_jets();
    bge.set_selector(BGSelector);
    bge.set_jets(BGJets);
    fastjet::contrib::ConstituentSubtractor subtractor(&bge);
    //
    subtractor.set_common_bge_for_rho_and_rhom(true);
    subtractor.set_max_standardDeltaR(jetParameterR);
    // subtractor.set_alpha(0.5);

    for (int j = 0; j < hybridJets.size(); j++) {
      const fastjet::PseudoJet& jet = hybridJets[j];
      fastjet::PseudoJet subtracted_Jet = subtractor(jet);
      hybridJets[j] = subtracted_Jet;
    }

    // Match jets
    // Need to do rudimentary matching
    std::map<int, int> trueToHybridIndex;
    std::map<int, int> hybridToTrueIndex;
    std::tie(trueToHybridIndex, hybridToTrueIndex) = MatchJets(hybridJets, trueJets);

    // Extract the splittings for each set of matched jets.
    //std::cout << "About to recluster event " << iEvent << "\n";
    bool storeRecursiveSplittings = true;
    bool applyTwoParticleAcceptanceCut = false;
    for (std::size_t hybridIndex = 0; hybridIndex < hybridJets.size(); hybridIndex++) {
      if (hybridToTrueIndex.count(hybridIndex) == 0) {
        // This jet doesn't have a match. Skip it.
        std::cout << "No match for this hybrid jet. Skipping\n";
        continue;
      }
      // Hybrid jet
      fastjet::PseudoJet & hybridJet = hybridJets[hybridIndex];
      if (hybridJet.pt() < 0.1) {
        std::cout << "Hybrid jet pt too low.\n";
        continue;
      }
      hybridJetSplittings.SetJetPt(hybridJet.pt());
      Reclustering(hybridJetSplittings, hybridJet, storeRecursiveSplittings, applyTwoParticleAcceptanceCut);
      // True jet
      std::cout << "hybridToTrueIndex.size(): " << hybridToTrueIndex.size() << "\n";
      int trueJetIndex = hybridToTrueIndex[hybridIndex];
      if (trueJetIndex == -1) {
        // No match - continue.
        continue;
      }
      fastjet::PseudoJet & trueJet = trueJets[trueJetIndex];
      std::cout << "hybridJet pt: " << hybridJet.pt() << "\n";
      std::cout << "trueJet address: " << &trueJet << "\n";
      std::cout << "trueJetIndex: " << trueJetIndex << "\n";
      std::cout << "trueJets.size(): " << trueJets.size() << "\n";
      std::cout << "hybridToTrueIndex.size(): " << hybridToTrueIndex.size() << "\n";
      std::cout << "trueToHybridIndex.size(): " << trueToHybridIndex.size() << "\n";
      // Minimum jet pt check
      if (trueJet.pt() < 0.1) {
        std::cout << "True jet pt too low.\n";
        continue;
      }
      // Check distance is reasonable.
      if (hybridJet.delta_R(trueJet) > jetParameterR) {
        // Too far away!
        std::cout << "Too far away! Delta_R = " << hybridJet.delta_R(trueJet) << "\n";
        continue;
      }
      // Check shared momentum fraction
      if (SharedMomentumFraction(hybridJet, trueJet) < 0.5)
      {
        // Insufficiently similar jets.
        std::cout << "Insufficient shared momentum fraction: " << SharedMomentumFraction(hybridJet, trueJet) << "\n";
        continue;
      }

      trueJetSplittings.SetJetPt(trueJet.pt());
      Reclustering(trueJetSplittings, trueJet, storeRecursiveSplittings, applyTwoParticleAcceptanceCut);
      // Fill the matched jets.
      tree.Fill();
    }

    //ExtractWMass(hybridJets, 1, etamin_Sig, etamax_Sig);
    //ExtractWMassDijet(hybridJets, 1, etamin_Sig, etamax_Sig);

  } // end of event

  std::cout << "Number of entries: " << tree.GetEntries() << "\n";
  tree.Show(1);

  //____________________________________________________
  //          SAVE OUTPUT

  TString tag = TString::Format("pythia+thermal_substructure_toy_antikt_%02d", TMath::Nint(jetParameterR * 10));

  TFile* outFile =
   new TFile(TString::Format("%s_tune_%d_seed_%d_%s%s.root", tag.Data(), tune, randomSeed, charged ? "charged" : "full", underlingEvent ? "_underlyingEvent" : ""), "RECREATE");

  outFile->cd();
  tree.Write();
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

