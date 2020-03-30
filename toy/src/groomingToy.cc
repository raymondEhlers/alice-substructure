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

#define nThermalParticles 4000

using namespace Pythia8;

double Calculate_pX(double pT, double eta, double phi) { return (pT * TMath::Cos(phi)); }

double Calculate_pY(double pT, double eta, double phi) { return (pT * TMath::Sin(phi)); }

double Calculate_pZ(double pT, double eta, double phi) { return (pT * TMath::SinH(eta)); }

double Calculate_E(double pT, double eta, double phi)
{
  double pZ = Calculate_pZ(pT, eta, phi);

  return (TMath::Sqrt(pT * pT + pZ * pZ));
}

bool JetInsideEtaLimits(fastjet::PseudoJet fjJet, double etaMin, double etaMax)
{
  if (fjJet.eta() > etaMax || fjJet.eta() < etaMin) {
    return false;
  } else {
    return true;
  }
}

bool AcceptJet(fastjet::PseudoJet & jet, double etaMin, double etaMax)
{
  if (JetInsideEtaLimits(jet, etaMin, etaMax) == false) {
    return false;
  }
  if (jet.pt() < 0.15) {
    return false;
  }

  return true;
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
  /*if (innerJets.size() == 0) {
    std::cout << "indexMap size: " << indexMap.size() << "\n";
  }*/

  return indexMap;
}

std::tuple<std::map<int, int>, std::map<int, int>> MatchJets(std::vector<fastjet::PseudoJet> & hybridJets, std::vector<fastjet::PseudoJet> & pythiaJets)
{
  std::map<int, int> trueToHybridIndex = PerformGeometricalMatching(pythiaJets, hybridJets);
  std::map<int, int> hybridToTrueIndex = PerformGeometricalMatching(hybridJets, pythiaJets);

  //std::cout << "trueToHybridIndex.size(): " << trueToHybridIndex.size() << "\n";

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

  //std::cout << "trueToHybridIndexVerified.size(): " << trueToHybridIndexVerified.size() << "\n";
  //std::cout << "hybridToTrueIndexVerified.size(): " << hybridToTrueIndexVerified.size() << "\n";

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
    // Skip ghosts
    if (constituent.user_index() == -1) {
      continue;
    }
    j1ConstituentIndices.emplace_back(constituent.user_index());
  }
  for (auto constituent: j2.constituents()) {
    // Skip ghosts
    if (constituent.user_index() == -1) {
      continue;
    }
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

void Reclustering(SubstructureTree::JetSubstructureSplittings & jetSplittings, fastjet::PseudoJet & jet, const bool storeRecursiveSplittings = true, const bool isData = false)
{
  // Grab the jet constituents from the jet.
  // They will be stored with the splitting, and also used for the declustering.
  std::vector<fastjet::PseudoJet> inputVectors;
  fastjet::PseudoJet pseudoTrack;
  unsigned int constituentIndex = 0;
  for (const auto & part : jet.constituents()) {
    //if (isData == true && fDoTwoTrack == kTRUE && CheckClosePartner(jet, part))
    //    continue;
    // Exclude ghosts
    if (part.user_index() == -1) {
      continue;
    }
    //if (part.pt() < 0.15) {
    //    std::cout << "Soft particle! pt: " << part.pt() << "\n";
    //}
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

void RetrieveFinalStateDaughterIndices(const Event & event, unsigned int index, std::vector<int> & daughters, bool keepOnlyCharged = true)
{
  std::vector<int> daughterIndices = event.daughterList(index);
  for (auto i : daughterIndices)
  {
    // If final state particle, store it.
    //if (event[i].pT() < 0.150) {
    //    continue;
    //}
    //if (std::abs(event[i].eta()) > 1)
    //    continue; // eta cut
    if (event[i].isFinal() ) {
      bool store = keepOnlyCharged ? event[i].isCharged() : true;
      if (store == true) {
        daughters.emplace_back(i);
      }
    }
    // Otherwise, recurse.
    RetrieveFinalStateDaughterIndices(event, i, daughters, keepOnlyCharged);
  }
}

std::vector<fastjet::PseudoJet> RetrieveFinalStateDaughters(const Event & event, unsigned int index, bool keepOnlyCharged, bool storeGlobalIndex = true)
{
  std::vector<fastjet::PseudoJet> daughters;
  std::vector<int> daughterIndices;
  RetrieveFinalStateDaughterIndices(event, index, daughterIndices, keepOnlyCharged);
  unsigned int constituentIndex = 0;
  //std::cout << "index " << index << " -> daughterIndices.size(): " << daughterIndices.size() << "\n";
  for (auto i : daughterIndices)
  {
    //std::cout << "daughter " << i << ", index: " << (storeGlobalIndex ? i : constituentIndex) << "\n";
    fastjet::PseudoJet j(
      event[i].px(),
      event[i].py(),
      event[i].pz(),
      event[i].e()
    );
    j.set_user_index(storeGlobalIndex ? i : constituentIndex);
    // Store the pythia constituents in the output
    daughters.emplace_back(j);
    constituentIndex++;
  }
  return daughters;
}

/*
void ExtractTruePythiaSplittings(SubstructureTree::JetSubstructureSplittings & jetSplittings, const Event & event, const int inputIndex, int splittingNodeIndex, bool followingIterativeSplitting, const bool storeRecursiveSplittings = true)
{
  if ((event[inputIndex].daughter1() > 0 && event[inputIndex].daughter2() > 0) == false) {
    // No parents, so we're done - just return.
    return;
  }

  // Retrieve the daughters.
  unsigned int index1 = event[inputIndex].daughter1();
  unsigned int index2 = event[inputIndex].daughter2();
  fastjet::PseudoJet j1(
    event[index1].px(),
    event[index1].py(),
    event[index1].pz(),
    event[index1].e()
  );
  fastjet::PseudoJet j2(
    event[index2].px(),
    event[index2].py(),
    event[index2].pz(),
    event[index2].e()
  );

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

}*/

/*void RecursiveTruePythia(SubstructureTree::JetSubstructureSplittings & jetSplittings, const Event & event, const int startingIndex, const bool storeRecursiveSplittings = true)
{
  // First, extract the constituents.
  // We only store those that are descended from the starting index.
  // NOTE: The order of the constituents may be different than for the other splittings, but it shouldn't matter.
  std::vector<fastjet::PseudoJet> daughters = RetrieveFinalStateDaughters(event, startingIndex);
  for (auto part : daughters) {
    // Store the pythia constituents in the output.
    jetSplittings.AddJetConstituent(part);
  }

  // Store the jet splittings.
  int splittingNodeIndex = -1;
  ExtractTruePythiaSplittings(jetSplittings, event, startingIndex, splittingNodeIndex, true, storeRecursiveSplittings);*/

  /*while (event[index].daughter1() > 0 && event[index].daughter2() > 0) {
    // j1 should always be the harder of the two subjets.
    if (j1.perp() < j2.perp()) {
      swap(j1, j2);
    }

    // Calculate the splitting properties and construct the object.
    double xz = j2.perp() / (j2.perp() + j1.perp());
    double xDeltaR = j1.delta_R(j2);
    double xkt = j2.perp() * std::sin(deltaR);

    if (xkt > kt && xDeltaR <= jetParameterR) {
      z = xz;
      kt = xkt;
      deltaR = xDeltaR;
    }
  }*/
//}

std::vector<unsigned short> FindSubjetConstituentsFromAllConstituents(const std::vector<int> & subjetConstituentGlobalIndices, const std::vector<fastjet::PseudoJet> & allConstituents)
{
  std::vector<unsigned short> subjetConstituentIndices;
  //std::cout << "Number of all constituents: " << allConstituents.size() << "\n";
  for (const auto & subjetConstituentGobalIndex : subjetConstituentGlobalIndices)
  {
    bool found = false;
    //std::cout << "Looking at subjet constituent with user_index: " << subjetConstituentGobalIndex << "\n";
    for (std::size_t constituentIndex = 0; constituentIndex < allConstituents.size(); constituentIndex++)
    {
      //std::cout << "constituent " << constituentIndex << ", user_index: " << allConstituents[constituentIndex].user_index() << "\n";
      if (subjetConstituentGobalIndex == allConstituents[constituentIndex].user_index()) {
        found = true;
        //std::cout << "Found match with constituentIndex: " << constituentIndex << "\n";
        subjetConstituentIndices.emplace_back(constituentIndex);
        break;
      }
    }
    if (found == false) {
      std::cout << "Failed to find matching constituent for " << subjetConstituentGobalIndex << "\n";
      std::exit(1);
    }
  }

  return subjetConstituentIndices;
}

void ExtractFirstPythiaSplitting(SubstructureTree::JetSubstructureSplittings & splittingsObj, const Event & event, const int startingIndex, const bool charged)
{
  // Search for the first splitting with a non-zero kt, starting with the startingIndex splitting.
  int i = startingIndex;
  bool foundSplitting = false;
  unsigned int index1 = 0;
  unsigned int index2 = 0;
  fastjet::PseudoJet j1, j2;
  while (foundSplitting == false)
  {
    index1 = event[i].daughter1();
    index2 = event[i].daughter2();
    //if (index1 == index2) {
    //    std::cout << "Daughter indices are equal!!!" << "\n";
    //}
    j1.reset(
      event[index1].px(),
      event[index1].py(),
      event[index1].pz(),
      event[index1].e()
    );
    j2.reset(
      event[index2].px(),
      event[index2].py(),
      event[index2].pz(),
      event[index2].e()
    );

    // j1 should always be the harder of the two subjets.
    if (j1.perp() < j2.perp()) {
      swap(j1, j2);
      swap(index1, index2);
    }

    // If we are looking at index1 0, there are no more daughters. We didn't find a kt > 0 splitting,
    // and we have to give up here.
    if (index1 == 0) {
      break;
    }

    double xDeltaR = j1.delta_R(j2);
    double xkt = j2.perp() * std::sin(xDeltaR);

    // kt > 0 should be roughly equivalent to:
    // if (((index1 > 0) && (index2 > 0)) && (index1 != index2)) {
    // However, kt > 0 is better because it's a clearly physics motivated measure of the splitting.
    if (xkt > 0) {
      foundSplitting = true;
    }
    else {
      // Reassign i to the leading daughter index
      //std::cout << "Found kt=" << xkt << " for " << i << "->" << index1 << ", " << index2 << ". Recursing with " << index1 << "\n";
      i = index1;
    }
  }

  // This really shouldn't be possible!
  if (foundSplitting == false) {
    // For example, we ran out of daughters.
    std::cout << "Didn't find any kt > 0 splitting for index " << i << "->" << index1 << ", " << index2 <<"! Skipping splitting " << startingIndex << "!\n";
    return;
  }

  // We have found a splitting. Now extract the properties.
  // Base properties
  splittingsObj.SetJetPt(event[i].pT());

  // Add jet constituents
  //std::cout << "\nEvent " << iEvent << ", particle " << i << "\n\n";
  std::vector<fastjet::PseudoJet> daughters = RetrieveFinalStateDaughters(event, i, charged, true);
  for (const auto & part : daughters) {
    splittingsObj.AddJetConstituent(part);
  }

  // Calculate the splitting properties and store them
  double xz = j2.perp() / (j2.perp() + j1.perp());
  double xDeltaR = j1.delta_R(j2);
  double xkt = j2.perp() * std::sin(xDeltaR);
  //std::cout << "Found non-zero kt=" << xkt << ". Storing!\n";
  // Add splitting
  int splittingNodeIndex = -1;
  splittingsObj.AddSplitting(xkt, xDeltaR, xz, splittingNodeIndex);

  // Add subjet
  splittingNodeIndex = splittingsObj.GetNumberOfSplittings() - 1;
  // Given our mode here, this should always be 0!
  assert(splittingNodeIndex == 0);
  std::vector<int> j1DaughterIndices;
  RetrieveFinalStateDaughterIndices(event, index1, j1DaughterIndices, charged);
  std::vector<int> j2DaughterIndices;
  RetrieveFinalStateDaughterIndices(event, index2, j2DaughterIndices, charged);
  std::vector<unsigned short> j1ConstituentIndices = FindSubjetConstituentsFromAllConstituents(j1DaughterIndices, daughters);
  std::vector<unsigned short> j2ConstituentIndices = FindSubjetConstituentsFromAllConstituents(j2DaughterIndices, daughters);
  splittingsObj.AddSubjet(splittingNodeIndex, true, j1ConstituentIndices);
  splittingsObj.AddSubjet(splittingNodeIndex, false, j2ConstituentIndices);
}


//___________________________________________________________________
int main(int argc, char* argv[])
{
  int randomSeed = -1;  // unique number for each file
  int tune = -1;   // pythia tune
  int charged = 1; // full or track-based jets
  int underlingEvent = 1;    // underlying event (ISR+MPI)

  if (argc != 6) {
    cout << "Usage:" << endl << "./pygen <PythiaTune> <Seed> <nEvts> <underlingEvent> <jetR>" << endl;
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
  double trackLowPtCut = 0.150;                 // GeV
  double trackEtaCut = 1;
  Float_t ptHatMin = 50;
  Float_t ptHatMax = 300;

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

  // Force production to higher pT
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

  double jetEtaMin = -trackEtaCut + jetParameterR; // signal jet eta range
  double jetEtaMax = -jetEtaMin;

  fastjet::Strategy strategy = fastjet::Best;
  fastjet::RecombinationScheme recombScheme = fastjet::E_scheme;

  fastjet::JetDefinition* jetDefAKT_Sig = new fastjet::JetDefinition(fastjet::antikt_algorithm, jetParameterR, recombScheme, strategy);

  fastjet::GhostedAreaSpec ghostareaspec(trackEtaCut, 1, 0.05); // ghost
  // max rap, repeat, ghostarea default 0.01
  fastjet::AreaType areaType = fastjet::active_area_explicit_ghosts;
  fastjet::AreaDefinition* areaDef = new fastjet::AreaDefinition(areaType, ghostareaspec);

  // Fastjet inputs
  std::vector<fastjet::PseudoJet> inputsPythia;
  std::vector<fastjet::PseudoJet> inputsHybrid;

  //_________Thermal Particle density distributions (toy model)

  TF1* f_pT = new TF1("f_pT", "x*exp(-x/0.3)", 0.0, 400.0);
  f_pT->SetNpx(40000);

  TF1* f_eta = new TF1("f_eta", "1", -1.0, 1.0);
  f_eta->SetNpx(200);

  TF1* f_phi = new TF1("f_phi", "1", (-1.0) * TMath::Pi(), TMath::Pi());
  f_phi->SetNpx(700);

  //___________________________________________________________

  // Hists
  // Store all hists here for convenience later.
  std::vector<TH1*> hists;

  // Particle distributions
  // All particles
  TH1D hPtAllParticles("hPtAllParticles", ";p_{T}", 4000, 0.0, 400);
  TH1D hEtaAllParticles("hEtaAllParticles", ";#eta", 200, -1.0, 1.0);
  TH1D hPhiAllParticles("hPhiAllParticles", ";#phi", 700, -3.5, 3.5);
  hists.emplace_back(&hPtAllParticles);
  hists.emplace_back(&hEtaAllParticles);
  hists.emplace_back(&hPhiAllParticles);
  // Pythia
  TH1D hPtPythia("hPtPythia", ";p_{T}", 4000, 0.0, 400);
  TH1D hEtaPythia("hEtaPythia", ";#eta", 200, -1.0, 1.0);
  TH1D hPhiPythia("hPhiPythia", ";#varphi", 700, -3.5, 3.5);
  hists.emplace_back(&hPtPythia);
  hists.emplace_back(&hEtaPythia);
  hists.emplace_back(&hPhiPythia);
  // Thermal
  TH1D hPtThermal("hPtThermal", ";p_{T}", 4000, 0.0, 400);
  TH1D hEtaThermal("hEtaThermal", ";#eta", 200, -1.0, 1.0);
  TH1D hPhiThermal("hPhiThermal", ";#varphi", 700, -3.5, 3.5);
  hists.emplace_back(&hPtThermal);
  hists.emplace_back(&hEtaThermal);
  hists.emplace_back(&hPhiThermal);

  // Pythia information
  TProfile hXsection("fHistXsection", "fHistXsection;;xsection", 1, 0, 1);
  hists.emplace_back(&hXsection);
  TH1D hTrials("fHistTrials", "fHistTrials;;trials", 1, 0, 1);
  hists.emplace_back(&hTrials);

  // Jet matching information
  TH1D hMatchingDistance("hMatchingDistance", ";#Delta R", 100, 0, 1);
  hists.emplace_back(&hMatchingDistance);

  // Enable Sumw2
  for (auto & h: hists) {
    h->Sumw2();
  }

  // Define output trees.
  bool storeRecursiveSplittings = true;
  bool applyTwoParticleAcceptanceCut = false;
  int splitLevel = 4;
  int bufferSize = 32000;
  // We have two sets of trees:
  // 1. true <-> hybrid (which doesn't care about matching). Called `trueSplittingsTree`.
  // 2. pythia <-> hybrid (which does care). Called `pythiaHybridTree`.
  SubstructureTree::JetSubstructureSplittings trueJetSplittings;
  SubstructureTree::JetSubstructureSplittings pythiaJetSplittings;
  SubstructureTree::JetSubstructureSplittings hybridJetSplittings;
  // Contains the correlation between true splittings and hybrid splittings.
  TTree trueSplittingsTree("trueSplittingsTree", "trueSplittingsTree");
  // data will contain the hybrid jets
  trueSplittingsTree.Branch("data.", &hybridJetSplittings, bufferSize, splitLevel);
  // True will containing the true splittings as determined from pythia.
  trueSplittingsTree.Branch("true.", &trueJetSplittings, bufferSize, splitLevel);

  // Contains matches between pythia and hybrid (pythia + thermal) jets and splittings.
  // We reuse the hybrid jet splittings object above, adding the pythia jet splittings.
  TTree pythiaHybridTree("pythiaHybridTree", "pythiaHybridTree");
  // pythia will contain the pythia only jets.
  pythiaHybridTree.Branch("pythia.", &pythiaJetSplittings, bufferSize, splitLevel);
  // data will contain the hybrid (pythia + thermal) jets
  pythiaHybridTree.Branch("data.", &hybridJetSplittings, bufferSize, splitLevel);


  //___________________________________________________
  // Begin event loop. Generate event. Skip if error. List first one.
  for (int iEvent = 0; iEvent < nEvent; iEvent++) {
    //if (iEvent % 100 == 0) {
    //    std::cout << "Event " << iEvent << "\n";
    //}
    if (!pythia.next()) {
      continue;
    }

    // Cleanup
    // Reset jet finding inputs
    inputsPythia.clear();
    inputsHybrid.clear();
    // Reset jet splittings
    trueJetSplittings.Clear();
    pythiaJetSplittings.Clear();
    hybridJetSplittings.Clear();

    SubstructureTree::JetSubstructureSplittings parton6Splittings;
    SubstructureTree::JetSubstructureSplittings parton7Splittings;
    unsigned int globalIndex = 0;
    for (int i = 0; i < pythia.event.size(); ++i) {
      // Extract final state charged hadrons for jet-finding
      if (pythia.event[i].isFinal()) {
        if (charged == 1)
          if (!pythia.event[i].isCharged())
            continue; // only charged particles
        if (pythia.event[i].pT() < trackLowPtCut)
          continue; // pt cut
        if (std::abs(pythia.event[i].eta()) > trackEtaCut)
          continue; // eta cut
        fastjet::PseudoJet particle(
          pythia.event[i].px(),
          pythia.event[i].py(),
          pythia.event[i].pz(),
          pythia.event[i].e()
        );
        particle.set_user_index(globalIndex);
        inputsPythia.push_back(particle);
        inputsHybrid.push_back(particle);
        // Store particle properties
        hPtAllParticles.Fill(particle.pt());
        hEtaAllParticles.Fill(particle.eta());
        hPhiAllParticles.Fill(particle.phi_std());
        hPtPythia.Fill(particle.pt());
        hEtaPythia.Fill(particle.eta());
        hPhiPythia.Fill(particle.phi_std());
        ++globalIndex;
      }

      // Extract the primary splitting for comparison.
      if (std::abs(pythia.event[i].status()) == 23) {
        SubstructureTree::JetSubstructureSplittings splittingsObj;
        ExtractFirstPythiaSplitting(splittingsObj, pythia.event, i, static_cast<bool>(charged));
        if (i == 5) {
          parton6Splittings = splittingsObj;
        }
        if (i == 6) {
          parton7Splittings = splittingsObj;
        }
      }
    }
    //std::cout << "Number of particles (globalIndex): " << globalIndex << "\n";

    // Thermal Particles loop
    // Add an offset for thermal particles.
    globalIndex = 100000;
    for (int j = 0; j < nThermalParticles; j++) {
      double pT = f_pT->GetRandom();
      double eta = f_eta->GetRandom();
      double phi = f_phi->GetRandom();
      if (pT < trackLowPtCut)
        continue; // pt cut
      fastjet::PseudoJet thermalParticle(
        Calculate_pX(pT, eta, phi),
        Calculate_pY(pT, eta, phi),
        Calculate_pZ(pT, eta, phi),
        Calculate_E(pT, eta, phi)
      );
      thermalParticle.set_user_index(globalIndex);
      inputsHybrid.push_back(thermalParticle);
      // Store particle properties
      hPtAllParticles.Fill(thermalParticle.pt());
      hEtaAllParticles.Fill(thermalParticle.eta());
      hPhiAllParticles.Fill(thermalParticle.phi_std());
      hPtThermal.Fill(thermalParticle.pt());
      hEtaThermal.Fill(thermalParticle.eta());
      hPhiThermal.Fill(thermalParticle.phi_std());
      ++globalIndex;
    }

    //________________signal jets____________________________________________________
    fastjet::ClusterSequenceArea clustSeq_Sig(inputsPythia, *jetDefAKT_Sig, *areaDef);
    std::vector<fastjet::PseudoJet> pythiaJets = clustSeq_Sig.inclusive_jets(30.);

    if (pythiaJets.size() == 0) {
      // No true jets, so nothing else to be done.
      continue;
    }

    // We'll just take the leading jet for convenience.
    // We put it in a vector since the code was written for taking everything.
    fastjet::PseudoJet probeJet = pythiaJets[0];

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

    for (std::size_t j = 0; j < hybridJets.size(); j++) {
      const fastjet::PseudoJet& jet = hybridJets[j];
      fastjet::PseudoJet subtracted_Jet = subtractor(jet);
      hybridJets[j] = subtracted_Jet;
    }

    // Fill true tree!
    // NOTE: We want the leading hybrid jet, so we sort by pt.
    fastjet::PseudoJet hybridProbeJet = sorted_by_pt(hybridJets)[0];
    fastjet::PseudoJet parton6(pythia.event[5].px(), pythia.event[5].py(), pythia.event[5].pz(),
                  pythia.event[5].e());
    fastjet::PseudoJet parton7(pythia.event[6].px(), pythia.event[6].py(), pythia.event[6].pz(),
                  pythia.event[6].e());
    double deltaR6 = hybridProbeJet.delta_R(parton6);
    double deltaR7 = hybridProbeJet.delta_R(parton7);
    // Fill true pt with what?

    if (deltaR6 < 0.1) {
      trueJetSplittings = parton6Splittings;
    }
    if (deltaR7 < 0.1) {
      trueJetSplittings = parton7Splittings;
    }
    // Only fill if we're actually close to a splitting. Otherwise, we get empty true jet splittings
    // and/or we pull the hybrid jets to the edges of the eta acceptance.
    if ((deltaR6 < 0.1 || deltaR7 < 0.1) && trueJetSplittings.GetJetPt() > 0) {
      //std::cout << "True pt: " << trueJetSplittings.GetJetPt() << "\n";
      //// Splitting properties
      //float kt = 0, deltaR = 0, z = 0;
      //short parentIndex = 0;
      //std::tie(kt, deltaR, z, parentIndex) = trueJetSplittings.GetSplitting(0);
      //std::cout << "splitting info: kt=" << kt << ", deltaR=" << deltaR << ", z=" << z <<  "\n";
      //// Constituents
      // True jet splittings info
      //std::cout << "True jet: " << trueJetSplittings << "\n";
      hybridJetSplittings.SetJetPt(hybridProbeJet.pt());
      Reclustering(hybridJetSplittings, hybridProbeJet, storeRecursiveSplittings, applyTwoParticleAcceptanceCut);
      //std::cout << "hybrid jet pt=" << hybridJetSplittings.GetJetPt() << "\n";
      // Hybrid jet splittings info
      //std::cout << "Hybrid jet: " << hybridJetSplittings << "\n";

      // Sanity check on the kt value.
      // It must be positive (but apparently sometimes it isn't...)!
      // Apparently if deltaR is sufficiently large, it can lead to a negative kt because sin goes negative for values
      // greater than pi!
      //float kt = 0, deltaR = 0, z = 0;
      //short parentIndex = 0;
      //std::tie(kt, deltaR, z, parentIndex) = trueJetSplittings.GetSplitting(0);
      //if (kt <= 0) {
      //    std::cout << "kt <= 0. Waaaaat?\n";
      //    std::exit(1);
      //}
      trueSplittingsTree.Fill();
    }
    else {
      //std::cout << "Failed to match true splitting! deltaR6=" << deltaR6 << ", deltaR7=" << deltaR7 << "\n";
    }

    // Match jets
    // Need to do rudimentary matching
    std::map<int, int> pythiaToHybridIndex;
    std::map<int, int> hybridToPythiaIndex;
    std::tie(pythiaToHybridIndex, hybridToPythiaIndex) = MatchJets(hybridJets, pythiaJets);

    // Extract the splittings for each set of matched jets.
    //std::cout << "About to recluster event " << iEvent << "\n";
    for (std::size_t hybridIndex = 0; hybridIndex < hybridJets.size(); hybridIndex++) {
      // Setup. Ensure that the tree outputs are clear for each set of jets to fill it.
      hybridJetSplittings.Clear();
      pythiaJetSplittings.Clear();

      if (hybridToPythiaIndex.count(hybridIndex) == 0) {
        // This jet doesn't have a match. Skip it.
        //std::cout << "No match for this hybrid jet. Skipping\n";
        continue;
      }
      // Hybrid jet
      fastjet::PseudoJet & hybridJet = hybridJets[hybridIndex];
      if (AcceptJet(hybridJet, jetEtaMin, jetEtaMax) == false) {
        //std::cout << "Hybrid jet rejected.\n";
        continue;
      }
      hybridJetSplittings.SetJetPt(hybridJet.pt());
      Reclustering(hybridJetSplittings, hybridJet, storeRecursiveSplittings, applyTwoParticleAcceptanceCut);
      // True jet
      int pythiaJetIndex = hybridToPythiaIndex[hybridIndex];
      if (pythiaJetIndex == -1) {
        // No match - continue.
        continue;
      }
      fastjet::PseudoJet & pythiaJet = pythiaJets[pythiaJetIndex];
      if (AcceptJet(pythiaJet, jetEtaMin, jetEtaMax) == false) {
        //std::cout << "True jet rejected.\n";
        continue;
      }
      double matchingDistance = hybridJet.delta_R(pythiaJet);
      hMatchingDistance.Fill(matchingDistance);
      // Check distance is reasonable.
      if (matchingDistance > jetParameterR) {
        // Too far away!
        std::cout << "Too far away! Delta_R = " << matchingDistance << "\n";
        continue;
      }
      // Check shared momentum fraction
      if (SharedMomentumFraction(hybridJet, pythiaJet) < 0.5)
      {
        // Insufficiently similar jets.
        std::cout << "Insufficient shared momentum fraction: " << SharedMomentumFraction(hybridJet, pythiaJet) << "\n";
        continue;
      }

      pythiaJetSplittings.SetJetPt(pythiaJet.pt());
      Reclustering(pythiaJetSplittings, pythiaJet, storeRecursiveSplittings, applyTwoParticleAcceptanceCut);

      // Check number of stored jet constituents
      /*std::cout << "Number of hybrid jet constituents: " << hybridJet.constituents().size() << "\n";
      std::cout << "hybrid constituents: " << hybridJetSplittings.GetNumberOfJetConstituents() << "\n";
      std::cout << "Number of true jet constituents: " << pythiaJet.constituents().size() << "\n";
      std::cout << "true constituents: " << pythiaJetSplittings.GetNumberOfJetConstituents() << "\n";
      std::cout << "fill tree\n";*/
      // Fill the matched jets.
      pythiaHybridTree.Fill();
    }

  } // end of event

  std::cout << "Number of true splittings: " << trueSplittingsTree.GetEntries() << "\n";
  std::cout << "Number of matched jets: " << pythiaHybridTree.GetEntries() << "\n";

  //____________________________________________________
  //          SAVE OUTPUT

  TString tag = TString::Format("pythia+thermal_substructure_toy_antikt_%02d", TMath::Nint(jetParameterR * 10));

  TFile* outFile =
   new TFile(TString::Format("%s_tune_%d_seed_%03d_%s%s_ptHatMin_%d.root", tag.Data(), tune, randomSeed, charged ? "charged" : "full", underlingEvent ? "_underlyingEvent" : "", static_cast<int>(ptHatMin)), "RECREATE");

  outFile->cd();
  // Write out hists
  for (auto & h: hists) {
    h->Write();
  }
  // And tree
  trueSplittingsTree.Write();
  pythiaHybridTree.Write();
  outFile->Close();

  pythia.stat();
  return 0;
}

//_________________________________________________________________________
//_________________________________________________________________________
//_________________________________________________________________________
//_________________________________________________________________________

