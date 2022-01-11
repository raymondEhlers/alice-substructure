
#include <TFile.h>
#include <TTree.h>
#include <fastjet/PsuedoJet.hH>

#include "output.h"

void contrived_root_tree()
{
  TFile * f = TFile::Open("test_splittings_tree.root", "RECREATE");
  int splitLevel = 4;
  int bufferSize = 32000;
  SubstructureTree::JetSubstructureSplittings hybridJetSplittings;
  SubstructureTree::JetSubstructureSplittings matchedJetSplittings;

  TTree tree("tree", "tree");
  tree.Branch("data.", &hybridJetSplittings, bufferSize, splitLevel);
  tree.Branch("matched.", &matchedJetSplittings, bufferSize, splitLevel);

  // Create constituents
  std::vector<fastjet::PseudoJet> detLevelConstituents = {
  // Skipping for now.
  void AddJetConstituent(const PWG::JETFW::AliEmcalParticleJetConstituent& part);
  void AddJetConstituent(PWG::JETFW::AliEmcalParticleJetConstituent());
    fastjet::PseudoJet(1, 1, 1, 1),
    fastjet::PseudoJet(2, 2, 2, 2),
    fastjet::PseudoJet(3, 3, 3, 3),
  }
  for (std::size_t i = 0; i < detLevelConstituents.size(); i++) {
    detLevelConstituents.at(i).set_user_index(i);
  }
  std::vector<fastjet::PseudoJet> hybridConstituents = {
    fastjet::PseudoJet(1, 1, 1, 1),
    fastjet::PseudoJet(2, 2, 2, 2),
    fastjet::PseudoJet(3, 3, 3, 3),
  }
  for (std::size_t i = 0; i < hybridConstituents.size(); i++) {
    hybridConstituents.at(i).set_user_index(i + 100);
  }

  // Create fake jet splittings
  dataJetSplittings.SetJetPt(100);
  // Constituents
  for (auto part : hybridConstituents) {
    dataJetSplittings.AddJetConstituent(part);
  }
  dataJetSplittings.AddJetConstituent(detLevelConstituents.at(0));
  dataJetSplittings.AddJetConstituent(detLevelConstituents.at(2));
  // Splittings
  //void AddSplitting(float kt, float deltaR, float z, short parentIndex);
  dataJetSplittings.AddSplitting(5, 0.3, 0.2, -1);
  dataJetSplittings.AddSplitting(10, 0.3, 0.2, 0);
  dataJetSplittings.AddSplitting(2, 0.3, 0.2, 0);
  dataJetSplittings.AddSplitting(15, 0.3, 0.2, 2);
  // Subjets
  // void AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting, const std::vector<unsigned short>& constituentIndices);
  // The constituents indices don't matter for this test.
  std::vector<unsigned short> constituentsIndices = {0, 1, 2};
  dataJetSplittings.AddSubjet(0, true, constituentsIndices);
  dataJetSplittings.AddSubjet(0, false, constituentsIndices);
  dataJetSplittings.AddSubjet(1, true, constituentsIndices);
  dataJetSplittings.AddSubjet(1, false, constituentsIndices);
  dataJetSplittings.AddSubjet(2, false, constituentsIndices);
  dataJetSplittings.AddSubjet(2, false, constituentsIndices);
  dataJetSplittings.AddSubjet(3, false, constituentsIndices);
  dataJetSplittings.AddSubjet(3, false, constituentsIndices);

  // Fill and write
  tree.Fill();
  tree.Write();
  f->Close();
}

