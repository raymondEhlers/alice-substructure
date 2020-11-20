
#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TChain.h>
#include <TH1.h>
#include <TH2F.h>

void JESForEmcalPerformance()
{
  // Setup
  std::string detLevelPrefix = "data";
  std::string truePrefix = "matched";
  std::string outputFilename = "pythia_";
  outputFilename = "JESForEmcalPerformance_pythia";
  outputFilename += ".root";
  std::cout << "outputFilename: " << outputFilename << "\n";

  // Define hists
  // Response hists
  std::vector<TH1*> hists;
  TH2F hJES("hJES", "hJES;p_{T}^{part};(p_{T}^{det}-p_{T}^{part})/p_{T}^{part}", 250, 0, 250, 250, -5, 5);
  hists.push_back(&hJES);
  for (auto h : hists) {
    h->Sumw2();
  }

  // Setup response tree.
  TChain embeddedChain("tree");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/pythia/2110/run-by-run/FAST/skim/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/pythia/2110/run-by-run/cent_woSDD/skim/*.root");

  TTreeReader mcReader(&embeddedChain);
  TTreeReaderValue<float> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<float> detLevelJetPt(mcReader, ("jet_pt_" + detLevelPrefix).c_str());
  TTreeReaderValue<float> trueJetPt(mcReader, ("jet_pt_" + truePrefix).c_str());

  int counter = 0;
  while (mcReader.Next()) {
    if (counter % 1000000 == 0) {
        std::cout << "Jet: " << counter << "\n";
    }
    counter++;

    // JES
    hJES.Fill(*trueJetPt, (*detLevelJetPt - *trueJetPt) / *trueJetPt, *scaleFactor);
  }

  TFile fOut(outputFilename.c_str(), "RECREATE");
  fOut.cd();
  for (auto h: hists) {
      h->Write();
  }

  // Cleanup
  fOut.Write();
  fOut.Close();
}
