
#include <string>
#include <vector>

#include <TTree.h>
#include <TTreeReader.h>
#include <TH1.h>
#include <TH2F.h>


void embeddingFallBack()
{
  // Setup
  std::string hybridPrefix = "hybrid";
  std::string truePrefix = "true";
  std::string groomingMethod = "leading_kt";
  std::string substructureVariableName = "kt";
  std::string outputFilename = "embeddingResponse_";
  outputFilename += substructureVariableName;
  outputFilename += "_grooming_method_";
  outputFilename += groomingMethod;

  // Define hists
  std::vector<TH1*> hists;
  TH2F hHybridTrueKtResponse("hHybridTrueKtResponse", "hHybridTrueKtResponse", 25, 0, 25, 25, 0, 25);
  hists.push_back(&hHybridTrueKtResponse);

  for (h : hists) {
    h->Sumw2();
  }

  // Setup response tree.
  TChain embeddedChain("tree");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5884/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5885/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5886/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5887/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5888/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5889/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5890/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5891/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5892/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5893/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5894/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5895/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5896/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5897/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5898/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5898/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5900/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5901/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5902/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5903/skim/merged/*.root");

  std::string truePrefix = "true";
  std::string hybridPrefix = "hybrid";
  TTreeReader mcReader(&embeddedChain);
  TTreeReaderValue<float> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<float> hybridJetPt(mcReader, ("jet_pt_" + hybridPrefix).c_str());
  TTreeReaderValue<float> hybridSubstructureVariable(mcReader, (groomingMethod + "_" + hybridPrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<float> trueJetPt(mcReader, ("jet_pt_" + truePrefix).c_str());
  TTreeReaderValue<float> trueSubstructureVariable(mcReader, (groomingMethod + "_" + truePrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<long long> matchingLeading(mcReader, (groomingMethod + "_hybrid_detector_matching_leading").c_str());
  TTreeReaderValue<long long> matchingSubleading(mcReader, (groomingMethod + "_hybrid_detector_matching_subleading").c_str());

  while (mcReader.Next()) {
    // Ensure that we are in the right true pt and substructure variable range.
    if (*trueJetPt > 160) {
      continue;
    }
    if (*trueSubstructureVariable > 100) {
      continue;
    }

    h2fulleff.Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    h2smearednocuts.Fill(*hybridSubstructureVariable, *hybridJetPt, *scaleFactor);
    responsenotrunc.Fill(*hybridSubstructureVariable, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor);

    // Now start making cuts on the hybrid level.
    if (*hybridJetPt < 40 || *hybridJetPt > 120) {
      continue;
    }
    // Also cut on hybrid substructure variable.
    double hybridSubstructureVariableValue = *hybridSubstructureVariable;
    // TODO: This only works for kt!!!
    /*if (hybridSubstructureVariableValue < 0) {
      // Assign to the untagged bin.
      hybridSubstructureVariableValue = 0.5;
    }
    else {
      if (hybridSubstructureVariableValue < minSmearedSplittingVariable || hybridSubstructureVariableValue > smearedSplittingVariableBins[smearedSplittingVariableBins.size() - 1]) {
        continue;
      }
    }
    // Matching cuts: Requiring a pure match.
    if (usePureMatches && !(*matchingLeading == 1 && *matchingSubleading == 1)) {
      continue;
    }*/
    hHybridTrueKtResponse.Fill(hybridSubstructureVariableValue, &trueSubstructureVariable);
    //h2smeared.Fill(hybridSubstructureVariableValue, *hybridJetPt, *scaleFactor);
    //h2true.Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    //response.Fill(hybridSubstructureVariableValue, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor);

    // Next, if this works: matching
  }

  TFile fOut(outputFilename.c_str(), "RECREATE");
  fout->cd();
  hHybridTrueKtResponse->Write();

}
