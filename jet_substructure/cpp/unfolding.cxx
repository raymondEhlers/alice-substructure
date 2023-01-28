//#if !(defined(__CINT__) || defined(__CLING__)) || defined(__ACLIC__)
#include <RooUnfoldBayes.h>
#include <RooUnfoldResponse.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TNtuple.h>
#include <TPostScript.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TRandom3.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include <iostream>
#include <memory>
#include <vector>
//#endif

namespace unfolding {

constexpr double DISABLE_UNTAGGED_BIN_VALUE = -1e5;

/**
 * Determine if we have a pure match.
 *
 * @param[in] matchingLeading Matching status for the leading subjet.
 * @param[in] matchingSubleading Matching status for the subleading subjet.
 * @param[in] responseSmearedSubstructureVariableValue Substructure variable value.
 * @param[in] smearedUntaggedBinValue Value that will be assigned to the untagged bin for checking if we have this
 * value.
 *
 * @returns True if we have a pure match.
 */
inline bool isPureMatch(int matchingLeading, int matchingSubleading,
                        double responseSmearedSubstructureVariableValue, double smearedUntaggedBinValue)
{
  return (((std::abs(matchingLeading - 1) < 0.001) && (std::abs(matchingSubleading - 1) < 0.001)) ||
      (std::abs(responseSmearedSubstructureVariableValue - smearedUntaggedBinValue) < 0.001));
}

/**
 * Find reweighting bin, ensuring that we take the last good bin if we're out of range.
 *
 * For example, if we're below the lowest bin, then we take the first bin. Effectively,
 * we neglect the overflow bins.
 *
 * @param[in] value Value that we want to reweight.
 * @param[in] bins Binning of the reweighting hist.
 *
 * @returns Bin to use when retrieving the reweighting value.
 */
int findReweightingBin(double value, const std::vector<double>& bins)
{
  auto bin = std::distance(bins.begin(), std::upper_bound(bins.begin(), bins.end(), value));
  if (bin == 0) {
    ++bin;
  }
  if (bin >= bins.size()) {
    --bin;
  }

  return bin;
}

/**
 * Wrapper around the RooUnfoldResponse. Just for convenience.
 *
 * Since this wrapper is passed back to python, the field are named for Python.
 */
struct ResponseResult {
  std::shared_ptr<RooUnfoldResponse> response;
  std::shared_ptr<RooUnfoldResponse> response_no_trunc;
};

struct Settings2D {
  const std::string groomingMethod;
  const std::string substructureVariableName;
  std::vector<double> smearedJetPtBins;
  std::vector<double> trueJetPtBins;
  std::vector<double> smearedSplittingVariableBins;
  std::vector<double> trueSplittingVariableBins;
  double smearedUntaggedBinValue;
  bool disableUntaggedBin;
  double minSmearedSplittingVariable;
  double maxSmearedSplittingVariable;
  const bool usePureMatches = false;
  const bool unfoldingForPP = false;
  TH2D* hReweightingResponse = nullptr;
  const bool normalizeSubstructureVariableByJetPt = false;
};

struct DoubleCountingCut {
  bool useDetLevelTrackPtCut = false;
};

struct InputFilenames {
  const std::vector<std::string> data;
  const std::vector<std::string> response;
};

struct TreeNames {
  const std::string data = "tree";
  const std::string response = "tree";
};

struct Prefixes {
  const std::string data = "data";
  const std::string responseSmeared = "hybrid";
  const std::string responseTrue = "true";
  const std::string responseDetLevel = "det_level";
};

enum ClosureVariation_t { splitMC = 0, reweightPseudoData = 1, reweightResponse = 2 };

struct ClosureSettings {
  const ClosureVariation_t variation;
  const double fractionForResponse = 0.75;
};

/**
 * Interface to 2D unfolding with python.
 *
 */
ResponseResult create_response_2D(
  std::map<std::string, TH2D*> hists,
  const Settings2D settings,
  const DoubleCountingCut doubleCountingCut,
  const InputFilenames inputFilenames,
  const TreeNames treeNames,
  const Prefixes prefixes,
  std::ostream & stdoutStream = std::cout,
  std::ostream & stderrStream = std::cerr
) {
  // Print out the status.
  stdoutStream << "Binning and values:\n";
  stdoutStream << "Grooming method: " << settings.groomingMethod << "\n";
  stdoutStream << "Smeared jet pt bins:";
  for (auto v : settings.smearedJetPtBins) {
      stdoutStream << " " << v;
  }
  stdoutStream << "\nTrue pt bins:";
  for (auto v : settings.trueJetPtBins) {
      stdoutStream << " " << v;
  }
  stdoutStream << "\nSmeared substructure variable bins:";
  for (auto v : settings.smearedSplittingVariableBins) {
      stdoutStream << " " << v;
  }
  stdoutStream << "\nTrue substructure variable bins:";
  for (auto v : settings.trueSplittingVariableBins) {
      stdoutStream << " " << v;
  }
  stdoutStream << "\n";
  stdoutStream << "Smeared untagged bin value: " << settings.smearedUntaggedBinValue << "\n";
  stdoutStream << std::boolalpha << "Disable untagged bin: " << settings.disableUntaggedBin << "\n";
  stdoutStream << "Min smeared substructure variable: " << settings.minSmearedSplittingVariable << "\n";
  stdoutStream << "Max smeared substructure variable: " << settings.maxSmearedSplittingVariable << "\n";
  stdoutStream << std::boolalpha << "Use pure matches: " << settings.usePureMatches << "\n";
  stdoutStream << std::boolalpha << "Unfolding for pp: " << settings.unfoldingForPP << "\n";
  // Add some space before the filename printouts.
  stdoutStream << "\n";

  // General setup
  double untaggedBelowThisValue = 0.;
  if (settings.disableUntaggedBin) {
      // Select a very large negative value. We'll never have such a large negative value, so
      // practically this means that we'll never mark a value as untagged. This means that everything
      // will have to be encapsulated in the standard binning or it will be cut.
      untaggedBelowThisValue = DISABLE_UNTAGGED_BIN_VALUE;
  }

  // First, we handle the data. Setup the Reader, the columns, and store the data in the appropriate hists.
  TChain dataChain(treeNames.data.c_str());
  stdoutStream << "Data filenames:\n";
  for (auto filename : inputFilenames.data) {
    stdoutStream << " - " << filename << "\n";
    dataChain.Add(filename.c_str());
  }
  TTreeReader dataReader(&dataChain);

  TTreeReaderValue<float> dataJetPt(dataReader, (prefixes.data + "_jet_pt").c_str());
  TTreeReaderValue<float> dataSubstructureVariable(
   dataReader, (settings.groomingMethod + "_" + prefixes.data + "_" + settings.substructureVariableName).c_str());
  while (dataReader.Next()) {
    // Jet pt cut.
    if (*dataJetPt < settings.smearedJetPtBins.front() || *dataJetPt > settings.smearedJetPtBins.back()) {
      continue;
    }
    // Substructure variable cut.
    double dataSubstructureVariableValue = *dataSubstructureVariable;
    // Option to normalize by jet pt
    // NOTE: It's okay to normalize before checking the untagged below value
    //       because that's either 0 or hugely negative, and normalizing won't change the sign.
    if (settings.normalizeSubstructureVariableByJetPt) {
      dataSubstructureVariableValue /= *dataJetPt;
    }
    if (dataSubstructureVariableValue < untaggedBelowThisValue) {
      // Assign to the untagged bin.
      dataSubstructureVariableValue = settings.smearedUntaggedBinValue;
      // If we have the high bin (for the systematic), then we need to normalize in
      // order to match the binning (which will shift depending on whether it is normalized or not).
      // NOTE: This is not necessarily for the lower range, and in fact is preferable
      //       to not touch it to make it easier to detect for pure matches.
      // TODO: As of 27 Jan, we try to skip this. If it doesn't work, return it, but then pure matches needs some help!
      /*if (dataSubstructureVariableValue > 0) {
        dataSubstructureVariableValue /= *dataJetPt;
      }*/
    } else {
      if (dataSubstructureVariableValue < settings.minSmearedSplittingVariable ||
        dataSubstructureVariableValue > settings.maxSmearedSplittingVariable) {
        continue;
      }
    }
    hists["h2_raw"]->Fill(dataSubstructureVariableValue, *dataJetPt);
  }

  // Now, switch to response (ie. embedded for PbPb)
  // First, setup the response
  // NOTE: We allocate a shared_ptr, but don't delete here because we want to return the response
  //       without copying. If we do copy, RooUnfold doesn't seem to behave identically. It may not make
  //       a difference, but better not to tempt fate. Instead, we pass ownership to the caller.
  auto response = std::make_shared<RooUnfoldResponse>();
  auto responsenotrunc = std::make_shared<RooUnfoldResponse>();
  response->Setup(hists["h2_smeared"], hists["h2_true"]);
  responsenotrunc->Setup(hists["h2_smeared_no_cuts"], hists["h2_full_eff"]);

  // Next, we setup the Reader, the columns, and store the data in the appropriate hists.
  TChain responseChain(treeNames.response.c_str());
  stdoutStream << "Embedded filenames:\n";
  for (auto filename : inputFilenames.response) {
    stdoutStream << " - " << filename << "\n";
    responseChain.Add(filename.c_str());
  }
  // NOTE: We have no more need for friend trees because we skim everything. But we'll
  //       keep the below code commented as an example if we need to revive it.
  // NOTE: If we were to use this, we would need to make it conditional, as it's most
  //       likely wouldn't always be appropriate.
  /*TChain responseScaleFactors("tree");
  for (const auto filename : responseFilenames) {
      // This is supposed to be equivalent to python with:
      // friend_tree.Add(str(filename.parent.parent / "scale_factor" / filename.name))
      std::string temp = filename;
      // Remove entries from the end of the path.
      // This depends on passing the directory in the "skim" directory.
      // +1 to skip over the "/"
      std::string name = temp.substr(temp.rfind("/") + 1);
      temp.erase(temp.rfind("/"));
      temp.erase(temp.rfind("/"));
      responseScaleFactors.Add((temp + "/scale_factor/" + name).c_str());
  }
  responseChain.AddFriend(&responseScaleFactors);*/
  //responseChain.Print();
  TTreeReader mcReader(&responseChain);

  // Values
  TTreeReaderValue<float> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<float> responseSmearedJetPt(mcReader, (prefixes.responseSmeared + "_jet_pt").c_str());
  TTreeReaderValue<float> responseSmearedSubstructureVariable(
   mcReader, (settings.groomingMethod + "_" + prefixes.responseSmeared + "_" + settings.substructureVariableName).c_str());
  TTreeReaderValue<float> trueJetPt(mcReader, (prefixes.responseTrue + "_jet_pt").c_str());
  TTreeReaderValue<float> trueSubstructureVariable(
   mcReader, (settings.groomingMethod + "_" + prefixes.responseTrue + "_" + settings.substructureVariableName).c_str());
  // These values are only conditionally defined (ie. they are for embedding, but not for pythia)
  // It's kind of awkward to have a pointer to a TTreeReaderValue, but it's the only way that I can
  // see to conditionally define them. So I just suck it up.
  std::unique_ptr<TTreeReaderValue<int16_t>> matchingLeading = nullptr;
  std::unique_ptr<TTreeReaderValue<int16_t>> matchingSubleading = nullptr;
  // For the double counting cut.
  std::unique_ptr<TTreeReaderValue<float>> responseSmearedUnsubLeadingTrackPt = nullptr;
  std::unique_ptr<TTreeReaderValue<float>> detLevelLeadingTrackPt = nullptr;
  if (!settings.unfoldingForPP) {
      matchingLeading = std::make_unique<TTreeReaderValue<int16_t>>(mcReader,
                            (settings.groomingMethod + "_hybrid_det_level_matching_leading").c_str());
      matchingSubleading = std::make_unique<TTreeReaderValue<int16_t>>(mcReader,
                              (settings.groomingMethod + "_hybrid_det_level_matching_subleading").c_str());
      // Potentially useful for the double counting cut.
      responseSmearedUnsubLeadingTrackPt = std::make_unique<TTreeReaderValue<float>>(mcReader, (prefixes.responseSmeared + "_leading_track_pt").c_str());
      detLevelLeadingTrackPt = std::make_unique<TTreeReaderValue<float>>(mcReader, (prefixes.responseDetLevel + "_leading_track_pt").c_str());
  }

  int treeNumber = -1;
  // double scaleFactor = 0;
  while (mcReader.Next()) {
    // Check if the file changed.
    if (treeNumber < responseChain.GetTreeNumber()) {
      // File changed. Update the scale factor.
      // auto f = responseChain.GetFile();
      // scaleFactor = GetScaleFactor(f);
      // Update the tree number so we hold onto the scale factor until the next time we need to update.
      treeNumber = responseChain.GetTreeNumber();
    }
    // Ensure that we are in the right true pt (and substructure variable) range.
    if (*trueJetPt < settings.trueJetPtBins.front() ||
        *trueJetPt > settings.trueJetPtBins.back()) {
      continue;
    }
    // To preserve the option to normalize the true substructure variable, we need to store the value here
    double trueSubstructureVariableValue = *trueSubstructureVariable;
    // Normalize by jet pt if requested.
    // NOTE: It's okay to normalize here, even if value is the untagged value because it's less than 0 and
    //       we don't move it around. Thus, dividing by pt will always keep the same sign, and therefore
    //       it will always go into the right position.
    if (settings.normalizeSubstructureVariableByJetPt) {
      trueSubstructureVariableValue /= *trueJetPt;
    }
    // Ensure that we are in the right substructure variable range.
    // NOTE: For the enabled vs disabled untagged case, we have:
    //       disabled: We want to require that the min value is within the bin range
    //       enabled: We want to require that the min value is within the bin range, but
    //                the bins include the untagged on the lower side (it won't move, unlike
    //                smeared, which is why the true is handled more simply). This cut will
    //                require that the value is within the lower range of the untagged bin,
    //                which means that it will trivially be within the actual unfolding range
    //                (and thus, this cut won't necessarily address that requirement). To use
    //                this cut to have a gap between the untagged bin and the true bin range,
    //                we would have to change our binning convention. (We could also try to increase
    //                the upper range of the untagged to match whatever min true value we would want,
    //                but then we would still have to cut on that front + 1 value of the binning,
    //                which we don't currently do. As of 27 Jan 2023, I don't want to add the complexity
    //                for something that we don't use - it would require changing a similar but slightly
    //                different condition for the untagged bin being enabled vs disabled - so we skip
    //                it for now).
    if (trueSubstructureVariableValue < settings.trueSplittingVariableBins.front() ||
        trueSubstructureVariableValue > settings.trueSplittingVariableBins.back()) {
      continue;
    }
    // Finish up possible double counting cut selections (other aspects of the cut are implemented earlier)
    if (!settings.unfoldingForPP && doubleCountingCut.useDetLevelTrackPtCut) {
      // NOTE: We use less than because if the leading track pt is the same in both jets (which is most likely
      //       what we would want), then the pt will be equal.
      if (**detLevelLeadingTrackPt < **responseSmearedUnsubLeadingTrackPt) {
        continue;
      }
    }

    // Potentially Reweight
    double reweightFactor = 1;
    if (settings.hReweightingResponse) {
        // NOTE: We intentionally look at the true values even though it's binned at detector level.
        // We need to handle the binning carefully, so we use a dedicated function.
        int ktBin = findReweightingBin(trueSubstructureVariableValue, settings.smearedSplittingVariableBins);
        int jetPtBin = findReweightingBin(*trueJetPt, settings.smearedJetPtBins);
        reweightFactor = settings.hReweightingResponse->GetBinContent(ktBin, jetPtBin);
    }

    // To preserve the option to normalize the smeared substructure variable, we need to store the value here
    double responseSmearedSubstructureVariableValue = *responseSmearedSubstructureVariable;
    if (settings.normalizeSubstructureVariableByJetPt) {
      responseSmearedSubstructureVariableValue /= *responseSmearedJetPt;
    }

    // Full efficiency hists (and response).
    hists["h2_full_eff"]->Fill(trueSubstructureVariableValue, *trueJetPt, *scaleFactor * reweightFactor);
    hists["h2_smeared_no_cuts"]->Fill(responseSmearedSubstructureVariableValue, *responseSmearedJetPt, *scaleFactor * reweightFactor);
    responsenotrunc->Fill(responseSmearedSubstructureVariableValue, *responseSmearedJetPt, trueSubstructureVariableValue, *trueJetPt,
               *scaleFactor * reweightFactor);

    // Now start making cuts on the response smeared level.
    // Jet pt
    if (*responseSmearedJetPt < settings.smearedJetPtBins.front() ||
        *responseSmearedJetPt > settings.smearedJetPtBins.back()) {
      continue;
    }
    // Also cut on smeared substructure variable.
    if (responseSmearedSubstructureVariableValue < untaggedBelowThisValue) {
      // Assign to the untagged bin.
      responseSmearedSubstructureVariableValue = settings.smearedUntaggedBinValue;
      // If we have the high bin (for the systematic), then we need to normalize in
      // order to match the binning (which will shift depending on whether it is normalized or not).
      // NOTE: This is not necessarily for the lower range, and in fact is preferable
      //       to not touch it to make it easier to detect for pure matches.
      // TODO: As of 27 Jan, we try to skip this. If it doesn't work, return it, but then pure matches needs some help!
      /*if (responseSmearedSubstructureVariableValue > 0) {
        responseSmearedSubstructureVariableValue /= *responseSmearedJetPt;
      }*/
    } else {
      // NOTE: These min and max values take into account the location of the untagged, so they're
      //       filtering the "main" variable range (ie. not including the untagged)
      if (responseSmearedSubstructureVariableValue < settings.minSmearedSplittingVariable ||
        responseSmearedSubstructureVariableValue > settings.maxSmearedSplittingVariable) {
        continue;
      }
    }
    // Matching cuts: Requiring a pure match.
    if (!settings.unfoldingForPP &&
         settings.usePureMatches &&
        !isPureMatch(**matchingLeading, **matchingSubleading,
                     responseSmearedSubstructureVariableValue,
                     settings.smearedUntaggedBinValue)
                     // TODO: As of 27 Jan, we try to skip this. If it doesn't work, return it, but then pure matches needs some help!
                     //responseSmearedSubstructureVariableValue < 0 ? settings.smearedUntaggedBinValue : settings.smearedUntaggedBinValue / *responseSmearedJetPt)
        ) {
      continue;
    }

    // At this point, we've passed all of our cuts, so we store the result.
    hists["h2_smeared"]->Fill(responseSmearedSubstructureVariableValue, *responseSmearedJetPt, *scaleFactor * reweightFactor);
    hists["h2_true"]->Fill(trueSubstructureVariableValue, *trueJetPt, *scaleFactor * reweightFactor);
    hists["h2_substructure_variable"]->Fill(responseSmearedSubstructureVariableValue, trueSubstructureVariableValue,
                        *scaleFactor * reweightFactor);
    response->Fill(responseSmearedSubstructureVariableValue, *responseSmearedJetPt, trueSubstructureVariableValue, *trueJetPt,
            *scaleFactor * reweightFactor);
  }

  return ResponseResult{ response, responsenotrunc };
}

/**
 * Create a reweighted response (or pseudo-data).
 *
 * Since we utilize pseudo-data here, we split statistics according to a specified fraction.
 * Consequently, this also covers the split MC closure.
 * NOTE: We rely on the user to validate that the binning matches the reweighting hist.
 *
 */
ResponseResult create_closure_response_2D(
  std::map<std::string, TH2D*> hists,
  const Settings2D settings,
  const ClosureSettings closureSettings,
  const DoubleCountingCut doubleCountingCut,
  const InputFilenames inputFilenames,
  const TreeNames treeNames,
  const Prefixes prefixes,
  std::ostream & stdoutStream = std::cout,
  std::ostream & stderrStream = std::cerr
) {
  // NOTE: We rely on the user to validate that the binning matches the reweighting hist.
  // Setup
  TRandom3 random(0);
  // Handle untagged bin.
  double untaggedBelowThisValue = 0.;
  if (settings.disableUntaggedBin) {
      // Select a very large negative value. We'll never have such a large negative value, so
      // practically this means that we'll never mark a value as untagged. This means that everything
      // will have to be encapsulated in the standard binning or it will be cut.
      untaggedBelowThisValue = DISABLE_UNTAGGED_BIN_VALUE;
  }

  // We don't have to deal with data for this closure test. It's all based around the response data.
  // So we go directly to the response.
  // First, setup the response
  // NOTE: We allocate a shared_ptr, but don't delete here because we want to return the response
  //       without copying. If we do copy, RooUnfold doesn't seem to behave identically. It may not make
  //       a difference, but better not to tempt fate. Instead, we pass ownership to the caller.
  auto response = std::make_shared<RooUnfoldResponse>();
  auto responsenotrunc = std::make_shared<RooUnfoldResponse>();
  response->Setup(hists["h2_smeared"], hists["h2_true"]);
  responsenotrunc->Setup(hists["h2_smeared_no_cuts"], hists["h2_full_eff"]);

  // Next, we setup the Reader, the columns, and store the data in the appropriate hists.
  TChain responseChain(treeNames.response.c_str());
  for (auto filename : inputFilenames.response) {
    responseChain.Add(filename.c_str());
  }
  TTreeReader mcReader(&responseChain);

  // Values
  TTreeReaderValue<float> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<float> responseSmearedJetPt(mcReader, (prefixes.responseSmeared + "_jet_pt").c_str());
  TTreeReaderValue<float> responseSmearedSubstructureVariable(
   mcReader, (settings.groomingMethod + "_" + prefixes.responseSmeared + "_" + settings.substructureVariableName).c_str());
  TTreeReaderValue<float> trueJetPt(mcReader, (prefixes.responseTrue + "_jet_pt").c_str());
  TTreeReaderValue<float> trueSubstructureVariable(
   mcReader, (settings.groomingMethod + "_" + prefixes.responseTrue + "_" + settings.substructureVariableName).c_str());
  // These values are only conditionally defined (ie. they are for embedding, but not for pythia)
  // It's kind of awkward to have a pointer to a TTreeReaderValue, but it's  the only way that I can
  // see to conditionally define them. So I just suck it up.
  std::unique_ptr<TTreeReaderValue<int16_t>> matchingLeading = nullptr;
  std::unique_ptr<TTreeReaderValue<int16_t>> matchingSubleading = nullptr;
  // For the double counting cut.
  std::unique_ptr<TTreeReaderValue<float>> responseSmearedUnsubLeadingTrackPt = nullptr;
  std::unique_ptr<TTreeReaderValue<float>> detLevelLeadingTrackPt = nullptr;
  if (!settings.unfoldingForPP) {
      matchingLeading = std::make_unique<TTreeReaderValue<int16_t>>(mcReader,
                            (settings.groomingMethod + "_hybrid_det_level_matching_leading").c_str());
      matchingSubleading = std::make_unique<TTreeReaderValue<int16_t>>(mcReader,
                              (settings.groomingMethod + "_hybrid_det_level_matching_subleading").c_str());
      // Potentially useful for the double counting cut.
      responseSmearedUnsubLeadingTrackPt = std::make_unique<TTreeReaderValue<float>>(mcReader, (prefixes.responseSmeared + "_leading_track_pt").c_str());
      detLevelLeadingTrackPt = std::make_unique<TTreeReaderValue<float>>(mcReader, (prefixes.responseDetLevel + "_leading_track_pt").c_str());
  }

  int treeNumber = -1;
  // double scaleFactor = 0;
  while (mcReader.Next()) {
    // Check if the file changed.
    if (treeNumber < responseChain.GetTreeNumber()) {
      // File changed. Update the scale factor.
      // auto f = responseChain.GetFile();
      // scaleFactor = GetScaleFactor(f);
      // Update the tree number so we hold onto the scale factor until the next time we need to update.
      treeNumber = responseChain.GetTreeNumber();
    }
    // Ensure that we are in the right true pt (and substructure variable) range.
    if (*trueJetPt < settings.trueJetPtBins.front() ||
        *trueJetPt > settings.trueJetPtBins.back()) {
      continue;
    }
    // To preserve the option to normalize the true substructure variable, we need to store the value here
    double trueSubstructureVariableValue = *trueSubstructureVariable;
    // Normalize by jet pt if requested.
    // NOTE: It's okay to normalize here, even if value is the untagged value because it's less than 0 and
    //       we don't move it around. Thus, dividing by pt will always keep the same sign, and therefore
    //       it will always go into the right position.
    if (settings.normalizeSubstructureVariableByJetPt) {
      trueSubstructureVariableValue /= *trueJetPt;
    }
    // Ensure that we are in the right substructure variable range.
    // NOTE: See the long note around this cut in the standard create_response_2D function,
    //       which explains why this is the right cut given our binning conventions. It's
    //       trickier than would be expected from the first glance.
    if (trueSubstructureVariableValue < settings.trueSplittingVariableBins.front() ||
        trueSubstructureVariableValue > settings.trueSplittingVariableBins.back()) {
      continue;
    }
    // Finish up possible double counting cut selections (other aspects of the cut are implemented earlier)
    if (!settings.unfoldingForPP && doubleCountingCut.useDetLevelTrackPtCut) {
      // NOTE: We use less than because if the leading track pt is the same in both jets (which is most likely
      //       what we would want), then the pt will be equal.
      if (**detLevelLeadingTrackPt < **responseSmearedUnsubLeadingTrackPt) {
        continue;
      }
    }

    // To preserve the option to normalize the smeared substructure variable, we need to store the value here
    double responseSmearedSubstructureVariableValue = *responseSmearedSubstructureVariable;
    if (settings.normalizeSubstructureVariableByJetPt) {
      responseSmearedSubstructureVariableValue /= *responseSmearedJetPt;
    }

    // Full efficiency hists (and response).
    hists["h2_full_eff"]->Fill(trueSubstructureVariableValue, *trueJetPt, *scaleFactor);
    hists["h2_smeared_no_cuts"]->Fill(responseSmearedSubstructureVariableValue, *responseSmearedJetPt, *scaleFactor);
    responsenotrunc->Fill(responseSmearedSubstructureVariableValue, *responseSmearedJetPt, trueSubstructureVariableValue, *trueJetPt,
               *scaleFactor);

    // Now start making cuts on the response smeared level.
    // Jet pt
    if (*responseSmearedJetPt < settings.smearedJetPtBins.front() ||
        *responseSmearedJetPt > settings.smearedJetPtBins.back()) {
      continue;
    }
    // Also cut on hybrid substructure variable.
    if (responseSmearedSubstructureVariableValue < untaggedBelowThisValue) {
      // Assign to the untagged bin.
      responseSmearedSubstructureVariableValue = settings.smearedUntaggedBinValue;
      // If we have the high bin (for the systematic), then we need to normalize in
      // order to match the binning (which will shift depending on whether it is normalized or not).
      // NOTE: This is not necessarily for the lower range, and in fact is preferable
      //       to not touch it to make it easier to detect for pure matches.
      // TODO: As of 27 Jan, we try to skip this. If it doesn't work, return it, but then pure matches needs some help!
      /*if (responseSmearedSubstructureVariableValue > 0) {
        responseSmearedSubstructureVariableValue /= *responseSmearedJetPt;
      }*/
    } else {
      // NOTE: These min and max values take into account the location of the untagged, so they're
      //       filtering the "main" variable range (ie. not including the untagged)
      if (responseSmearedSubstructureVariableValue < settings.minSmearedSplittingVariable ||
        responseSmearedSubstructureVariableValue > settings.maxSmearedSplittingVariable) {
        continue;
      }
    }

    // Potentially Reweight
    // NOTE: We intentionally look at the true values even though it's binned at detector level.
    // We need to handle the binning carefully, so we use a dedicated function.
    int ktBin = findReweightingBin(trueSubstructureVariableValue, settings.smearedSplittingVariableBins);
    int jetPtBin = findReweightingBin(*trueJetPt, settings.smearedJetPtBins);
    // NOTE: In the case of the split MC, both reweight factors will automatically be 1.
    //       However, we set the factor to 1 here just to be safe.
    double reweightFactor =
     closureSettings.variation == ClosureVariation_t::splitMC ? 1 : settings.hReweightingResponse->GetBinContent(ktBin, jetPtBin);

    // The matching cuts should only be applied to the response, so we start storing hists here.
    double randomValue = random.Rndm();
    if (randomValue >= closureSettings.fractionForResponse) {
      // Variation 1 is where we reweight the pseudo-data.
      double pseudoReweightFactor =
       closureSettings.variation == ClosureVariation_t::reweightPseudoData ? reweightFactor : 1;
      hists["h2_pseudo_data"]->Fill(responseSmearedSubstructureVariableValue, *responseSmearedJetPt,
                                    *scaleFactor * pseudoReweightFactor);
      hists["h2_pseudo_true"]->Fill(trueSubstructureVariableValue, *trueJetPt,
                                    *scaleFactor * pseudoReweightFactor);
    }

    // Matching cuts: Requiring a pure match.
    if (!settings.unfoldingForPP &&
         settings.usePureMatches &&
        !isPureMatch(**matchingLeading, **matchingSubleading,
                     responseSmearedSubstructureVariableValue,
                     settings.smearedUntaggedBinValue)
                     // TODO: As of 27 Jan, we try to skip this. If it doesn't work, return it, but then pure matches needs some help!
                     //responseSmearedSubstructureVariableValue < 0 ? settings.smearedUntaggedBinValue : settings.smearedUntaggedBinValue / *responseSmearedJetPt)
      ) {
      continue;
    }

    // Determine the response reweighting factor.
    // Variation 2 is where we reweight the response.
    double responseReweightFactor = closureSettings.variation == ClosureVariation_t::reweightResponse ? reweightFactor : 1;

    // At this point, we've passed all of our cuts, so we store the result.
    // These don't matter so terribly much for our closure test, but it's not a bad thing to have.
    // Note that they will always get the full stats and correspond to the response, as is the convention for the
    // others.
    hists["h2_smeared"]->Fill(responseSmearedSubstructureVariableValue, *responseSmearedJetPt, *scaleFactor * responseReweightFactor);
    hists["h2_true"]->Fill(trueSubstructureVariableValue, *trueJetPt, *scaleFactor * responseReweightFactor);
    hists["h2_substructure_variable"]->Fill(responseSmearedSubstructureVariableValue, trueSubstructureVariableValue,
                        *scaleFactor * responseReweightFactor);

    // We've filled the pseudo data above (before the pure matches requirement), but we still
    // need to fill the response when appropriate.
    if (randomValue < closureSettings.fractionForResponse) {
      response->Fill(responseSmearedSubstructureVariableValue, *responseSmearedJetPt, trueSubstructureVariableValue, *trueJetPt,
              *scaleFactor * responseReweightFactor);
    }
  }

  return ResponseResult{ response, responsenotrunc };
}

namespace older {

//=============================================
// Utility functions, now implemented in python
//=============================================

/**
 * Correlation histogram for the substructure variable.
 *
 * Varies from the pt by the indexing of the covariance matrix.
 *
 * @param[in] cov Covariance matrix derived from the unfolding.
 * @param[in] name Name of the covariance matrix.
 * @param[in] title Title of the covariance matrix.
 * @param[in] na Number of x bins.
 * @param[in] nb Number of y bins.
 * @param[in] kbin Bin in the selected dimension.
 *
 * @returns The correlation histogram.
 */
TH2D* CorrelationHistSubstructureVar(const TMatrixD& cov, const char* name, const char* title, Int_t na, Int_t nb,
                   Int_t kbin)
{
  TH2D* h = new TH2D(name, title, nb, 0, nb, nb, 0, nb);

  for (int l = 0; l < nb; l++) {
    for (int n = 0; n < nb; n++) {
      int index1 = kbin + na * l;
      int index2 = kbin + na * n;
      Double_t Vv = cov(index1, index1) * cov(index2, index2);
      if (Vv > 0.0)
        h->SetBinContent(l + 1, n + 1, cov(index1, index2) / sqrt(Vv));
    }
  }
  return h;
}

/**
 * Correlation histogram for the substructure variable.
 *
 * Varies from the substructure variable by the indexing of the covariance matrix.
 *
 * @param[in] cov Covariance matrix derived from the unfolding.
 * @param[in] name Name of the covariance matrix.
 * @param[in] title Title of the covariance matrix.
 * @param[in] na Number of x bins.
 * @param[in] nb Number of y bins.
 * @param[in] kbin Bin in the selected dimension.
 *
 * @returns The correlation histogram.
 */
TH2D* CorrelationHistPt(const TMatrixD& cov, const char* name, const char* title, Int_t na, Int_t nb, Int_t kbin)
{
  TH2D* h = new TH2D(name, title, na, 0, na, na, 0, na);

  for (int l = 0; l < na; l++) {
    for (int n = 0; n < na; n++) {
      int index1 = l + na * kbin;
      int index2 = n + na * kbin;
      Double_t Vv = cov(index1, index1) * cov(index2, index2);
      if (Vv > 0.0)
        h->SetBinContent(l + 1, n + 1, cov(index1, index2) / sqrt(Vv));
    }
  }
  return h;
}

/**
 * Calculate the correlation matrix for a given set of properties.
 *
 * As of October 2020, it's not used.
 *
 * @param[in] cov Covariance matrix derived from the unfolding.
 * @param[in] name Name of the covariance matrix.
 * @param[in] title Title of the covariance matrix.
 * @param[in] lo Lower bin limit.
 * @param[in] hi Higher bin limit.
 *
 * @returns The correlation histogram.
 */
TH2D* CorrelationHist(const TMatrixD& cov, const char* name, const char* title, Double_t lo,
           Double_t hi /*, Double_t lon, Double_t hin*/)
{
  Int_t nb = cov.GetNrows();
  Int_t na = cov.GetNcols();
  std::cout << nb << " " << na << "\n";
  TH2D* h = new TH2D(name, title, nb, 0, nb, na, 0, na);
  h->SetAxisRange(-1.0, 1.0, "Z");
  for (int i = 0; i < na; i++) {
    for (int j = 0; j < nb; j++) {
      Double_t Viijj = cov(i, i) * cov(j, j);
      if (Viijj > 0.0)
        h->SetBinContent(i + 1, j + 1, cov(i, j) / sqrt(Viijj));
    }
  }
  return h;
}

/**
 * Normalize a given 2D histogram.
 *
 * This is something that ROOT won't do itself...
 *
 * @param[in, out] h Histogram to be normalized.
 */
void Normalize2D(TH2* h)
{
  Int_t nbinsYtmp = h->GetNbinsY();
  const Int_t nbinsY = nbinsYtmp;
  Double_t norm[nbinsY];
  for (Int_t biny = 1; biny <= nbinsY; biny++) {
    norm[biny - 1] = 0;
    for (Int_t binx = 1; binx <= h->GetNbinsX(); binx++) {
      norm[biny - 1] += h->GetBinContent(binx, biny);
    }
  }

  for (Int_t biny = 1; biny <= nbinsY; biny++) {
    for (Int_t binx = 1; binx <= h->GetNbinsX(); binx++) {
      if (norm[biny - 1] == 0)
        continue;
      else {
        h->SetBinContent(binx, biny, h->GetBinContent(binx, biny) / norm[biny - 1]);
        h->SetBinError(binx, biny, h->GetBinError(binx, biny) / norm[biny - 1]);
      }
    }
  }
}

/**
 * Determine pt hard scale factor of the currently open file.
 *
 * This should only be necessary as a cross check, or when the scaling information isn't stored directly
 * in the tree (ie. if it's not from a skim).
 *
 * @param[in] f Current file.
 *
 * @returns Pt hard scale factor.
 */
double GetScaleFactor(TFile* f)
{
  // Retrieve the embedding helper to extract the cross section and ntrials.
  // auto task =
  // dynamic_cast<TList*>(f->Get("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_Raw_EventSub_Incl"));
  auto task = dynamic_cast<TList*>(f->Get("AliAnalysisTaskEmcalEmbeddingHelper_histos"));

  auto hcross = dynamic_cast<TProfile*>(task->FindObject("fHistXsection"));
  auto htrials = dynamic_cast<TH1D*>(task->FindObject("fHistTrials"));

  // std::cout << "hcross=" << hcross << ", htrials=" << htrials << "\n";

  int ptHardBin = 0;
  for (Int_t i = 1; i <= htrials->GetXaxis()->GetNbins(); i++) {
    if (htrials->GetBinContent(i) != 0) {
      ptHardBin = i;
    }
  }
  // Code adapted from Leticia.
  // The values don't seem to be correct...
  /*double scaleFactor = hcross->Integral(ptHardBin, ptHardBin) / htrials->Integral(ptHardBin, ptHardBin);
  // Needs a -1 to offset for the bin counting.
  std::cout << f->GetName() << ": ptHardBin=" << (ptHardBin - 1) << ", scaleFactor=" << scaleFactor;*/

  // These values seem more reasonable...
  double crossSection = hcross->GetBinContent(ptHardBin) * hcross->GetEntries();
  double nTrials = htrials->GetBinContent(ptHardBin);
  double scaleFactor = crossSection / nTrials;
  std::cout << f->GetName() << ": ptHardBin=" << (ptHardBin - 1) << ", scaleFactor=" << scaleFactor << "\n";

  return scaleFactor;
}

//=========================
// Main functionality
//=========================

/**
 * Perform the actual unfolding.
 *
 * We separate this functionality so that we can run the unfolding with different input spectra for the same
 * response, such as the when we used the hybrid as the smeared input as a trivial closure.
 *
 * We also calculate the covariance matrix at a specified iteration.
 *
 * @param[in] response Roounfold response.
 * @param[in] h2true True spectra. Only used for binning, so retrieved as a const.
 * @param[in] inputSpectra Input spectra to be unfolded (for example, data). Spectra could be 2D: pt + kt (or Rg,
 * etc...)
 * @param[in] errorTreatment Roounfold error treatment.
 * @param[in] fout ROOT output file. It's never called explicitly, but I like to pass it because we're implicitly
 * writing to it.
 * @param[in] tag Tag to be prepended to all histograms generated in the unfolding. Default: "".
 * @param[in] nIter Number of iterations. Default: 10.
 * @param[in] nIterForCovariance Number of iterations where we will calculate the covariance. Default: 8.
 */
void Unfold2D(RooUnfoldResponse& response, const TH2D& h2true, TH2D& inputSpectra,
       RooUnfold::ErrorTreatment errorTreatment, TFile* fout, std::string tag = "", const int nIter = 10,
       const int nIterForCovariance = 8)
{
  std::cout << "\n=======================================================\n"
       << "Unfolding for tag \"" << tag << "\".\n";
  // Determine the tag. If we have a non-empty tag, we append it to all of the histograms.
  if (tag != "") {
    tag += "_";
  }

  for (int iter = 1; iter < nIter; iter++) {
    std::cout << "Iteration " << iter << "\n";

    // Setup the response for unfolding.
    RooUnfoldBayes unfold(&response, &inputSpectra, iter);
    // And then unfold.
    TH2D* hunf = dynamic_cast<TH2D*>(unfold.Hreco(errorTreatment));

    // Refold the truth (ie. fold back).
    TH1* hfold = response.ApplyToTruth(hunf, "");

    // Clone unfolded and refolded hists to write to the output file.
    TH2D* htempUnf =
     dynamic_cast<TH2D*>(hunf->Clone(TString::Format("%sBayesian_Unfoldediter%d", tag.c_str(), iter)));
    TH2D* htempFold =
     dynamic_cast<TH2D*>(hfold->Clone(TString::Format("%sBayesian_Foldediter%d", tag.c_str(), iter)));
    htempUnf->Write();
    htempFold->Write();

    // Retrieve the covariance matrix. Only for a selected iteration.
    if (iter == nIterForCovariance) {
      TMatrixD covmat = unfold.Ereco(RooUnfold::kCovariance);
      // Substructure variable.
      for (Int_t k = 0; k < h2true.GetNbinsX(); k++) {
        auto hCorr = dynamic_cast<TH2D*>(
         CorrelationHistSubstructureVar(covmat, TString::Format("%scorr%d", tag.c_str(), k),
                         "Covariance matrix", h2true.GetNbinsX(), h2true.GetNbinsY(), k));
        auto covSubstructureVar = dynamic_cast<TH2D*>(hCorr->Clone("covSubstructureVar"));
        covSubstructureVar->SetName(
         TString::Format("%spearsonmatrix_iter%d_binSubstructureVar%d", tag.c_str(), iter, k));
        covSubstructureVar->SetDrawOption("colz");
        covSubstructureVar->Write();
      }

      // Jet pt.
      for (Int_t k = 0; k < h2true.GetNbinsY(); k++) {
        auto hCorr = dynamic_cast<TH2D*>(
         CorrelationHistPt(covmat, TString::Format("%scorr%dpt", tag.c_str(), k), "Covariance matrix",
                  h2true.GetNbinsX(), h2true.GetNbinsY(), k));
        auto covpt = dynamic_cast<TH2D*>(hCorr->Clone("covpt"));
        covpt->SetName(TString::Format("%spearsonmatrix_iter%d_binpt%d", tag.c_str(), iter, k));
        covpt->SetDrawOption("colz");
        covpt->Write();
      }
    }
  }

  std::cout << "Finished unfolding!\n"
       << "=======================================================\n";
}


} /* namespace older */

} /* namespace unfolding */

