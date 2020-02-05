






/// \file runJetSubstructure.C
///
/// \ingroup EMCALJETFW
/// The script runJetSubstructure.sh in the same folder allows to easily set
/// the input values
///
/// \author Laura Havener <laura.brittany.havener@cern.ch>, Yale University
/// \date July 8, 2019

/** this comes from http://alitrain.cern.ch/train-workdir/PWGJE/Jets_EMC_pp_MC/1694_20190202-1732/merge_runlist_1/MLTrainDefinition.cfg
  * Markus' wagon configuration file for the 2015 MC pp
  *                                                                                                            
  */

class AliESDInputHandler;
class AliAODInputHandler;
class AliVEvent;
class AliAnalysisGrid;
class AliAnalysisManager;
class AliAnalysisAlien;
class AliPhysicsSelectionTask;
class AliMultSelectionTask;
class AliCentralitySelectionTask;
class AliTaskCDBconnect;
class AliAnalysisTaskNewJetSubstructure;

class AliClusterContainer;
class AliParticleContainer;
class AliJetContainer;

class AliAnalysisTaskRhoMass;
class AliAnalysisTaskEmcalEmbeddingHelper;
class AliEmcalCorrectionTask;
class AliEmcalJetTask;
//class AliPHOSTenderTask;

namespace PWGJE {
  namespace EMCALJetTasks {
     class AliEmcalJetTaggerTaskFast;
    class AliAnalysisTaskEmcalSoftDropResponse;
  }
}



// Include AddTask macros for ROOT 6 compatibility
#ifdef __CLING__
// Tell ROOT where to find AliRoot headers
R__ADD_INCLUDE_PATH($ALICE_ROOT)
// Tell ROOT where to find AliPhysics headers
R__ADD_INCLUDE_PATH($ALICE_PHYSICS)
#include "OADB/macros/AddTaskPhysicsSelection.C"
#include "OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C"
#include "OADB/macros/AddTaskCentrality.C"
#include "PWGPP/PilotTrain/AddTaskCDBconnect.C"
#include "PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C"
#include "PWGJE/EMCALJetTasks/macros/AddTaskRhoNew.C"
#include "PWGJE/EMCALJetTasks/macros/AddTaskRhoMass.C"
#include "PWGJE/EMCalJetTasks/macros/AddTaskEmcalSoftdropResponse.C"
#include "PWGJE/EMCALJetTasks/macros/AddTaskNewJetSubstructure.C"

//#include "PWGJE/EMCALJetTasks/macros/AddAODPHOSTender.C"
#endif

void LoadMacros();
void StartGridAnalysis(AliAnalysisManager* pMgr, const char* uniqueName, const char* cGridMode);
AliAnalysisGrid* CreateAlienHandler(const char* uniqueName, const char* gridDir, const char* gridMode, const char* runNumbers,
    const char* pattern, TString additionalCode, TString additionalHeaders, Int_t maxFilesPerWorker, Int_t workerTTL, Bool_t isMC);

//______________________________________________________________________________
AliAnalysisManager* runJetSubstructure(
				       const char   *cDataType      = "AOD",                                   // set the analysis type, AOD or ESD
				       const char   *cRunPeriod     = "LHC11h",                                // set the run period
				       const char   *cLocalFiles    = "files_test.txt",   // set the local list file
				       const UInt_t  iNumEvents     = 1000,                                    // number of events to be analyzed
				       //				       const UInt_t  kPhysSel       = AliVEvent::kAnyINT |
				       //				           AliVEvent::kCentral | AliVEvent::kSemiCentral,                          // physics selection
				       //const UInt_t kPhysSel        = 0,                                      // for MC!
				       const UInt_t kPhysSel        = AliVEvent::kCentral,
				       const char   *cTaskName      = "EMCalEmbeddingDataExtractor",                           // sets name of analysis manager
				       //				       const Bool_t  bDoChargedJets = kTRUE,
				       //				       const Bool_t  bDoFullJets    = kFALSE,
				       const char   *obsolete       = "",                                      // Previous handled the ocdb settings, but obsolete due to CDBconnect task
				       // 0 = only prepare the analysis manager but do not start the analysis
				       // 1 = prepare the analysis manager and start the analysis
				       // 2 = launch a grid analysis
				       Int_t         iStartAnalysis = 1,
				       const UInt_t  iNumFiles      = 5,                                     // number of files analyzed locally
				       const char   *cGridMode      = "test"
				       )
{

  // Debug configurations                                                                                                                                                                             
  //  AliLog::SetClassDebugLevel("AliAnalysisTaskEmcalEmbeddingHelper", AliLog::kDebug+2);                                                                                                              
  //  AliLog::SetClassDebugLevel("AliAnalysisTaskRho", AliLog::kDebug+1);                                                                                                                               
  //  AliLog::SetClassDebugLevel("AliEmcalCorrectionTask", AliLog::kDebug+1);                                                                                                                           
  //  AliLog::SetClassDebugLevel("AliEmcalJetTask", AliLog::kDebug+1);    
  //  AliLog::SetClassDebugLevel("AliEmcalJetTaggerTaskFast",  AliLog::kDebug+2);
  //  AliLog::SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalSoftDropResponse", 4);
  // Common settings 
  const Double_t minClusterPt = 0.30;
  const Double_t minTrackPt = 0.15;
  const Double_t kGhostArea = 0.005;
  const UInt_t kComPhysSel = AliVEvent::kCentral;

  //setup period
  TString sRunPeriod(cRunPeriod);
  sRunPeriod.ToLower();

  // set Run 2
  Bool_t bIsRun2 = kFALSE;
  if ((sRunPeriod.Length() == 6) && ((sRunPeriod.BeginsWith("lhc15"))||(sRunPeriod.BeginsWith("lhc18"))||(sRunPeriod.BeginsWith("lhc19")))) bIsRun2 = kTRUE;

  cout << "run2: " << bIsRun2 << endl;

  //set beam type
  AliAnalysisTaskEmcal::BeamType iBeamType = AliAnalysisTaskEmcal::kpp;
  if (sRunPeriod == "lhc10h" || sRunPeriod == "lhc11h" || sRunPeriod == "lhc15o" || sRunPeriod == "lhc18r" || sRunPeriod == "lhc18q" || sRunPeriod == "lhc18l8a4") {
    iBeamType = AliAnalysisTaskEmcal::kAA;
  }
  else if (sRunPeriod == "lhc12g" || sRunPeriod == "lhc13b" || sRunPeriod == "lhc13c" ||
      sRunPeriod == "lhc13d" || sRunPeriod == "lhc13e" || sRunPeriod == "lhc13f" ||
      sRunPeriod == "lhc16q" || sRunPeriod == "lhc16r" || sRunPeriod == "lhc16s" ||
      sRunPeriod == "lhc16t") {
    iBeamType = AliAnalysisTaskEmcal::kpA;
    }

  //ghost area
  //  Double_t kGhostArea = 0.01;
  //  if (iBeamType != AliAnalysisTaskEmcal::kpp) kGhostArea = 0.005;

  //setup track container
  //  AliTrackContainer::SetDefTrackCutsPeriod(sRunPeriod);
  AliTrackContainer::SetDefTrackCutsPeriod("lhc15o");
  Printf("Default track cut period set to: %s", AliTrackContainer::GetDefTrackCutsPeriod().Data());

  //set data file type
  enum eDataType { kAod, kEsd };

  eDataType iDataType;
  if (!strcmp(cDataType, "ESD")) {
    iDataType = kEsd;
  }
  else if (!strcmp(cDataType, "AOD")) {
    iDataType = kAod;
  }
  else {
    Printf("Incorrect data type option, check third argument of run macro.");
    Printf("datatype = AOD or ESD");
    return 0;
    }

  Printf("%s analysis chosen.", cDataType);

  TString sLocalFiles(cLocalFiles);
  if (iStartAnalysis == 1) {
    if (sLocalFiles == "") {
      Printf("You need to provide the list of local files!");
      return 0;
    }
    Printf("Setting local analysis for %d files from list %s, max events = %d", iNumFiles, sLocalFiles.Data(), iNumEvents);
  }

  //load macros needed for the analysis
  #ifndef __CLING__
  LoadMacros();
  #endif

  ////////////////////////                                                                                                                              // Configuration options                                                                                                                            
  ////////////////////////                                                                                                                             
  // Use full or charged jets                                                                                                       
  const bool fullJets =false;
  const bool enableBackgroundSubtraction =true;
  // Embedding files list                                                                                                                                              
  const std::string embeddedFilesList = "LHC19f4_1AOD.txt"; //"LHC12a15e_fixAOD.txt";                                                                                     // If true, events that are not selected in the PbPb will not be used for embedding.                                                                                    // This ensures that good embedded events are not wasted on bad PbPb events.                                                                                      
  const bool internalEventSelection = true;


  // General track and cluster cuts (used particularly for jet finding)                                                             
  const Double_t kTrackPtCut = 0.15;
  const Double_t kClusPtCut = 0.3;

  AliEmcalJet::JetAcceptanceType acceptanceType = AliEmcalJet::kTPCfid;
  const std::string acceptanceTypeStr = "TPCfid";

  // Determine track, cluster, and cell names                                                                                                        
  const bool IsEsd = (iDataType == kEsd);
 
  TString tracksName = AliEmcalContainerUtils::DetermineUseDefaultName(AliEmcalContainerUtils::kTrack, IsEsd);
  TString emcalCellsName = AliEmcalContainerUtils::DetermineUseDefaultName(AliEmcalContainerUtils::kCaloCells, IsEsd);
  TString emcalCellsNameCombined = emcalCellsName + "Combined";
  TString clustersName = AliEmcalContainerUtils::DetermineUseDefaultName(AliEmcalContainerUtils::kCluster, IsEsd);
  TString clustersNameCombined = clustersName + "Combined";
 
  // Handle charged jets                                                                                                                             
  if (fullJets == false) {
    emcalCellsName = "";
    emcalCellsNameCombined = "";
    clustersName = "";
    clustersNameCombined = "";
  }

  // Analysis manager
  AliAnalysisManager* pMgr = new AliAnalysisManager(cTaskName);
  
  // Create Input Handler                                                                                                                            
  if (iDataType == kAod) {
    AliAODInputHandler * pESDHandler = AliAnalysisTaskEmcal::AddAODHandler();
  }
  else {
    AliESDInputHandler * pESDHandler = AliAnalysisTaskEmcal::AddESDHandler();
  }
  
  // Physics selection task
  if (iDataType == kEsd) {
    AliPhysicsSelectionTask *pPhysSelTask = AddTaskPhysicsSelection();
  }

  // Centrality task
  // The Run 2 condition is too restrictive, but until the switch to MultSelection is complete, it is the best we can do                              
  //  if (iDataType == kEsd && iBeamType != AliAnalysisTaskEmcal::kpp && bIsRun2 == kFALSE) {
  if (iDataType == kEsd && iBeamType != AliAnalysisTaskEmcal::kpp) {
    AliCentralitySelectionTask *pCentralityTask = AddTaskCentrality(kTRUE);
    pCentralityTask->SelectCollisionCandidates(AliVEvent::kAny);
  }

  // AliMultSelection                                                                                                                                    // Works for pp, pPb, and PbPb for the periods that it is calibrated                                                                             
  if (bIsRun2 == kTRUE) {
    AliMultSelectionTask * pMultSelectionTask = AddTaskMultSelection(kFALSE);
    pMultSelectionTask->SelectCollisionCandidates(kComPhysSel);
  }

  // CDBconnect task                                                                                                                               
  if (fullJets) {
    AliTaskCDBconnect * taskCDB = AddTaskCDBconnect();
    taskCDB->SetFallBackToRaw(kTRUE);
  }

  //  AliPHOSTenderTask * taskPHOS = AddAODPHOSTender();

  // Setup embedding task                                                                                                                                               
  AliAnalysisTaskEmcalEmbeddingHelper * embeddingHelper = AliAnalysisTaskEmcalEmbeddingHelper::AddTaskEmcalEmbeddingHelper();
  embeddingHelper->SelectCollisionCandidates(kPhysSel);
  // The pt hard bin should be set via the filenames in this file                                                                                                         // If using a file pattern, it could be configured via embeddingHelper->SetPtHardBin(ptHardBin);                                                                      
  embeddingHelper->SetFileListFilename(embeddedFilesList.c_str());

  // Some example settings for LHC12a15e_fix (anchored to LHC11h)                                                                                                       
  embeddingHelper->SetNPtHardBins(20);
  embeddingHelper->SetMCRejectOutliers();
  embeddingHelper->SetTriggerMask(0); // $$ Change for train run copy                                                                                                   
  embeddingHelper->SetConfigurationPath("alien:///alice/cern.ch/user/l/lhavener/embeddingHelper_LHC18_LHC19f4_kCentral.yaml"); // $$ Change for train run copy

  // Initialize the task to complete the setup.                                                                                                                         
  embeddingHelper->Initialize();

  if (fullJets) {
    // EMCal corrections                                                                                                                                             
    TObjArray correctionTasks;
    // Create the Correction Tasks                                                                                                                                      // "data" corresponds to the PbPb level                                                                                                                            
    // "embed" corresponds to the embedded detector level                                                                                                            
    // "combined" corresponds to the hybrid (PbPb + embedded detector) level                                                                                            
    correctionTasks.Add(AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask("data"));
    correctionTasks.Add(AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask("embed"));
    // It is important that combined is last!                                                                                                                           
    correctionTasks.Add(AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask("combined"));

    // Loop over all of the correction tasks to configure them                                                                                                          
    AliEmcalCorrectionTask * tempCorrectionTask = 0;
    TIter next(&correctionTasks);
    while (( tempCorrectionTask = static_cast<AliEmcalCorrectionTask *>(next())))
      {
	tempCorrectionTask->SelectCollisionCandidates(kPhysSel);
	// Configure centrality                                                                                                                                           
	tempCorrectionTask->SetNCentBins(5);
	if (bIsRun2) {
	  tempCorrectionTask->SetUseNewCentralityEstimation(kTRUE);
	}
	tempCorrectionTask->SetUserConfigurationFilename("emcalCorrections_LHC15o_LHC16j5_V2.yaml");
	tempCorrectionTask->Initialize();
      }
  }


  // Background                                                                                                                                        
  std::string sRhoChargedName = "";
  std::string sRhoFullName = "";
  if (iBeamType != AliAnalysisTaskEmcal::kpp && enableBackgroundSubtraction == true) {
    const AliJetContainer::EJetAlgo_t rhoJetAlgorithm = AliJetContainer::kt_algorithm;
    const AliJetContainer::EJetType_t rhoJetType = AliJetContainer::kChargedJet;
    const AliJetContainer::ERecoScheme_t rhoRecoScheme = AliJetContainer::E_scheme;
    const double rhoJetRadius = 0.2;
    sRhoChargedName = "Rho";
    sRhoFullName = "Rho_Scaled";

    AliEmcalJetTask * pJetFinder_charged_KT_02 = AliEmcalJetTask::AddTaskEmcalJet("usedefault", "", rhoJetAlgorithm, rhoJetRadius, rhoJetType, kTrackPtCut, kClusPtCut, kGhostArea, rhoRecoScheme, "Jet", 0., kFALSE, kFALSE);
    pJetFinder_charged_KT_02->SetUseNewCentralityEstimation(kTRUE);
    pJetFinder_charged_KT_02->SetNCentBins(5);
    pJetFinder_charged_KT_02->SelectCollisionCandidates(kComPhysSel);
    pJetFinder_charged_KT_02->SetForceBeamType(AliAnalysisTaskEmcal::kAA);
    
    AliAnalysisTaskRho * pRhoTask = AddTaskRhoNew("usedefault", "usedefault", sRhoChargedName.c_str(), rhoJetRadius, AliJetContainer::kTPCfid,AliJetContainer::kChargedJet,kTRUE,AliJetContainer::E_scheme);
    pRhoTask->SetExcludeLeadJets(2);
    pRhoTask->SetNCentBins(5);
    pRhoTask->SelectCollisionCandidates(kComPhysSel);
    pRhoTask->SetUseNewCentralityEstimation(kTRUE);
    pRhoTask->SetForceBeamType(AliAnalysisTaskEmcal::kAA);

    AliAnalysisTaskRho * pRhoTaskScaled = AddTaskRhoNew("usedefault", "usedefault", sRhoFullName.c_str(), rhoJetRadius, AliJetContainer::kTPCfid,AliJetContainer::kChargedJet,kTRUE,AliJetContainer::E_scheme);
    pRhoTaskScaled->SetExcludeLeadJets(2);
    pRhoTaskScaled->SetNCentBins(5);
    pRhoTaskScaled->SelectCollisionCandidates(kComPhysSel);
    pRhoTaskScaled->SetUseNewCentralityEstimation(kTRUE);
    pRhoTaskScaled->SetForceBeamType(AliAnalysisTaskEmcal::kAA);
    
    if (fullJets)
      {
	TString sFuncPath = "alien:///alice/cern.ch/user/l/lhavener/scaleFactorLHC18q_PtDep.root";
	TString sFuncName = "fScaleFactorEMCal";
	pRhoTaskScaled->LoadRhoFunction(sFuncPath, sFuncName);
      }
  }

  //the rho mass which I might need
  AliAnalysisTaskRhoMass* pRhoMassTask = AddTaskRhoMass("Jet_KTChargedR020_tracks_pT0150_E_scheme","tracks","","Rhomass",0.2,"TPC",0.01,0,0,2,kTRUE,"RhoMass");
  pRhoMassTask->SelectCollisionCandidates(kComPhysSel);
  pRhoMassTask->SetNCentBins(5);
  pRhoMassTask->SetUseNewCentralityEstimation(kTRUE);
  pRhoMassTask->SetForceBeamType(AliAnalysisTaskEmcal::kAA);
  pRhoMassTask->SetHistoBins(250,0,250);
  AliJetContainer *cont = pRhoMassTask->GetJetContainer(0);
  cont->SetJetRadius(0.2);
  cont->SetJetAcceptanceType(AliJetContainer::kTPCfid);
  cont->SetMaxTrackPt(100);
  
  
  // Jet finding                                                                                                                                   
  const AliJetContainer::EJetAlgo_t jetAlgorithm = AliJetContainer::antikt_algorithm;
  const Double_t jetRadius = 0.4;
  AliJetContainer::EJetType_t jetType = AliJetContainer::kFullJet;
  const AliJetContainer::ERecoScheme_t recoScheme = AliJetContainer::E_scheme;
  const char * label = "Jet";
  const Double_t minJetPt = 1;
  const Bool_t lockTask = kFALSE;
  const Bool_t fillGhosts = kFALSE;

  // Do not pass clusters if we are only looking at charged jets                                                                                     
  if (fullJets == false) {
    jetType = AliJetContainer::kChargedJet;
  }

  ///////                                                                                                                                                            
  // Particle level PYTHIA jet finding
  ///////                                                                                                                                                              

  AliEmcalJetTask * pFullJetTaskPartLevel = AliEmcalJetTask::AddTaskEmcalJet("mcparticles", "", jetAlgorithm, jetRadius, jetType, 0, minClusterPt, kGhostArea, recoScheme, "partLevelJets", 0.1, lockTask, fillGhosts); //$$         
  pFullJetTaskPartLevel->SelectCollisionCandidates(kPhysSel);
  pFullJetTaskPartLevel->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  pFullJetTaskPartLevel->SetForceBeamType(AliAnalysisTaskEmcal::kAA); //$$                                                                                              
  pFullJetTaskPartLevel->SetUseNewCentralityEstimation(kTRUE);

  ///////                                                                                                                                                             
  // External event (called embedding) settings for particle level PYTHIA jet finding                                                                                 
  ///////                                                                                                                                                               
  // Setup the tracks properly to be retrieved from the external event                                                                                                
  // It does not matter here if it's a Particle Container or MCParticleContainer                                                                                        
  AliParticleContainer * partLevelTracks = pFullJetTaskPartLevel->GetMCParticleContainer(0);
  // Called Embedded, but really just means get from an external event!                                                                                                 
  partLevelTracks->SetIsEmbedding(kTRUE);

  ///////                                                                                                                                                             
  // Detector level PYTHIA jet finding                                                                                                                               
  ///////                                                                                                                                                              

  AliEmcalJetTask * pFullJetTaskDetLevel = AliEmcalJetTask::AddTaskEmcalJet(tracksName.Data(), clustersName.Data(),								    jetAlgorithm, jetRadius, jetType, minTrackPt, minClusterPt, kGhostArea, recoScheme, "detLevelJets", 0.1, lockTask, fillGhosts);
  pFullJetTaskDetLevel->SelectCollisionCandidates(kPhysSel);
  pFullJetTaskDetLevel->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  pFullJetTaskDetLevel->SetForceBeamType(AliAnalysisTaskEmcal::kAA); //$$                                                                                               
  pFullJetTaskDetLevel->SetUseNewCentralityEstimation(kTRUE);

  ///////                                                                                                                                                            
  // External event (embedding) settings for det level PYTHIA jet finding                                                                                            
  ///////                                                                                                                                                            

  // Tracks                                                                                                                                                          
  // Uses the name of the container passed into AliEmcalJetTask                                                                                                       
  AliTrackContainer * tracksDetLevel = pFullJetTaskDetLevel->GetTrackContainer(0);
  // Get the det level tracks from the external event!                                                                                                               
  tracksDetLevel->SetIsEmbedding(kTRUE);

  // Clusters                                                                                                                                                         
  if (fullJets) {
    // Uses the name of the container passed into AliEmcalJetTask                                                                                                     
    AliClusterContainer * clustersDetLevel = pFullJetTaskDetLevel->GetClusterContainer(0);
    // Get the det level clusters from the external event!                                                                                                           
    clustersDetLevel->SetIsEmbedding(kTRUE);
    // Additional configuration                                                                                                                                
    //clustersDetLevel->SetDefaultClusterEnergy(AliVCluster::kHadCorr);                                                                                              
    clustersDetLevel->SetClusTimeCut(-50e-9,100e-9); //$$                                                                                                            
  }


  ///////                                                                                                                                                             
  // Hybrid (PbPb + Detector) level PYTHIA jet finding                                                                                                                
  ///////                                                                                                                                                               
  // Sets up PbPb tracks and clusters                                                                                                                                  
  // NOTE: The clusters name is different here since we output to a different branch!                                                                                  
  AliEmcalJetTask * pFullJetTaskHybrid = AliEmcalJetTask::AddTaskEmcalJet(tracksName.Data(), clustersNameCombined.Data(), jetAlgorithm, jetRadius, jetType, minTrackPt, minClusterPt, kGhostArea, recoScheme, "hybridLevelJets", 0.1, lockTask, fillGhosts);
  pFullJetTaskHybrid->SelectCollisionCandidates(kPhysSel);
  pFullJetTaskHybrid->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  pFullJetTaskHybrid->SetForceBeamType(AliAnalysisTaskEmcal::kAA); //$$                                                                                                 
  pFullJetTaskHybrid->SetUseNewCentralityEstimation(kTRUE);
  // artificial tracking efficiency $$                                                                                                                                  
  pFullJetTaskHybrid->SetTrackEfficiency(.99);
  pFullJetTaskHybrid->SetTrackEfficiencyOnlyForEmbedding(kTRUE);

  AliEmcalJetUtilityConstSubtractor* constUtil = (AliEmcalJetUtilityConstSubtractor *)pFullJetTaskHybrid->AddUtility(new AliEmcalJetUtilityConstSubtractor("ConstSubtractor"));
  AliEmcalJetUtilityGenSubtractor* genSub = (AliEmcalJetUtilityGenSubtractor *)pFullJetTaskHybrid->AddUtility(new AliEmcalJetUtilityGenSubtractor("GenSubtractor"));

  genSub->SetGenericSubtractionJetMass(kTRUE);
  genSub->SetGenericSubtractionExtraJetShapes(kTRUE);
  genSub->SetUseExternalBkg(kTRUE);
  genSub->SetRhoName("Rho");
  genSub->SetRhomName("Rhomass");
  constUtil->SetJetsSubName(Form("%sConstSub",pFullJetTaskHybrid->GetName()));
  constUtil->SetParticlesSubName("tracksSubR02");
  constUtil->SetUseExternalBkg(kTRUE);
  constUtil->SetRhoName("Rho");
  constUtil->SetRhomName("Rhomass");

  if (fullJets) {
    pFullJetTaskHybrid->GetClusterContainer(0)->SetDefaultClusterEnergy(AliVCluster::kHadCorr);
    pFullJetTaskHybrid->SetClusTimeCut(-50e-9, 100e-9);
  }
  
  ///////                                                                                                                                                             
  // External event (ie embedding) settings for PbPb jet finding (adds detector level PYTHIA)                                                                         
  ///////                                                                                                                                                             

  // Add embedded tracks and clusters to jet finder                                                                                                                   
  // Tracks                                                                                                                                                           
  AliTrackContainer * tracksEmbedDetLevel = new AliTrackContainer(tracksName.Data());
  // Get the det level tracks from the external event!                                                                                                               
  tracksEmbedDetLevel->SetIsEmbedding(kTRUE);
  //tracksEmbedDetLevel->SetParticlePtCut(3.0);                                                                                                                      
  pFullJetTaskHybrid->AdoptTrackContainer(tracksEmbedDetLevel);
  // Clusters                                                                                                                                                        
  // Already combined in clusterizer, so we shouldn't add an additional cluster container here  


    // jet tagger 

  Bool_t kIsPtHard = false; 
  Int_t knPthardBins = 0;
  TArrayI kPtHardBinning;
  Bool_t kOldPtHardBinHandling = false;  // Special handling for old pt-hard productions which are handled as separate datasets
  const char* kDataset = "LHC18f5";
  AliTrackContainer::SetDefTrackCutsPeriod(kDataset);
  kIsPtHard=kTRUE;
  knPthardBins=21;
  kPtHardBinning.Set(22);
  Int_t binning[]={0,5, 7, 9, 12, 16, 21, 28, 36, 45, 57, 70, 85, 99, 115, 132, 150, 169, 190, 212, 235,1000};
  for(Int_t bin=0;bin<22;bin++){
    kPtHardBinning[bin]=binning[bin];}


  // Hybrid (PbPb + embed) jets are the "base" jet collection                                                                                                         
  const std::string hybridLevelJetsName = pFullJetTaskHybrid->GetName();
  // Embed det level jets are the "tag" jet collection                                                                                                                
  const std::string detLevelJetsName = pFullJetTaskDetLevel->GetName();
  // PbPb jets                                                                                                                                                        
  //const std::string backgroundJetsName = pFullJetTask->GetName();                                                                                                   
  // Centrality estimotor                                                                                                                                             
  const std::string centralityEstimator = "V0M";
  // Jet Matching Tasks                                                                                                                                                                               
  // 0.3 is the default max matching distance from the EMCal Jet Tagger                                                                                                                             
  double maxGeoMatchingDistance = 0.3;
  double fractionSharedMomentum = 0.5;


  // NOTE: The AddTask macro is "AddTaskEmcalJetTaggerFast" ("Emcal" is removed for the static definition...)                                                         
  PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast * jetTaggerDetLevel = PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::AddTaskJetTaggerFast(	       
	 hybridLevelJetsName.c_str(),     // "Base" jet collection which will be tagged ********************** changed to make same containers       
	 detLevelJetsName.c_str(),       // "Tag" jet collection which will be used to tag (and will be tagged)                                                      
	 jetRadius,                      // Jet radius                                                                                                               
	 "Rho",                             // Hybrid ("base") rho name                                                        
	 "",                             // Det level ("tag") rho name                                                                                 
	 "",                             // tracks to attach to the jet containers. Not meaningful here, so left empty                                               
	 "",                             // clusters to attach to the jet conatiners. Not meaingful here, so left empty (plus, it's not the same for the two jet collections!)                                                                                                                                                              
	 acceptanceTypeStr.c_str(),      // Jet acceptance type for the "base" collection                                                                           
	 centralityEstimator.c_str(),    // Centrality estimator                                                                                                    
	 kPhysSel,                       // Physics selection                                                                                                        
	 ""                              // Trigger class. We can just leave blank, as it's only used in the task name                                               
																		       );

  // Task level settings                                                                                                                                              
  jetTaggerDetLevel->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  // Tag via geometrical matching                                                                                                                                     
  jetTaggerDetLevel->SetJetTaggingMethod(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kGeo);
  // Tag the closest jet                                                                                                                                              
  jetTaggerDetLevel->SetJetTaggingType(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kClosest);
  // Don't impose any additional acceptance cuts beyond the jet containers                                                                                            
  jetTaggerDetLevel->SetTypeAcceptance(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kNoLimit);
  // Use default matching distance                                                                                                                                    
  jetTaggerDetLevel->SetMaxDistance(maxGeoMatchingDistance);
  // Redundant, but done for completeness                                                                                                                             
  jetTaggerDetLevel->SelectCollisionCandidates(kPhysSel);
  // Set fraction shared momentum for output hists                                                                                                                    
  //jetTaggerDetLevel->SetMinFractionShared(fractionSharedMomentum);$$                                                                                                
  jetTaggerDetLevel->SetUseNewCentralityEstimation(kTRUE);

  // Reapply the max track pt cut off to maintain energy resolution and avoid fake tracks                                                                             
  // ******* Change ******** Also apply Leading track bias                                                                                                            
  AliJetContainer * hybridJetCont = jetTaggerDetLevel->GetJetContainer(0);
  hybridJetCont->SetMaxTrackPt(100);
  //hybridJetCont->SetPtBiasJetTrack(5.);                                                                                                                             
  AliJetContainer * detLevelJetCont = jetTaggerDetLevel->GetJetContainer(1);
  detLevelJetCont->SetMaxTrackPt(100);
  //detLevelJetCont->SetPtBiasJetTrack(5.);                                                                                                                           

  // Embed det level jets are the "base" jet collection                                                                                                               
  // Embed part level jets are the "tag" jet collection                                                                                                               
  const std::string partLevelJetsName = pFullJetTaskPartLevel->GetName();

  PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast * jetTaggerPartLevel = PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::AddTaskJetTaggerFast(                       
			detLevelJetsName.c_str(),       // "Base" jet collection which will be tagged                                                                
			partLevelJetsName.c_str(),      // "Tag" jet collection which will be used to tag (and will be tagged)                                      
			jetRadius,                      // Jet radius                                                                                              
			"",                             // Det level ("base") rho name                                                                                
			"",                             // Part level ("tag") rho name                                                                                
			"",                             // tracks to attach to the jet containers. Not meaningful here, so left empty                                
			"",                             // clusters to attach to the jet conatiners. Not meaingful here, so left empty (plus, it's not the same for the two jet collections!)                                                                                                                                              
			acceptanceTypeStr.c_str(),      // Jet acceptance type for the "base" collection                                                           
			centralityEstimator.c_str(),    // Centrality estimator                                                                                     
			kPhysSel,                       // Physics selection                                                                                       
			""                              // Trigger class. We can just leave blank, as it's only used in the task name                                 
																	 );                                                                                                                                                                
  // Task level settings                                                                                                                                            
  jetTaggerPartLevel->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);                                                                                    
  // Tag via geometrical matching                                                                                                                                    
  jetTaggerPartLevel->SetJetTaggingMethod(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kGeo);                                                                     
  // Tag the closest jet                                                                                                                                             
  jetTaggerPartLevel->SetJetTaggingType(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kClosest);                                                                   
  // Don't impose any additional acceptance cuts beyond the jet containers                                                                                           
  jetTaggerPartLevel->SetTypeAcceptance(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kNoLimit);                                                                  
  // Use default matching distance                                                                                                                                   
  jetTaggerPartLevel->SetMaxDistance(maxGeoMatchingDistance);                                                                                                        
  // Redundant, but done for completeness                                                                                                                             
  jetTaggerPartLevel->SelectCollisionCandidates(kPhysSel);    
  jetTaggerPartLevel->SetUseNewCentralityEstimation(kTRUE);                                                                                                         
  // We don't want to apply a shared momentum fraction cut here, as it's not meaningful                                                                              
                                                                                                                             
  // Reapply the max track pt cut off to maintain energy resolution and avoid fake tracks                                                                             
  // However, don't apply to the particle level jets which don't suffer this effect                                                                                   
  detLevelJetCont = jetTaggerPartLevel->GetJetContainer(0);                                                                                                           
  detLevelJetCont->SetMaxTrackPt(100);                                                                                                                                
  detLevelJetCont->SetPtBiasJetTrack(5.);                                                                                                                                                                                                                                                                                                   
  AliJetContainer * partLevelJetCont = jetTaggerDetLevel->GetJetContainer(1);                                                                                         
  partLevelJetCont->SetPtBiasJetTrack(5.);  
    




  /*  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalSoftDropResponse* Softdrop_Responsemaker_Fulljets_R04_EJ1 = AddTaskEmcalSoftdropResponse(0.4, AliJetContainer::kChargedJet, AliJetContainer::E_scheme, true,partLevelTracks->GetName(), "");
    Softdrop_Responsemaker_Fulljets_R04_EJ1->SetIsPythia(kIsPtHard);
    //    __R_ADDTASK__->GetClusterContainer(0)->SetDefaultClusterEnergy(AliVCluster::kHadCorr);
    //    __R_ADDTASK__->GetClusterContainer(0)->SetClusHadCorrEnergyCut(0.3);
    if(kIsPtHard) Softdrop_Responsemaker_Fulljets_R04_EJ1->SetUsePtHardBinScaling(kTRUE);
    if(kIsPtHard){
      Softdrop_Responsemaker_Fulljets_R04_EJ1->SetNumberOfPtHardBins(knPthardBins);
      if(knPthardBins!=11) Softdrop_Responsemaker_Fulljets_R04_EJ1-> SetUserPtHardBinning(kPtHardBinning);
    }
      std::cout << "IS THIS WHERE IT IS BREAKING?" << std::endl;
    Softdrop_Responsemaker_Fulljets_R04_EJ1->SetMakeGeneralHistograms(kTRUE);
    Softdrop_Responsemaker_Fulljets_R04_EJ1->SelectCollisionCandidates(kPhysSel);
    Softdrop_Responsemaker_Fulljets_R04_EJ1->SetUseNewCentralityEstimation(kTRUE);
    Softdrop_Responsemaker_Fulljets_R04_EJ1->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
    AliJetContainer *contConstSub = Softdrop_Responsemaker_Fulljets_R04_EJ1->GetJetContainer(1);
    contConstSub->SetRhoName("Rho");
    contConstSub->SetRhoMassName("RhoMass");
    AliJetContainer* jetContFu02 = Softdrop_Responsemaker_Fulljets_R04_EJ1->AddJetContainer("hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub");
    std::cout << "ADDED THE CONST SUB CONTAINER _ DID IT ORK????????" << std::endl;
  std::cout << "jet container 2: " <<   Softdrop_Responsemaker_Fulljets_R04_EJ1->GetJetContainer(2)->GetName() << std::endl;
  //jetContFu02->SetName("detLevel");
  Softdrop_Responsemaker_Fulljets_R04_EJ1->SetNameDetLevelJetContainer(jetContFu02->GetName());
    jetContFu02->SetJetPtCut(0.);
    jetContFu02->SetMaxTrackPt(1000.);

        if(fullJets){
      Softdrop_Responsemaker_Fulljets_R04_EJ1->GetClusterContainer(0)->SetIsEmbedding(kFALSE);
    }
    AliParticleContainer * partCont = Softdrop_Responsemaker_Fulljets_R04_EJ1->GetMCParticleContainer(0);
  // Called Embedded, but really just means get from an external event!                                                                                                 
    partCont->SetIsEmbedding(kTRUE);

  std::cout << "jet container 0: " <<   Softdrop_Responsemaker_Fulljets_R04_EJ1->GetJetContainer(0)->GetName() << std::endl;
  std::cout << "jet container 1: " <<   Softdrop_Responsemaker_Fulljets_R04_EJ1->GetJetContainer(1)->GetName() << std::endl;
  std::cout << "jet container 2: " <<   Softdrop_Responsemaker_Fulljets_R04_EJ1->GetJetContainer(2)->GetName() << std::endl;*/

  AliAnalysisTaskNewJetSubstructure* JetSubstructureTask =  AddTaskNewJetSubstructure("hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub","hybridLevelJets_AKTChargedR040_tracks_pT0150_E_scheme","detLevelJets_AKTChargedR040_tracks_pT0150_E_scheme","partLevelJets_AKTChargedR040_mcparticles_pT0000_E_scheme",0.4,"Rho","tracks","tracks","mcparticles","","mcparticles","TPC","V0M",1<<31,"","","Raw",AliAnalysisTaskNewJetSubstructure::kDetEmbPartPythia,AliAnalysisTaskNewJetSubstructure::kConstSub , AliAnalysisTaskNewJetSubstructure::kInclusive);

  JetSubstructureTask->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  JetSubstructureTask->SetUseNewCentralityEstimation(kTRUE);
  
  AliJetContainer *contsub = JetSubstructureTask->GetJetContainer(0);
  contsub->SetRhoName("Rho");
  contsub->SetRhoMassName("Rhomass");
  contsub->SetJetRadius(0.4);
  contsub->SetJetAcceptanceType(AliJetContainer::kTPCfid);
  contsub->SetMaxTrackPt(100);


  AliJetContainer *cont2sub = JetSubstructureTask->GetJetContainer(1);
  cont2sub->SetRhoName("Rho");
  cont2sub->SetJetRadius(0.4);
  cont2sub->SetJetAcceptanceType(AliJetContainer::kTPCfid);
  cont2sub->SetMaxTrackPt(100);
  //cont2sub->SetIsEmbedding(kTRUE);
  //  JetSubstructureTask->SetMaxCentrality(10);
  //  JetSubstructureTask->SetMinCentrality(0);
  //  JetSubstructureTask->SetMinFractionShared(0.5);
  //  JetSubstructureTask->SetJetPtThreshold(20);
  JetSubstructureTask->SelectCollisionCandidates(kPhysSel);
  // JetSubstructureTask->SetVzRange(-10,10);
  //  JetSubstructureTask->SetNeedEmcalGeom(kFALSE);
  //  JetSubstructureTask->SetZvertexDiffValue(0.1);
  //  JetSubstructureTask->SetHardCutoff(0.1);

  AliJetContainer *cont4= JetSubstructureTask->GetJetContainer(2);
  cont4->SetJetRadius(0.4);
  cont4->SetJetAcceptanceType(AliJetContainer::kTPCfid);
  cont4->SetMaxTrackPt(10000.);
  //  cont4->SetIsEmbedding(kTRUE);

  TObjArray *pTopTasks = pMgr->GetTasks();
  for (Int_t i = 0; i < pTopTasks->GetEntries(); ++i) {
    AliAnalysisTaskSE *pTask = dynamic_cast<AliAnalysisTaskSE*>(pTopTasks->At(i));
    if (!pTask) continue;
    if (pTask->InheritsFrom("AliAnalysisTaskEmcal")) {
      AliAnalysisTaskEmcal *pTaskEmcal = static_cast<AliAnalysisTaskEmcal*>(pTask);
      Printf("Setting beam type %d for task %s", iBeamType, pTaskEmcal->GetName());
      pTaskEmcal->SetForceBeamType(iBeamType);
    }
    if (pTask->InheritsFrom("AliEmcalCorrectionTask")) {
      AliEmcalCorrectionTask * pTaskEmcalCorrection = static_cast<AliEmcalCorrectionTask*>(pTask);
      Printf("Setting beam type %d for task %s", iBeamType, pTaskEmcalCorrection->GetName());
      pTaskEmcalCorrection->SetForceBeamType(static_cast<AliEmcalCorrectionTask::BeamType>(iBeamType));
    }
  }



  if (!pMgr->InitAnalysis()) return 0;
  pMgr->PrintStatus();
    
  pMgr->SetUseProgressBar(kTRUE, 250);
  
  TFile *pOutFile = new TFile("train.root","RECREATE");
  pOutFile->cd();
  pMgr->Write();
  pOutFile->Close();
  delete pOutFile;

  if (iStartAnalysis == 1) { // start local analysis
    TChain* pChain = 0;
    if (iDataType == kAod) {
      #ifdef __CLING__
      std::stringstream aodChain;
      aodChain << ".x " << gSystem->Getenv("ALICE_PHYSICS") <<  "/PWG/EMCAL/macros/CreateAODChain.C(";
      aodChain << "\"" << sLocalFiles.Data() << "\", ";
      aodChain << iNumEvents << ", ";
      aodChain << 0 << ", ";
      aodChain << std::boolalpha << kFALSE << ");";
      pChain = reinterpret_cast<TChain *>(gROOT->ProcessLine(aodChain.str().c_str()));
      #else
      gROOT->LoadMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/CreateAODChain.C");
      pChain = CreateAODChain(sLocalFiles.Data(), iNumFiles, 0, kFALSE);
      #endif
    }
    else {
      #ifdef __CLING__
      std::stringstream esdChain;
      esdChain << ".x " << gSystem->Getenv("ALICE_PHYSICS") <<  "/PWG/EMCAL/macros/CreateESDChain.C(";
      esdChain << "\"" << sLocalFiles.Data() << "\", ";
      esdChain << iNumEvents << ", ";
      esdChain << 0 << ", ";
      esdChain << std::boolalpha << kFALSE << ");";
      pChain = reinterpret_cast<TChain *>(gROOT->ProcessLine(esdChain.str().c_str()));
      #else
      gROOT->LoadMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/CreateESDChain.C");
      pChain = CreateESDChain(sLocalFiles.Data(), iNumFiles, 0, kFALSE);
      #endif
    }

    // start analysis
    Printf("Starting Analysis...");
    pMgr->StartAnalysis("local", pChain, iNumEvents);
  }
  else if (iStartAnalysis == 2) {  // start grid analysis
    StartGridAnalysis(pMgr, cTaskName, cGridMode);
  }

  return pMgr;
}

void LoadMacros()
{
  // Aliroot macros
  gROOT->LoadMacro("$ALICE_PHYSICS/OADB/macros/AddTaskCentrality.C");
  gROOT->LoadMacro("$ALICE_PHYSICS/OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C");
  gROOT->LoadMacro("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C");
  gROOT->LoadMacro("$ALICE_PHYSICS/PWGPP/PilotTrain/AddTaskCDBconnect.C");
  //  gROOT->LoadMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetSample.C");
  gROOT->LoadMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskRhoNew.C");
  gROOT->LoadMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetPerfTree.C");
}

void StartGridAnalysis(AliAnalysisManager* pMgr, const char* uniqueName, const char* cGridMode)
{
  Int_t maxFilesPerWorker = 4;
  Int_t workerTTL = 7200;
  const char* runNumbers = "180720";
  const char* pattern = "pass2/AOD/*/AliAOD.root";
  const char* gridDir = "/alice/data/2012/LHC12c";
  const char* additionalCXXs = "";
  const char* additionalHs = "";

  AliAnalysisGrid *plugin = CreateAlienHandler(uniqueName, gridDir, cGridMode, runNumbers, pattern, additionalCXXs, additionalHs, maxFilesPerWorker, workerTTL, kFALSE);
  pMgr->SetGridHandler(plugin);

  // start analysis
   Printf("Starting GRID Analysis...");
   pMgr->SetDebugLevel(0);
   pMgr->StartAnalysis("grid");
}

AliAnalysisGrid* CreateAlienHandler(const char* uniqueName, const char* gridDir, const char* gridMode, const char* runNumbers,
    const char* pattern, TString additionalCode, TString additionalHeaders, Int_t maxFilesPerWorker, Int_t workerTTL, Bool_t isMC)
{
  TDatime currentTime;
  TString tmpName(uniqueName);

  // Only add current date and time when not in terminate mode! In this case the exact name has to be supplied by the user
  if (strcmp(gridMode, "terminate")) {
    tmpName += "_";
    tmpName += currentTime.GetDate();
    tmpName += "_";
    tmpName += currentTime.GetTime();
  }

  TString macroName("");
  TString execName("");
  TString jdlName("");
  macroName = Form("%s.C", tmpName.Data());
  execName = Form("%s.sh", tmpName.Data());
  jdlName = Form("%s.jdl", tmpName.Data());

  AliAnalysisAlien *plugin = new AliAnalysisAlien();
  plugin->SetOverwriteMode();
  plugin->SetRunMode(gridMode);

  // Here you can set the (Ali)PHYSICS version you want to use
  plugin->SetAliPhysicsVersion("vAN-20170628-1");

  plugin->SetGridDataDir(gridDir); // e.g. "/alice/sim/LHC10a6"
  plugin->SetDataPattern(pattern); //dir structure in run directory

  if (!isMC) plugin->SetRunPrefix("000");

  plugin->AddRunList(runNumbers);

  plugin->SetGridWorkingDir(Form("work/%s",tmpName.Data()));
  plugin->SetGridOutputDir("output"); // In this case will be $HOME/work/output

  plugin->SetAnalysisSource(additionalCode.Data());

  plugin->SetDefaultOutputs(kTRUE);
  plugin->SetAnalysisMacro(macroName.Data());
  plugin->SetSplitMaxInputFileNumber(maxFilesPerWorker);
  plugin->SetExecutable(execName.Data());
  plugin->SetTTL(workerTTL);
  plugin->SetInputFormat("xml-single");
  plugin->SetJDLName(jdlName.Data());
  plugin->SetPrice(1);
  plugin->SetSplitMode("se");

  // merging via jdl
  plugin->SetMergeViaJDL(kTRUE);
  plugin->SetOneStageMerging(kFALSE);
  plugin->SetMaxMergeStages(2);

  return plugin;
}
