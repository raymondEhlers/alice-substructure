TChain* CreateChain(const char *xmlfile, const char *type="ESD");

const char *anatype = "AOD";

void lego_train()
{
// Analysis using AOD data
// Automatically generated analysis steering macro executed in grid subjobs

   TStopwatch timer;
   timer.Start();

// connect to AliEn and make the chain
   if (!TGrid::Connect("alien://")) return;
// Set temporary merging directory to current one
   gSystem->Setenv("TMPDIR", gSystem->pwd());

// Set temporary compilation directory to current one
   gSystem->SetBuildDir(gSystem->pwd(), kTRUE);

// Reset existing include path and add current directory first in the search
   gSystem->SetIncludePath("-I.");
// load base root libraries
   gSystem->Load("libTree");
   gSystem->Load("libGeom");
   gSystem->Load("libVMC");
   gSystem->Load("libPhysics");

   gSystem->Load("libMinuit");

// Add aditional libraries
   gSystem->Load("libVMC");
   gSystem->Load("libPhysics");
   gSystem->Load("libTree");
   gSystem->Load("libMinuit");
   gSystem->Load("libProof");
   gSystem->Load("libSTEERBase");
   gSystem->Load("libESD");
   gSystem->Load("libAOD");
// Load analysis framework libraries
   gSystem->Load("libSTEERBase");
   gSystem->Load("libESD");
   gSystem->Load("libAOD");
   gSystem->Load("libANALYSIS");
   gSystem->Load("libANALYSISalice");
   gSystem->Load("libOADB");
   gSystem->Load("libCORRFW");

// include path
   TString intPath = gInterpreter->GetIncludePath();
   TObjArray *listpaths = intPath.Tokenize(" ");
   TIter nextpath(listpaths);
   TObjString *pname;
   while ((pname=(TObjString*)nextpath())) {
      TString current = pname->GetName();
      if (current.Contains("AliRoot") || current.Contains("ALICE_ROOT")) continue;
      gSystem->AddIncludePath(current);
   }
   if (listpaths) delete listpaths;
   gROOT->ProcessLine(".include $ALICE_ROOT/include");
   printf("Include path: %s\n", gSystem->GetIncludePath());

// Add aditional AliRoot libraries
   gSystem->Load("libCGAL");
   gSystem->Load("libfastjet");
   gSystem->Load("libsiscone");
   gSystem->Load("libsiscone_spherical");
   gSystem->Load("libfastjetplugins");
   gSystem->Load("libfastjettools");
   gSystem->Load("libfastjetcontribfragile");
   gSystem->Load("libJETAN");
   gSystem->Load("libFASTJETAN");
   gSystem->Load("libESDfilter");
   gSystem->Load("libPWGGAEMCALTasks");
   gSystem->Load("libPWGJEEMCALJetTasks");

// analysis source to be compiled at runtime (if any)

// read the analysis manager from file
   AliAnalysisManager *mgr = AliAnalysisAlien::LoadAnalysisManager("lego_train.root");
   if (!mgr) return;
   mgr->PrintStatus();
   AliLog::SetGlobalLogLevel(AliLog::kError);
   TChain *chain = CreateChain("wn.xml", anatype);

   mgr->StartAnalysis("localfile", chain, 123456789, 0);
   timer.Stop();
   timer.Print();
}

//________________________________________________________________________________
TChain* CreateChain(const char *xmlfile, const char *type)
{
// Create a chain using url's from xml file
   TString filename;
   Int_t run = 0;
   TString treename = type;
   treename.ToLower();
   treename += "Tree";
   printf("***************************************\n");
   printf("    Getting chain of trees %s\n", treename.Data());
   printf("***************************************\n");
   TGridCollection *coll = gGrid->OpenCollection(xmlfile);
   if (!coll) {
      ::Error("CreateChain", "Cannot create an AliEn collection from %s", xmlfile);
      return NULL;
   }
   AliAnalysisManager *mgr = AliAnalysisManager::GetAnalysisManager();
   TChain *chain = new TChain(treename);
   coll->Reset();
   while (coll->Next()) {
      filename = coll->GetTURL();
      if (mgr) {
         Int_t nrun = AliAnalysisManager::GetRunFromAlienPath(filename);
         if (nrun && nrun != run) {
            printf("### Run number detected from chain: %d\n", nrun);
            mgr->SetRunFromPath(nrun);
            run = nrun;
         }
      }
      chain->Add(filename);
   }
   if (!chain->GetNtrees()) {
      ::Error("CreateChain", "No tree found from collection %s", xmlfile);
      return NULL;
   }
   return chain;
}

