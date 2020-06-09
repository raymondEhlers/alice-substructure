void Split(const char* radius);

void SplitLHC15o()
{
  Split("R020");
  Split("R040");
  Split("R060");
}

void Split(const char* radius)
{
   TFile *file = TFile::Open(Form("./LHC15o_%s.root", radius));
   TTree *T = (TTree*)file->Get(Form("JetTree_AliAnalysisTaskJetExtractor_Jet_AKTCharged%s_tracks_pT0150_E_scheme_RhoR020KT_allJets", radius));
   TList *L = (TList*)file->Get(Form("ChargedJetsHadronCF/AliAnalysisTaskJetExtractor_Jet_AKTCharged%s_tracks_pT0150_E_scheme_RhoR020KT_allJets_histos", radius));

   Long64_t nentries = T->GetEntries();
   printf("tree has %lld entries\n",nentries);

   for(Int_t part=0; part<4; part++)
   {
       Int_t start = (nentries/4)*part;
       Int_t end = (nentries/4)*(part+1);
       //now loop on the N selected entries and wrtite them into tree
       TFile *filep = new TFile(Form("./LHC15o_%s_Part%d.root", radius, part+1), "UPDATE");
       TTree *newtree = T->CloneTree(0);
       L->Write(0, TObject::kSingleKey);
       gROOT->cd();
       TTree* tree = T;
       for (Int_t i=start;i<end;i++) {
          if (i % 10000 == 0)
            cout << "Done:  " << (Double_t)(i-start)/(end-start) << endl;
          tree->GetEntry(i);
          newtree->Fill();
       }
       newtree->AutoSave();
       delete filep;
   }
   delete file;
   return;

}
