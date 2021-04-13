void SwitchOffBranches(TTree* fTree, TString fBranches);
void SwitchOnBranchesOld(TTree* fTree, TString fBranchesOn);
void SwitchOnBranches(TTree* fTree, TString fBranchesOn);
//void SwitchOnBranchImpl(TTree* fTree, TString branchName);
void SwitchOnBranchImpl(TTree* fTree, TBranch * branch);
void SetBranchStatus(TTree * tree, TString branchesToFilter, int status);
void SetBranchStatusImpl(TBranch * branch, int status);

void test_tree_branch_status()
{
  TFile* inFile = TFile::Open("/Volumes/Elements/alice/data/data/2018/LHC18q/000296550/pass1/AOD/001/AliAOD.root");
  TTree* tree = (TTree*)inFile->Get("aodTree");
  TList* treeList = (TList*)tree->GetListOfBranches();
  //TBranch* br = (TBranch*)tree->GetBranch("header");
  TBranch* br = (TBranch*)tree->GetBranch("Forward");
  TList* headerList = (TList*)br->GetListOfBranches();
  TIter iter(treeList);
  TIter hIter(headerList);
  std::cout << "***\tInitial branch status***\n";
  TNamed* obj = 0;
  while ((obj = (TNamed*)iter())) {
    std::cout << "Get branch status of branch: " << obj->GetName() << " " << tree->GetBranchStatus(obj->GetName())
         << std::endl;
  }
  // treeList->Print();
  SwitchOffBranches(tree, "*");
  //tree->SetBranchStatus("*", 0);
  //tree->SetBranchStatus("header", 1);
  std::cout << "***\tStatus after turning branches off***\n";
  TIter iter2(treeList);
  while ((obj = (TNamed*)iter2())) {
    std::cout << "Get branch status of branch: " << obj->GetName() << " " << tree->GetBranchStatus(obj->GetName())
         << std::endl;
  }

  //SwitchOnBranches(tree, "header tracks vertices emcalCells MultSelection");
  SwitchOnBranches(tree, "Forward");
  //tree->SetBranchStatus("Forward.fIpZ", 1);
  std::cout << "***\tStatus after turning branches back on***\n";
  TIter iter3(treeList);
  while ((obj = (TNamed*)iter3())) {
    std::cout << "Get branch status of branch: " << obj->GetName() << " " << tree->GetBranchStatus(obj->GetName())
         << std::endl;
  }
  std::cout << "***\tParticular branch specific***\n";
  while ((obj = (TNamed*)hIter())) {
    std::cout << "Get branch status of branch: " << obj->GetName() << " "
         << tree->GetBranchStatus(Form("%s", obj->GetName())) << std::endl;
  }

  // Individual tests.
  // This won't ever work, it seems.
  std::cout << "Get branch status " << tree->GetBranchStatus("Forward.fIpZ") << std::endl;
  // But this will
  std::cout << "Get branch status " << tree->GetBranchStatus("fIpZ") << std::endl;

  //tree->Print();
  int runNumber = -1;
  tree->SetBranchAddress("fRunNumber", &runNumber);
  tree->GetEntry(0);
  std::cout << "run number: " << runNumber << "\n";
}

//______________________________________________________________________________
void SwitchOffBranches(TTree* fTree, TString fBranches)
{
  //
  // Switch of branches on user request
  /*TObjArray* tokens = fBranches.Tokenize(" ");
  Int_t ntok = tokens->GetEntries();
  for (Int_t i = 0; i < ntok; i++) {
    TString str = ((TObjString*)tokens->At(i))->GetString();
    if (str.Length() == 0)
      continue;
    fTree->SetBranchStatus(Form("%s%s%s", "*", str.Data(), "*"), 0);
    std::cout << Form("Branch %s switched off", str.Data()) << std::endl;
  }
  delete tokens;*/
  SetBranchStatus(fTree, fBranches, 0);
}

//______________________________________________________________________________
void SwitchOnBranchesOld(TTree* fTree, TString fBranchesOn)
{
  //
  // Switch of branches on user request
  TObjArray* tokens = fBranchesOn.Tokenize(" ");
  Int_t ntok = tokens->GetEntries();

  for (Int_t i = 0; i < ntok; i++) {
    TString str = ((TObjString*)tokens->At(i))->GetString();
    if (str.Length() == 0)
      continue;
    fTree->SetBranchStatus(Form("%s%s%s", "*", str.Data(), "*"), 1);
    std::cout << Form("Branch %s switched on", str.Data()) << std::endl;
  }
  delete tokens;
}

void SwitchOnBranches(TTree* fTree, TString fBranchesOn)
{
  //
  // Switch branches status on user request
  /*TObjArray* tokens = fBranchesOn.Tokenize(" ");
  Int_t ntok = tokens->GetEntries();

  TObjArray * leaves = fTree->GetListOfLeaves();
  unsigned int nLeaves = leaves->GetEntriesFast();

  for (Int_t j = 0; j < ntok; j++) {
    TString str = ((TObjString*)tokens->At(j))->GetString();
    if (str.Length() == 0)
      continue;
    SwitchOnBranchImpl(fTree, fTree->GetBranch(str));
  }
  delete tokens;*/
  SetBranchStatus(fTree, fBranchesOn, 1);
}


void SetBranchStatus(TTree * tree, TString branchesToFilter, int status)
{
  // Switch status of branches on user request
  TObjArray* tokens = branchesToFilter.Tokenize(" ");
  Int_t ntok = tokens->GetEntries();
  bool foundStar = false;
  for (Int_t j = 0; j < ntok; j++) {
    TString filterName = ((TObjString*)tokens->At(j))->GetString();
    if (filterName.Length() == 0) {
      continue;
    }
    if (filterName == "*") {
      foundStar = true;
      // Expand star to the names of all branches and call this function again.
      std::cout << "Handling * manually.\n";
      TString newSetOfBranches = "";
      TIter next(tree->GetListOfBranches());
      TNamed * named = nullptr;
      while ((named = static_cast<TNamed *>(next()))) {
        newSetOfBranches += named->GetName();
        // Split each with a trailing space.
        // NOTE: This will leave an unnecessary trailing space at the very end,
        //       but it will be ignored during the tokenize, so it's fine.
        newSetOfBranches += " ";
      }
      branchesToFilter = newSetOfBranches;
      break;
    }
  }
  std::cout << "branchesToFilter " << branchesToFilter << "\n";
  if (foundStar) {
    delete tokens;
    tokens = branchesToFilter.Tokenize(" ");
    ntok = tokens->GetEntries();
  }

  auto * leaves = tree->GetListOfLeaves();
  TIter nextLeaf(leaves);
  TLeaf * leaf = nullptr;
  while ((leaf = static_cast<TLeaf *>(nextLeaf()))) {
    auto branch = leaf->GetBranch();
    auto motherBranch = branch->GetMother();
    std::cout << "Branch retrieved " << branch->GetName() << "\n";

    // Iterate over filter branches, and check for mother branch.
    bool applyStatusToThisLeaf = false;
    for (Int_t j = 0; j < ntok; j++) {
      TString filterName = ((TObjString*)tokens->At(j))->GetString();
      //std::cout << "Checking filter name " << filterName.Data() << "\n";
      if (filterName.Length() == 0) {
        continue;
      }
      if (motherBranch->GetName() == filterName) {
        applyStatusToThisLeaf = true;
        break;
      }
    }

    if (applyStatusToThisLeaf) {
      std::cout << "Handling leaf " << leaf->GetName() << "\n";
      SetBranchStatusImpl(branch, status);
      auto leafCount = leaf->GetLeafCount();
      if (leafCount) {
        std::cout << "handling leaf count for " << leafCount->GetName() << "\n";
        SetBranchStatusImpl(leafCount->GetBranch(), status);
      }
    }
  }

  /*
  for (Int_t i = 0; i < ntok; i++) {
    TString str = ((TObjString*)tokens->At(i))->GetString();
    if (str.Length() == 0)
      continue;
    // Unfortunately, "*" won't work for switching on due to how AODs are
    // written, so we have to handle it more carefully. We treat both cases the same,
    // even though it could work for switching off.
    if (str == "*") {
      // Expand star to the names of all branches and call this function again.
      AliDebugStream(2) << "Handling * manually.\n";
      TString newSetOfBranches = "";
      TIter next(tree->GetListOfBranches());
      TNamed * named = nullptr;
      while ((named = static_cast<TNamed *>(next()))) {
        newSetOfBranches += named->GetName();
        // Split each with a trailing space.
        // NOTE: This will leave an unnecessary trailing space at the very end,
        //       but it will be ignored during the tokenize, so it's fine.
        newSetOfBranches += " ";
      }
      AliDebugStream(2) << "Calling with \"" << newSetOfBranches.Data() << "\"\n";
      SetBranchStatus(tree, newSetOfBranches, status);
    }
    else {
      SetBranchStatusImpl(tree->GetBranch(str), status);
    }
  }*/
  delete tokens;
}

void SetBranchStatusImpl(TBranch * branch, int status)
{
  if (!branch) {
    return;
  }
  // First, switch off the main branch
  branch->SetStatus(status);
  std::cout << "Branch " << branch->GetName() << " set to status " << status << "\n";
}


/*void SwitchOnBranchImpl(TTree* fTree, TString branchName)
{
  // First, switch off the main branch
  fTree->SetBranchStatus(branchName, 1);
  std::cout << Form("Branch %s switched on", branchName.Data()) << std::endl;
  // Now, check if we need to handle this recursively.
  TBranch * branch = dynamic_cast<TBranch *>(fTree->GetBranch(branchName));
  if (branch) {
    TIter next(branch->GetListOfBranches());
    TNamed * obj = nullptr;
    while ((obj = static_cast<TNamed *>(next()))) {
      TString inputName = obj->GetName();
      if (!(inputName.Contains(branchName + "."))) {
        inputName = branchName + "." + inputName;
      }
      SwitchOnBranchImpl(fTree, inputName);
    }
  }
}*/

void SwitchOnBranchImpl(TTree* fTree, TBranch * branch)
{
  if (!branch) {
    return;
  }
  // First, switch off the main branch
  branch->SetStatus(1);
  std::cout << "Branch " << branch->GetName() << " switched on\n";
  // Now, check if we need to handle this recursively.
  //TBranch * branch = dynamic_cast<TBranch *>(fTree->GetBranch(branchName));
  TIter next(branch->GetListOfBranches());
  TBranch * br = nullptr;
  while ((br = static_cast<TBranch *>(next()))) {
    std::cout << "Recurse for " << br->GetName() << "\n";
    SwitchOnBranchImpl(fTree, br);
  }
}
