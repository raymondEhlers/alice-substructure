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
#include "TClonesArray.h"
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

//__________________________________________________________________________

bool EtaCut(fastjet::PseudoJet fjJet, double etaMin, double etaMax)
{
  if (fjJet.eta() > etaMax || fjJet.eta() < etaMin) {
    return false;
  } else {
    return true;
  }
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
//__________________________________________________________________________

// Lund_____________________________________________________________________________________

THnSparse* fHLundIterative;
THnSparse* fHLundIterativeOriginal;

TH2D* fh2CorrHardProbe;
TH2D* fh2CorrTimeProbe;

TH2D* fh2CorrHardHybrid;
TH2D* fh2CorrTimeHybrid;

TH2D* fh2CorrHardProbeRad;
TH2D* fh2CorrTimeProbeRad;

TH2D* fh2CorrHardHybridRad;
TH2D* fh2CorrTimeHybridRad;

TH2D* fh2CorrHardProbeZ;
TH2D* fh2CorrTimeProbeZ;

TH2D* fh2CorrHardHybridZ;
TH2D* fh2CorrTimeHybridZ;

// ITERATIVE DECLUSTERING____________________________________________________

void IterativeDeclustering(fastjet::PseudoJet jet, Int_t type, Int_t flag, Double_t xktmaxgraph, Double_t tfmingraph)
{
  double flagSubjet = 0;
  double tform = 0;
  double erad = 0;
  double nall = 0;
  double rad = 0;
  double form = 0;
  double formfirst = 1;

  // cout<<xktmaxgraph<< " in iteration"<<tfmingraph<<endl;
  // cout<<"***************************************************"<<endl;
  //_________________________________________________________________________________________

  double jet_radius_ca = 0.4;
  fastjet::JetDefinition jet_def(fastjet::genkt_algorithm, jet_radius_ca, 0,
                  static_cast<fastjet::RecombinationScheme>(0), fastjet::Best);

  // Reclustering jet constituents with new algorithm
  try {
    std::vector<fastjet::PseudoJet> particles = jet.constituents();
    fastjet::ClusterSequence cs_gen(particles, jet_def);
    std::vector<fastjet::PseudoJet> output_jets = cs_gen.inclusive_jets(0);
    output_jets = sorted_by_pt(output_jets);

    // input jet but reclustered with ca
    fastjet::PseudoJet jj = output_jets[0];

    // Auxiliar variables
    fastjet::PseudoJet j1; // subjet 1 (largest pt)
    fastjet::PseudoJet j2; // subjet 2 (smaller pt)

    double aux = 1;
    double auxg = 1;
    double xktmax = -20;
    double tfmin = 1000;
    double xktmaxb = -20;
    double tfminb = 1000;
    double xktmaxc = -20;
    double tfminc = 1000;

    // Unclustering jet
    while (jj.has_parents(j1, j2)) {
      if (j1.perp() < j2.perp())
        std::swap(j1, j2);
      nall = nall + 1;

      // Calculate deltaR and Zg between j1 and j2
      double delta_R = j1.delta_R(j2);

      double xkt = j2.perp() * sin(delta_R);
      double lnpt_rel = log(xkt);
      double y = log(1. / delta_R);
      double zet = j2.perp() / (j1.perp() + j2.perp());
      form = 2 * 0.197 * j2.e() / ((1 - zet) * xkt * xkt); // in fermis
      rad = j2.e();
      if (nall == 1)
        formfirst = form;

      double lundEntriesOriginal[8] = { y,    lnpt_rel, output_jets[0].perp(), form / formfirst, rad, nall,
                       flag, type };
      fHLundIterativeOriginal->Fill(lundEntriesOriginal);

      if (xkt > xktmax)
        xktmax = xkt;
      if (form < tfmin)
        tfmin = form;

      if (xkt > xktmaxb && rad > 20)
        xktmaxb = xkt;
      if (form < tfminb && rad > 20)
        tfminb = form;

      if (xkt > xktmaxc && zet > 0.1) {
        xktmaxc = xkt;
        tfminc = form;
      }

      // continue unclustering
      jj = j1;
    }
    if (type == 0) {
      fh2CorrHardProbe->Fill(log(xktmax), log(xktmaxgraph));
      fh2CorrTimeProbe->Fill(tfmin, tfmingraph);
      fh2CorrHardProbeRad->Fill(log(xktmaxb), log(xktmaxgraph));
      fh2CorrTimeProbeRad->Fill(tfminb, tfmingraph);
      fh2CorrHardProbeZ->Fill(log(xktmaxc), log(xktmaxgraph));
      fh2CorrTimeProbeZ->Fill(tfminc, tfmingraph);
    }

    if (type == 1) {
      fh2CorrHardHybrid->Fill(log(xktmax), log(xktmaxgraph));
      fh2CorrTimeHybrid->Fill(tfmin, tfmingraph);
      fh2CorrHardHybridRad->Fill(log(xktmaxb), log(xktmaxgraph));
      fh2CorrTimeHybridRad->Fill(tfminb, tfmingraph);
      fh2CorrHardHybridZ->Fill(log(xktmaxc), log(xktmaxgraph));
      fh2CorrTimeHybridZ->Fill(tfminc, tfmingraph);
    }

  } catch (fastjet::Error) { /*return -1;*/
  }
}

//__________________________________________________________________________

int main(int argc, char* argv[])
{
  // y,lnpt_rel,pt,niter,flag
  const Int_t dimSpec = 7;
  const Int_t nBinsSpec[7] = { 100, 100, 100, 20, 20, 2, 2 };
  const Double_t lowBinSpec[7] = { -5, -10, 0, 0, 0, 5, 0 };
  const Double_t hiBinSpec[7] = { 5, 10, 200, 20, 20, 7, 2 };
  fHLundIterative = new THnSparseF("fHLundIterative", "LundIterativePlot[jetpt,tform,erad]", dimSpec, nBinsSpec,
                   lowBinSpec, hiBinSpec);

  // y,lnpt_rel,output_jets[0].perp(),form,rad,nall,flag,type
  const Int_t dimSpec2 = 8;
  const Int_t nBinsSpec2[8] = { 100, 100, 100, 100, 100, 20, 2, 2 };
  const Double_t lowBinSpec2[8] = { -5, -10, 0, 0, 0, 0, 5, 0 };
  const Double_t hiBinSpec2[8] = { 5, 10, 200, 20, 20, 20, 7, 2 };
  fHLundIterativeOriginal = new THnSparseF("fHLundIterativeOriginal", "LundIterativeOriginal[jetpt,tform,erad]",
                       dimSpec2, nBinsSpec2, lowBinSpec2, hiBinSpec2);

  fh2CorrHardProbe = new TH2D("h2CorrHardProbe", "h2CorrHardProbe", 100, -5, 5, 100, -5, 5);
  fh2CorrHardProbe->Sumw2();
  fh2CorrHardHybrid = new TH2D("h2CorrHardHybrid", "h2CorrHardHybrid", 100, -5, 5, 100, -5, 5);
  fh2CorrHardHybrid->Sumw2();

  fh2CorrHardProbeRad = new TH2D("h2CorrHardProbeRad", "h2CorrHardProbeRad", 100, -5, 5, 100, -5, 5);
  fh2CorrHardProbeRad->Sumw2();
  fh2CorrHardHybridRad = new TH2D("h2CorrHardHybridRad", "h2CorrHardHybridRad", 100, -5, 5, 100, -5, 5);
  fh2CorrHardHybridRad->Sumw2();

  fh2CorrHardProbeZ = new TH2D("h2CorrHardProbeZ", "h2CorrHardProbeZ", 100, -5, 5, 100, -5, 5);
  fh2CorrHardProbeZ->Sumw2();
  fh2CorrHardHybridZ = new TH2D("h2CorrHardHybridZ", "h2CorrHardHybridZ", 100, -5, 5, 100, -5, 5);
  fh2CorrHardHybridZ->Sumw2();

  fh2CorrTimeProbe = new TH2D("h2CorrTimeProbe", "h2CorrTimeProbe", 100, 0.0, 100.0, 100, 0, 100);
  fh2CorrTimeProbe->Sumw2();
  fh2CorrTimeHybrid = new TH2D("h2CorrTimeHybrid", "h2CorrTimeHybrid", 100, 0.0, 100.0, 100, 0, 100);
  fh2CorrTimeHybrid->Sumw2();

  fh2CorrTimeProbeZ = new TH2D("h2CorrTimeProbeZ", "h2CorrTimeProbeZ", 100, 0.0, 100.0, 100, 0, 100);
  fh2CorrTimeProbeZ->Sumw2();
  fh2CorrTimeHybridZ = new TH2D("h2CorrTimeHybridZ", "h2CorrTimeHybridZ", 100, 0.0, 100.0, 100, 0, 100);
  fh2CorrTimeHybridZ->Sumw2();

  fh2CorrTimeProbeRad = new TH2D("h2CorrTimeProbeRad", "h2CorrTimeProbeRad", 100, 0.0, 100.0, 100, 0, 100);
  fh2CorrTimeProbeRad->Sumw2();
  fh2CorrTimeHybridRad = new TH2D("h2CorrTimeHybridRad", "h2CorrTimeHybridRad", 100, 0.0, 100.0, 100, 0, 100);
  fh2CorrTimeHybridRad->Sumw2();

  Int_t cislo = -1; // unique number for each file
  Int_t tune = -1;  // pythia tune

  Int_t unev = 1; // underlying event (ISR+MPI)

  if (argc != 5) {
    cout << "Usage:" << endl << "./pygen <PythiaTune> <Number> <nEvts> <unev>" << endl;
    return 0;
  }
  tune = atoi(argv[1]);
  cislo = atoi(argv[2]);

  unev = atoi(argv[7]);

  Int_t nEvents = atoi(argv[3]); //(Int_t) 1e3 + 1.0;

  //__________________________________________________________________________
  // ANALYSIS SETTINGS

  double jetParameterR = 0.8; // jet R
  double trackEtaCut = 1;
  double trackLowPtCut = 0.15; // GeV
  //__________________________________________________________________________
  // PYTHIA SETTINGS

  TString name;

  int mecorr = 1;

  Float_t ptHatMin = 80;
  Float_t ptHatMax = 5020;

  // Generator. Process selection. LHC initialization. Histogram.
  Pythia pythia;
  pythia.readString("Beams:idA = 2212"); // beam 1 proton
  pythia.readString("Beams:idB = 2212"); // beam 2 proton
  pythia.readString("Beams:eCM = 5020.");
  pythia.readString(
   Form("Tune:pp = %d", tune)); // tune 1-13    5=defaulr TUNE4C,  6=Tune 4Cx, 7=ATLAS MB Tune A2-CTEQ6L1

  pythia.readString("Random:setSeed = on");
  pythia.readString(Form("Random:seed = %d", cislo));

  pythia.readString("HardQCD:all = on");
  if (ptHatMin < 0 || ptHatMax < 0) {
    pythia.readString("PhaseSpace:pTHatMin = 0."); // <<<<<<<<<<<<<<<<<<<<<<<
  } else {
    name = Form("PhaseSpace:pTHatMin = %f", (Float_t)ptHatMin);
    pythia.readString(name.Data());
    name = Form("PhaseSpace:pTHatMax = %f", (Float_t)ptHatMax);
    pythia.readString(name.Data());
  }

  if (unev == 0) {
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

  // ME corrections
  // use of matrix corrections where available

  pythia.init();

  //_________________________________________________________________________________________________
  // FASTJET  SETTINGS

  double etamin_Sig = -trackEtaCut + jetParameterR; // signal jet eta range
  double etamax_Sig = -etamin_Sig;

  fastjet::Strategy strategy = fastjet::Best;
  fastjet::RecombinationScheme recombScheme = fastjet::E_scheme;

  fastjet::JetDefinition* jetDefAKT_Sig = NULL;

  jetDefAKT_Sig = new fastjet::JetDefinition(fastjet::antikt_algorithm, jetParameterR, recombScheme, strategy);

  fastjet::GhostedAreaSpec ghostareaspec(trackEtaCut, 1, 0.05); // ghost
  // max rap, repeat, ghostarea default 0.01
  fastjet::AreaType areaType = fastjet::active_area_explicit_ghosts;
  fastjet::AreaDefinition* areaDef = new fastjet::AreaDefinition(areaType, ghostareaspec);

  // Fastjet input
  std::vector<fastjet::PseudoJet> fjInputs1;

  std::vector<fastjet::PseudoJet> fjInputs2;
  //___________________________________________________
  // HISTOGRAMS

  // After Thermal Particles: "A-Histograms"
  TH1D* hJetPt_A = new TH1D("hJetPt_A", "hJetPt_A", 200, 0.0, 400.0);
  hJetPt_A->Sumw2();

  TH1D* hJetArea_A = new TH1D("hJetArea_A", "hJetArea_A", 400, 0.0, 4.0);
  hJetArea_A->Sumw2();

  TH1D* hJet_deltaR_A = new TH1D("hJet_deltaR_A", "hJet_deltaR_A", 100, 0.0, 1.0);
  hJet_deltaR_A->Sumw2();

  TH1D* hJet_deltaRg_A = new TH1D("hJet_deltaRg_A", "hJet_deltaRg_A", 100, 0.0, 1.0);
  hJet_deltaRg_A->Sumw2();

  // Before Thermal Particles "B-Histograms"
  TH1D* hJetPt_B = new TH1D("hJetPt_B", "hJetPt_B", 200, 0.0, 400.0);
  hJetPt_B->Sumw2();

  TH1D* hJetArea_B = new TH1D("hJetArea_B", "hJetArea_B", 400, 0.0, 4.0);
  hJetArea_B->Sumw2();

  TH1D* hJet_deltaR_B = new TH1D("hJet_deltaR_B", "hJet_deltaR_B", 100, 0.0, 1.0);
  hJet_deltaR_B->Sumw2();

  TH1D* hJet_deltaRg_B = new TH1D("hJet_deltaRg_B", "hJet_deltaRg_B", 100, 0.0, 1.0);
  hJet_deltaRg_B->Sumw2();

  // PARTICLE DISTRIBUITIONS

  TH1D* h_pT = new TH1D("h_pT", "", 40000, 0.0, 400);
  TH1D* h_eta = new TH1D("h_eta", "", 200, -1.0, 1.0);
  TH1D* h_phi = new TH1D("h_phi", "", 700, -3.5, 3.5);

  TH1D* hP_pT = new TH1D("hP_pT", "", 40000, 0.0, 400);
  TH1D* hP_eta = new TH1D("hP_eta", "", 200, -1.0, 1.0);
  TH1D* hP_phi = new TH1D("hP_phi", "", 700, -3.5, 3.5);

  TH1D* hT_pT = new TH1D("hT_pT", "", 40000, 0.0, 400);
  TH1D* hT_eta = new TH1D("hT_eta", "", 200, -1.0, 1.0);
  TH1D* hT_phi = new TH1D("hT_phi", "", 700, -3.5, 3.5);

  //___________________________________________________
  // Thermal Particles Distribuitions (toy model)

  TF1* f_pT = new TF1("f_pT", "x*exp(-x/0.3)", 0.0, 400.0);
  f_pT->SetNpx(40000);

  TF1* f_eta = new TF1("f_eta", "1", -1.0, 1.0);
  f_eta->SetNpx(200);

  TF1* f_phi = new TF1("f_phi", "1", (-1.0) * TMath::Pi(), TMath::Pi());
  f_phi->SetNpx(700);
  //___________________________________________________

  // Begin event loop
  for (int i = 0; i < nEvents; i++) {
    double xktmax6 = -20;
    double tfmin6 = 1000;
    double xktmax5 = -20;
    double tfmin5 = 1000;

    // 1st Step: Pythia + FastJet -> Probe Jet := Hardest Jet generated by Pythia.
    if (!pythia.next())
      continue;

    fjInputs1.resize(0);
    Double_t index = 0;
    Double_t fourvec[4];
    Double_t vec1[4];
    Double_t vec2[4];
    Double_t vecx[4];
    Int_t flag = 0;
    Int_t in;
    Int_t niter = 0;
    for (Int_t i = 0; i < pythia.event.size(); ++i) {
      if (TMath::Abs(pythia.event[i].status()) == 23) {
        // status 23 is for outgoing partons
        cout << "flavour " << i << endl;
        flag = i;
        in = i;
        niter = 0;
        double form = 0;
        double xkt = 0;
        double y = 0;
        double formf = 1;
        double delta_R = 0;
        Double_t pt = TMath::Sqrt(pythia.event[i].px() * pythia.event[i].py() +
                     pythia.event[i].py() * pythia.event[i].py());

        xktmax6 = -20;
        tfmin6 = 1000;

        xktmax5 = -20;
        tfmin5 = 1000;

        while (pythia.event[in].daughter1() > 0 && pythia.event[in].daughter2() > 0) {
          int fhadron = 0;
          Int_t index1 = pythia.event[in].daughter1();
          Int_t index2 = pythia.event[in].daughter2();

          vec1[0] = pythia.event[index1].px();
          vec1[1] = pythia.event[index1].py();
          vec1[2] = pythia.event[index1].pz();
          vec1[3] = pythia.event[index1].e();

          vec2[0] = pythia.event[index2].px();
          vec2[1] = pythia.event[index2].py();
          vec2[2] = pythia.event[index2].pz();
          vec2[3] = pythia.event[index2].e();

          fastjet::PseudoJet j1(vec1);
          fastjet::PseudoJet j2(vec2);

          if ((TMath::Abs(pythia.event[index1].status()) >= 80) ||
            (TMath::Abs(pythia.event[index2].status()) >= 80))
            fhadron = 1;

          if (j1.perp() < j2.perp()) {
            delta_R = j1.delta_R(j2);
            xkt = j1.perp() * delta_R;
            double lnpt_rel = log(xkt);
            y = log(1. / delta_R);
            double zet = j2.perp() / (j1.perp() + j2.perp());
            form = 2 * 0.197 * j2.e() / ((1 - zet) * xkt * xkt);
            if (niter == 1 && xkt != 0)
              formf = form;
            double lundEntries[7] = { y, lnpt_rel, pt, form / formf, niter, flag, fhadron };
            fHLundIterative->Fill(lundEntries);
            // cout<<"here"<<endl;
            in = index2;
          }

          if (j2.perp() < j1.perp() || j2.perp() == j1.perp()) {
            delta_R = j1.delta_R(j2);
            xkt = j2.perp() * delta_R;
            double lnpt_rel = log(xkt);
            double zet = j2.perp() / (j1.perp() + j2.perp());
            y = log(1. / delta_R);
            form = 2 * 0.197 * j2.e() / ((1 - zet) * xkt * xkt);
            if (niter == 1 && xkt != 0)
              formf = form;
            if (xkt > 0) {
              double lundEntries[7] = { y, lnpt_rel, pt, form / formf, niter, flag, fhadron };
              fHLundIterative->Fill(lundEntries);
            }
            // cout<<"or here"<<endl;
            in = index1;
          }

          if (flag == 5) {
            if (xkt > xktmax5 && delta_R <= 0.4) {
              xktmax5 = xkt;
              tfmin5 = form;
            }
          }

          if (flag == 6) {
            if (xkt > xktmax6 && delta_R <= 0.4) {
              xktmax6 = xkt;
              tfmin6 = form;
            }
          }

          niter = niter + 1;
          // cout<<xkt<<" "<<" "<<y<<" "<<niter<<" "<<j1.perp()<<" "<<j2.perp()<<" "<<form<<endl;
          // cout<<"status"<<pythia.event[index1].status()<<" "<<pythia.event[index2].status()<<endl;
        }
      }

      // cout<<"right after graph"<<endl;
      // if(xktmax5==-20) cout<<" "<<niter<<endl;

      // cout<<xktmax6<<" xktmin"<<tfmin6<<endl;

      if (pythia.event[i].isFinal()) {
        // Apply cuts in the particles
        if (pythia.event[i].pT() < trackLowPtCut)
          continue; // pt cut
        if (TMath::Abs(pythia.event[i].eta()) > trackEtaCut)
          continue; // eta cut

        fourvec[0] = pythia.event[i].px();
        fourvec[1] = pythia.event[i].py();
        fourvec[2] = pythia.event[i].pz();
        fourvec[3] = pythia.event[i].e();

        fastjet::PseudoJet PythiaParticle(fourvec);

        fjInputs1.push_back(PythiaParticle);
      }
    } // end of loop over particle in the event

    // Jet Reconstruction:
    std::vector<fastjet::PseudoJet> PythiaJets; // Declaration of vector for Reconstructed Jets

    fastjet::GhostedAreaSpec ghost_spec(1, 1, 0.05); // Ghosts to calculate the Jet Area

    fastjet::AreaDefinition fAreaDef(fastjet::passive_area, ghost_spec); // Area Definition

    fastjet::ClusterSequenceArea clustSeq_Sig(fjInputs1, *jetDefAKT_Sig, fAreaDef); // Cluster Sequence

    PythiaJets = sorted_by_pt(clustSeq_Sig.inclusive_jets(1.)); // Vector with the Reconstructed Jets in pT order

    if (PythiaJets.size() == 0)
      continue;

    Int_t flago = 0;
    fastjet::PseudoJet ProbeJet = PythiaJets[0]; // Hardest Pythia Jet
    cout << "the jet pt " << ProbeJet.perp() << endl;
    if (ProbeJet.perp() < 80)
      continue;
    fastjet::PseudoJet parton6(pythia.event[5].px(), pythia.event[5].py(), pythia.event[5].pz(),
                  pythia.event[5].e());
    fastjet::PseudoJet parton7(pythia.event[6].px(), pythia.event[6].py(), pythia.event[6].pz(),
                  pythia.event[6].e());
    Double_t deltaR1 = ProbeJet.delta_R(parton6);
    Double_t deltaR2 = ProbeJet.delta_R(parton7);

    if (deltaR1 < 0.1)
      flago = 5;
    if (deltaR2 < 0.1)
      flago = 6;
    if (flago == 5 && xktmax5 == -20)
      continue;
    if (flago == 6 && xktmax6 == -20)
      continue;
    if (flago == 0)
      continue;
    // Fill "B-Histograms"

    hJetPt_B->Fill(ProbeJet.pt());

    hJetArea_B->Fill(ProbeJet.area());
    cout << "flago" << flago << "i am here" << endl;
    if (flago == 5)
      IterativeDeclustering(ProbeJet, 0, flago, xktmax5,
                 tfmin5); // Fill the deltaR and groomed deltaR "B-Histograms"
    if (flago == 6)
      IterativeDeclustering(ProbeJet, 0, flago, xktmax6, tfmin6);
    //_________________________________________________________________________________

    // 2nd Step: Thermal Particles + Probe Particles + FastJet (again) + CONSTITUENTS SUBTRACTION -> New Jets
    fjInputs2.resize(0);

    // Thermal Particles loop
    for (int j = 0; j < nThermalParticles; j++) {
      double pT = f_pT->GetRandom();

      double eta = f_eta->GetRandom();

      double phi = f_phi->GetRandom();

      if (pT < trackLowPtCut)
        continue; // pt cut

      hT_pT->Fill(pT);
      hT_eta->Fill(eta);
      hT_phi->Fill(phi);

      h_pT->Fill(pT);
      h_eta->Fill(eta);
      h_phi->Fill(phi);

      fourvec[0] = Calculate_pX(pT, eta, phi);
      fourvec[1] = Calculate_pY(pT, eta, phi);
      fourvec[2] = Calculate_pZ(pT, eta, phi);
      fourvec[3] = Calculate_E(pT, eta, phi);

      fastjet::PseudoJet ThermalParticle(fourvec);

      ThermalParticle.set_user_index(0);

      fjInputs2.push_back(ThermalParticle);
    }

    // Probe Particles loop

    std::vector<fastjet::PseudoJet> ProbeParticles = sorted_by_pt(ProbeJet.constituents());

    for (int j = 0; j < ProbeParticles.size(); j++) {
      hP_pT->Fill(ProbeParticles[j].pt());
      hP_eta->Fill(ProbeParticles[j].eta());
      hP_phi->Fill(ProbeParticles[j].phi());

      h_pT->Fill(ProbeParticles[j].pt());
      h_eta->Fill(ProbeParticles[j].eta());
      h_phi->Fill(ProbeParticles[j].phi());

      ProbeParticles[j].set_user_index(1);

      fjInputs2.push_back(ProbeParticles[j]);
    }

    //________________________

    // Jet Reconstruction:
    // fastjet::AreaType areaType = fastjet::active_area;

    std::vector<fastjet::PseudoJet> NewJets; // Declaration of vector for Reconstructed Jets

    fastjet::GhostedAreaSpec New_ghost_spec(1, 1, 0.05); // Ghosts to calculate the Jet Area

    fastjet::AreaDefinition New_fAreaDef(fastjet::active_area_explicit_ghosts, New_ghost_spec); // Area Definition

    fastjet::ClusterSequenceArea New_clustSeq_Sig(fjInputs2, *jetDefAKT_Sig, New_fAreaDef); // Cluster Sequence

    NewJets = New_clustSeq_Sig.inclusive_jets(1.); // Vector with the Reconstructed Jets

    // //________________________

    // //CONSTITUENTS SUBTRACTION JET BY JET:

    fastjet::JetMedianBackgroundEstimator bge;

    fastjet::Selector BGSelector = fastjet::SelectorAbsEtaMax(1.0);

    fastjet::JetDefinition jetDefBG(fastjet::kt_algorithm, jetParameterR, recombScheme, strategy);

    fastjet::AreaDefinition fAreaDefBG(fastjet::active_area_explicit_ghosts, New_ghost_spec);

    fastjet::ClusterSequenceArea clustSeqBG(fjInputs2, jetDefBG, fAreaDefBG);

    std::vector<fastjet::PseudoJet> BGJets = clustSeqBG.inclusive_jets();

    bge.set_selector(BGSelector);

    bge.set_jets(BGJets);

    fastjet::contrib::ConstituentSubtractor subtractor(&bge);

    subtractor.set_common_bge_for_rho_and_rhom(true);
    // // for massless input particles it does not make any difference (rho_m is always zero)

    subtractor.set_max_standardDeltaR(jetParameterR);
    // subtractor.set_alpha(0.5);

    for (int j = 0; j < NewJets.size(); j++) {
      // SUBTRACTION HERE:

      const fastjet::PseudoJet& jet = NewJets[j];

      fastjet::PseudoJet subtracted_Jet = subtractor(jet);

      NewJets[j] = subtracted_Jet;
    }
    // //______________

    // 3rd Step: MATCHING := Find in NewJets the Jet with the constituents that satisfy the condition
    //#Sigma_{constituents with index=1} p_{T}_{constituent} >= 0.5*p_{T}_{ProbeJet}

    int MATCH = -1;

    for (int j = 0; j < NewJets.size(); j++) {
      fastjet::PseudoJet NJet = NewJets.at(j);

      if (!EtaCut(NJet, etamin_Sig, etamax_Sig))
        continue;
      if (NJet.pt() < 0.5 * ProbeJet.pt())
        continue;

      double MATCH_pT = 0.0;

      std::vector<fastjet::PseudoJet> constituents = sorted_by_pt(NJet.constituents());

      for (int k = 0; k < constituents.size(); k++)
        if (constituents.at(k).user_index() == 1)
          MATCH_pT += constituents.at(k).pt();

      if (MATCH_pT >= 0.5 * ProbeJet.pt()) {
        MATCH = j;
        break;
      }
    }

    if (MATCH > 0) {
      fastjet::PseudoJet MatchedJet = NewJets.at(MATCH);

      // Fill "A-Histograms"

      hJetPt_A->Fill(MatchedJet.pt());

      // hJetArea_A->Fill(MatchedJet.area());

      if (flago == 5)
        IterativeDeclustering(MatchedJet, 1, flago, xktmax5,
                   tfmin5); // Fill the deltaR and groomed deltaR "A-Histograms"
      if (flago == 6)
        IterativeDeclustering(MatchedJet, 1, flago, xktmax6, tfmin6);
    }

    else
      continue;
    //_________________________________________________________________________________
  }
  // End event loop

  TFile* outFile = new TFile(Form("OutputFrac_Pythia_BG_JetCS%d.root", cislo), "RECREATE");

  outFile->cd();

  fHLundIterative->Write();
  fHLundIterativeOriginal->Write();
  fh2CorrHardProbe->Write();
  fh2CorrHardHybrid->Write();
  fh2CorrTimeHybrid->Write();
  fh2CorrTimeProbe->Write();

  fh2CorrHardProbeZ->Write();
  fh2CorrHardHybridZ->Write();
  fh2CorrTimeHybridZ->Write();
  fh2CorrTimeProbeZ->Write();

  fh2CorrHardProbeRad->Write();
  fh2CorrHardHybridRad->Write();
  fh2CorrTimeHybridRad->Write();
  fh2CorrTimeProbeRad->Write();

  outFile->Close();

  return 0;
}
