
#include <iostream>

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>

void crossCheck() {
  auto f = TFile::Open("unfolding_leading_kt_z_cut_02_test.root");
  auto raw = dynamic_cast<TH2D*>(f->Get("raw"));
  auto refolded_6 = dynamic_cast<TH2D*>(f->Get("Bayesian_Foldediter6"));

  std::cout << "Before projection\n";
  auto raw_kt = raw->ProjectionX("raw_kt", raw->GetYaxis()->FindBin(40), raw->FindBin(120));
  auto refolded_6_kt = refolded_6->ProjectionX("refolded_6_kt", refolded_6->GetYaxis()->FindBin(40), refolded_6->GetYaxis()->FindBin(120));

  std::cout << "before divide\n";
  refolded_6_kt->Divide(raw_kt);

  TCanvas c("c", "c");
  refolded_6_kt->Draw();
  c.SaveAs("test.pdf");
}
