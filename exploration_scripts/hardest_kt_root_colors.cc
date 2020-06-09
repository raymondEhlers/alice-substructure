#include <TColor.h>

void defineColors() {
  // Dynnamical Grooming
  // Greens of different shades, since they are related methods, but different.
  Int_t ci = TColor::GetFreeColorIndex();
  TColor dynamicalZ(ci, 0.20627450980392156, 0.37150326797385624, 0.25333333333333335);
  ci = TColor::GetFreeColorIndex();
  TColor dynamicalKt(ci, 0.21254901960784311, 0.5430065359477124, 0.30666666666666664);
  ci = TColor::GetFreeColorIndex();
  TColor dynamicalTime(ci, 0.3087581699346405, 0.6773856209150326, 0.3979084967320261);

  // Leading kt
  // Purple
  ci = TColor::GetFreeColorIndex();
  TColor leadingKt(ci, 0.4743919005510701, 0.4340381904395745, 0.7000640779187491);

  // Soft Drop
  // Red
  ci = TColor::GetFreeColorIndex();
  TColor softDrop(ci, 0.8905805459438677, 0.18734845572215816, 0.1543355119825708);

  // Markers
  Int_t noZCut = kFullCircle;
  Int_t zCut02 = kOpenDiamond;
  Int_t zCut04 = kFullSquare;
}
