
#include <TFile.h>
#include <TTree.h>

class TestObject : public TNamed {
 public:
  TestObject(): TNamed("testObject", "testObject"), a(0), b(0) {}
  TestObject(int a_, double b_): TNamed("testObject", "testObject"), a(a_), b(b_) {}
  void SetA(int a_) { a = a_; }
  void SetB(double b_) { b = b_; }

 private:
  int a;     ///< A
  double b;  ///< B

  ClassDef(TestObject, 1);
};

void testWritingObjectToTree() {
  TestObject testObject;
  TestObject testObject2;
  TTree t("tree", "tree");
  t.Branch("testObj.", &testObject, 32000, 2);
  t.Branch("testObj2.", &testObject2, 32000, 2);

  for (unsigned int i = 0; i < 10; i++)
  {
    testObject.SetA(i);
    testObject.SetB(i*i);
    testObject2.SetA(i*i*i);
    testObject2.SetB(i*i*i*i);

    t.Fill();
  }

  TFile f("testTree.root", "RECREATE");
  t.Write();
  f.Close();
}
