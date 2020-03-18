#pragma once

#include <ostream>
#include <string>
#include <vector>

#include <Rtypes.h>

#include <fastjet/PseudoJet.hh>

/**
 * These classes were copied directly over from AliPhysics on 12 March 2020.
 * They are slightly modified so we don't have to depend on AliPhysics to make life easier!
 *
 * Most significant changes:
 *
 * - Make filling the recursive splittings into a method of JetSubstructureSplittings.
 */

namespace SubstructureTree {
  class Subjets;
  class JetSplittings;
  class JetConstituents;
  class JetSubstructureSplittings;
}

std::ostream& operator<<(std::ostream& in, const SubstructureTree::Subjets& myTask);
std::ostream& operator<<(std::ostream& in, const SubstructureTree::JetSplittings& myTask);
std::ostream& operator<<(std::ostream& in, const SubstructureTree::JetConstituents& myTask);
std::ostream& operator<<(std::ostream& in, const SubstructureTree::JetSubstructureSplittings& myTask);
void swap(SubstructureTree::Subjets& first,
     SubstructureTree::Subjets& second);
void swap(SubstructureTree::JetSplittings& first,
     SubstructureTree::JetSplittings& second);
void swap(SubstructureTree::JetConstituents& first,
     SubstructureTree::JetConstituents& second);
void swap(SubstructureTree::JetSubstructureSplittings& first,
     SubstructureTree::JetSubstructureSplittings& second);

namespace SubstructureTree {

class Subjets {
 public:
  // TODO: Fully update and document!
  Subjets();
  // Additional constructors
  Subjets(const Subjets & other);
  Subjets& operator=(Subjets other);
  friend void ::swap(Subjets & first, Subjets & second);
  // Avoid implementing move since c++11 is not allowed in the header
  virtual ~Subjets() = default;

  /// Reset the properties for the next filling of the tree.
  bool Clear();

  // Getters and setters
  void AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting,
          const std::vector<unsigned short>& constituentIndices);
  #if !(defined(__CINT__) || defined(__MAKECINT__))
  std::tuple<unsigned short, bool, const std::vector<unsigned short>> GetSubjet(int i) const;
  #endif

  // Printing
  std::string toString() const;
  friend std::ostream & ::operator<<(std::ostream &in, const Subjets &myTask);
  void Print(Option_t* opt = "") const;
  std::ostream & Print(std::ostream &in) const;

 protected:
  std::vector<unsigned short> fSplittingNodeIndex;        ///<  Index of the parent splitting node.
  std::vector<bool> fPartOfIterativeSplitting;            ///<  True if the splitting is follow an iterative splitting.
  std::vector<std::vector<unsigned short>> fConstituentIndices;        ///<  Constituent jet indices (ie. index by the stored jet constituents, not the global index).

  /// \cond CLASSIMP
  ClassDef(Subjets, 2) // Subjets from splittings.
  /// \endcond
};

class JetSplittings {
 public:
  JetSplittings();
  // Additional constructors
  JetSplittings(const JetSplittings & other);
  JetSplittings& operator=(JetSplittings other);
  friend void ::swap(JetSplittings & first, JetSplittings & second);
  // Avoid implementing move since c++11 is not allowed in the header
  virtual ~JetSplittings() = default;

  /// Reset the properties for the next filling of the tree.
  bool Clear();

  // Getters and setters
  void AddSplitting(float kt, float deltaR, float z, short parentIndex);
  #if !(defined(__CINT__) || defined(__MAKECINT__))
  std::tuple<float, float, float, short> GetSplitting(int i) const;
  #endif
  unsigned int GetNumberOfSplittings() const { return fKt.size(); }

  // Printing
  std::string toString() const;
  friend std::ostream & ::operator<<(std::ostream &in, const JetSplittings &myTask);
  void Print(Option_t* opt = "") const;
  std::ostream & Print(std::ostream &in) const;

 protected:
  std::vector<float> fKt;             ///<  kT between the subjets.
  std::vector<float> fDeltaR;         ///<  Delta R between the subjets.
  std::vector<float> fZ;              ///<  Momentum sharing of the splitting.
  std::vector<short> fParentIndex;    ///<  Index of the parent splitting.

  /// \cond CLASSIMP
  ClassDef(JetSplittings, 1) // Jet splittings.
  /// \endcond
};

class JetConstituents
{
 public:
  // TODO: Fully update and document!
  JetConstituents();
  // Additional constructors
  JetConstituents(const JetConstituents & other);
  JetConstituents& operator=(JetConstituents other);
  friend void ::swap(JetConstituents & first, JetConstituents & second);
  // Avoid implementing move since c++11 is not allowed in the header
  virtual ~JetConstituents() = default;

  /// Reset the properties for the next filling of the tree.
  bool Clear();

  // Getters and setters
  void AddJetConstituent(const fastjet::PseudoJet& part);
  #if !(defined(__CINT__) || defined(__MAKECINT__))
  std::tuple<float, float, float, int> GetJetConstituent(int i) const;
  #endif
  unsigned int GetNumberOfJetConstituents() const { return fPt.size(); }

  // Printing
  std::string toString() const;
  friend std::ostream & ::operator<<(std::ostream &in, const JetConstituents &myTask);
  void Print(Option_t* opt = "") const;
  std::ostream & Print(std::ostream &in) const;

 protected:
  std::vector<float> fPt;                 ///<  Jet constituent pt
  std::vector<float> fEta;                ///<  Jet constituent eta
  std::vector<float> fPhi;                ///<  Jet constituent phi
  std::vector<unsigned int> fGlobalIndex; ///<  Jet constituent global index

  /// \cond CLASSIMP
  ClassDef(JetConstituents, 1) // Jet constituents.
  /// \endcond
};

/**
 * @class JetSubstructureSplittings
 * @brief Jet substructure splittings.
 *
 * Jet substructure splitting properties. There is sufficient information to calculate any
 * additional splitting properties.
 *
 * @author Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
 * @date 9 Feb 2020
 */
class JetSubstructureSplittings {
 public:
  JetSubstructureSplittings();
  // Additional constructors
  JetSubstructureSplittings(const JetSubstructureSplittings & other);
  JetSubstructureSplittings& operator=(JetSubstructureSplittings other);
  friend void ::swap(JetSubstructureSplittings & first, JetSubstructureSplittings & second);
  // Avoid implementing move since c++11 is not allowed in the header
  virtual ~JetSubstructureSplittings() = default;

  /// Reset the properties for the next filling of the tree.
  bool Clear();

  // Setters
  void SetJetPt(float pt) { fJetPt = pt; }
  void AddJetConstituent(const fastjet::PseudoJet& part);
  void AddSplitting(float kt, float deltaR, float z, short parentIndex);
  void AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting,
          const std::vector<unsigned short>& constituentIndices);
  // Getters
  float GetJetPt() { return fJetPt; }
  #if !(defined(__CINT__) || defined(__MAKECINT__))
  std::tuple<float, float, float, int> GetJetConstituent(int i) const;
  std::tuple<float, float, float, short> GetSplitting(int i) const;
  std::tuple<unsigned short, bool, const std::vector<unsigned short>> GetSubjet(int i) const;
  #endif
  unsigned int GetNumberOfJetConstituents() const { return fJetConstituents.GetNumberOfJetConstituents(); }
  unsigned int GetNumberOfSplittings() { return fJetSplittings.GetNumberOfSplittings(); }

  // Printing
  std::string toString() const;
  friend std::ostream & ::operator<<(std::ostream &in, const JetSubstructureSplittings &myTask);
  void Print(Option_t* opt = "") const;
  std::ostream & Print(std::ostream &in) const;

 private:
  // Jet properties
  float fJetPt;                                           ///<  Jet pt.
  JetConstituents fJetConstituents;     ///<  Jet constituents
  JetSplittings fJetSplittings;         ///<  Jet splittings.
  Subjets fSubjets;                     ///<  Subjets within the jet.

  /// \cond CLASSIMP
  ClassDef(JetSubstructureSplittings, 2) // Jet splitting properties.
  /// \endcond
};

} /* namespace SubstructureTree */
