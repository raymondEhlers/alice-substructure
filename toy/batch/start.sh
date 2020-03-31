#!/usr/bin/env bash

eval `alienv load AliPhysics/latest`

export CGAL_ROOT="$HOME/alice/sw/ubuntu1804_x86-64/cgal/latest"
export GMP_ROOT="$HOME/alice/sw/ubuntu1804_x86-64/GMP/latest"
export PYTHIA8_DIR="$HOME/install"

TUNE=5
NEVENTS=50000
INCLUDE_UNDERLYING_EVENT=1
JET_R=0.8
PARTON_DELTA_R_MAX=0.1
STORE_RECURSIVE_SPLITTINGS=0
TAKE_FIRST_TRUE_SPLITTING=0

for SEED in {1..4};
do
    # Usage:
    # ./pygen <PythiaTune> <Seed> <nEvts> <underlyingEvent> <jetR> <partonDeltaRMax> <storeRecursiveSplittings> <takeFirstTrueSplitting>
    $HOME/code/alice/substructure/toy/bin/groomingToy $TUNE $SEED $NEVENTS $INCLUDE_UNDERLYING_EVENT $JET_R $PARTON_DELTA_R_MAX $STORE_RECURSIVE_SPLITTINGS $TAKE_FIRST_TRUE_SPLITTING &> ${SEED}.log &
done;
