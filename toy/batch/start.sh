#!/usr/bin/env bash

SEED=$SLURM_ARRAY_TASK_ID

eval `/usr/local/bin/alienv -w ${ALIBUILD_WORK_DIR} --no-refresh printenv AliPhysics/latest`

export CGAL_ROOT="$ALIBUILD_WORK_DIR/ubuntu1804_x86-64/cgal/latest"
export GMP_ROOT="$ALIBUILD_WORK_DIR/ubuntu1804_x86-64/GMP/latest"
export PYTHIA8_DIR="$INSTALL"

# Setup output
BASE_DIR="/clusterfs4/rehlers/substructure/toy"
OUTPUT_DIR="${BASE_DIR}/output/${SLURM_ARRAY_JOB_ID}"
mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

TUNE=5
NEVENTS=20000
INCLUDE_UNDERLYING_EVENT=1
JET_R=0.8
PARTON_DELTA_R_MAX=0.1
STORE_RECURSIVE_SPLITTINGS=0
TAKE_FIRST_TRUE_SPLITTING=0

# Usage:
# ./pygen <PythiaTune> <Seed> <nEvts> <underlyingEvent> <jetR> <partonDeltaRMax> <storeRecursiveSplittings> <takeFirstTrueSplitting>
$BASE_DIR/bin/groomingToy $TUNE $SEED $NEVENTS $INCLUDE_UNDERLYING_EVENT $JET_R $PARTON_DELTA_R_MAX $STORE_RECURSIVE_SPLITTINGS $TAKE_FIRST_TRUE_SPLITTING &> ${SEED}.log
