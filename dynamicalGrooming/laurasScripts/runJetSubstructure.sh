
#!/bin/bash

if [ -z "$1" ]; then
	DATATYPE="AOD"
else
	DATATYPE="${1}"
fi

if [ -z "$2" ]; then
	PERIOD="lhc18r"
else
	PERIOD="${2}"
fi

if [ -z "$3" ]; then
    FILELIST="LHC18rAOD.txt"
else
	FILELIST="${3}"
fi

if [ -z "$4" ]; then
	NEVENTS="100"
else
	NEVENTS="${4}"
fi

root -b -q -x runJetSubstructure.C\(\""${DATATYPE}"\",\""${PERIOD}"\",\""${FILELIST}"\","${NEVENTS}"\)
