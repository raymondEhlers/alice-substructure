g++ ${1}.cc \
 -O2 -ansi -W -Wall -std=c++11 -Wshadow -m64 -Wno-shadow \
 -o ${1}.exe \
         -I$HEPMC_ROOT/include -L$HEPMC_ROOT/lib -lHepMC \
	 -I$PYTHIA_ROOT/include -L$PYTHIA_ROOT/lib/ -lpythia8 \
 `fastjet-config --cxxflags --libs` -lfastjetcontribfragile \
 -L$CGAL_ROOT/lib \
 -L/usr/local/lib/ \
 `root-config --cflags --ldflags --glibs ` -lEG -lz
