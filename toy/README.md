# Build notes

Need to export `CGAL_ROOT`, `GMP_ROOT`, and `PYTHIA8_DIR`!

For example,

```bash
$ export CGAL_ROOT=$HOME/alice/sw/osx_x86-64/cgal/latest
$ export GMP_ROOT=$HOME/alice/sw/osx_x86-64/GMP/latest
$ export PYTHIA8_DIR=$HOME/install
```

Cmake configuration with (debug is optional):

```bash
$ mkdir build && cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX="../" -DCMAKE_BUILD_TYPE=Debug
```
