cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(groomingToy)
# Information CMake where to find additional package modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Set policies
# CMP0042 - MACOSX_RPATH is enabled by default (v3.0)
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

# RPATH settings
# Enable MACOS_RPATH
set(CMAKE_MACOSX_RPATH TRUE)

# Additional settings from: https://cmake.org/Wiki/CMake_RPATH_handling#Always_full_RPATH
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Set install RPATH
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

# Ensure compilation with c++11
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(CMAKE_CXX_COMPILER_ID EQUAL Clang)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif(CMAKE_CXX_COMPILER_ID EQUAL Clang)

# ----- Deal with external packages -----

# Tell CMake how to find ROOT
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

#---Locate the ROOT package and defines a number of variables (e.g. ROOT_INCLUDE_DIRS)
find_package(ROOT REQUIRED COMPONENTS RIO)
message(STATUS "Found ROOT version ${ROOT_VERSION}")

#---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

# Pythia8
find_package(Pythia8 REQUIRED)
# Fastjet
find_package(FastJet REQUIRED)

# Add include and src directories
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
