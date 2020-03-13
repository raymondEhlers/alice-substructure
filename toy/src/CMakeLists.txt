cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(MODULE groomingToy)

# Add wanning options
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include_directories(
  # Variable should be plural to support both ROOT build methods...
  ${ROOT_INCLUDE_DIRS}
  ${PYTHIA8_INCLUDE_DIRS}
  ${FASTJET_INCLUDE_DIRS}
  )

set(SRCS
  output.cc
  groomingToy.cc
  )

link_directories(
  ${ROOT_LIBRARY_DIR}
  ${PYTHIA8_LIBRARY_DIR}
  ${FASTJET_LIBRARY_DIR}
  )

# Generate dictioanry
ROOT_GENERATE_DICTIONARY(G__output output.h LINKDEF LinkDef.h)

add_library(output SHARED output.cc G__output.cxx)
target_link_libraries(output ${ROOT_LIBRARIES} FastJet::FastJet)

add_executable(${MODULE} ${SRCS})
target_link_libraries(${MODULE} ${ROOT_LIBRARIES} FastJet::FastJet ${PYTHIA8_LIBRARIES})

install(TARGETS ${MODULE} DESTINATION bin)
