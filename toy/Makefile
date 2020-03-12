# Based on https://www.partow.net/programming/makefile/index.html

CXX      := -c++
#CXXFLAGS := -pedantic-errors -Wall -Wextra -Werror --std=c++11 -W-no-unused-command-line-argument
CXXFLAGS := -Wall --std=c++11 -Wno-unused-command-line-argument
LDFLAGS  := -L/usr/lib -lstdc++ -lm $(shell pythia8-config --libs) -L${CGAL_ROOT}/lib -L/usr/local/lib/ $(shell fastjet-config --libs) -lfastjetcontribfragile $(shell root-config --libs)
BUILD    := ./build
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)/apps
TARGET   := groomingToy
INCLUDE  := -Iinclude/ $(shell pythia8-config --cxxflags) $(shell fastjet-config --cxxflags) $(shell root-config --cflags)
SRC      :=                      \
   $(wildcard src/*.cc)          \

OBJECTS  := $(SRC:%.cc=$(OBJ_DIR)/%.o)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@ $(LDFLAGS)

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

.PHONY: all build clean debug release

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

