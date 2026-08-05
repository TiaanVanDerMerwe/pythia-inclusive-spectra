CXX     = g++
PYTHIA  = /home/tiaan/pythia8316

CXXFLAGS = -O2 -std=c++17 $(shell $(PYTHIA)/bin/pythia8-config --cxxflags)
LDFLAGS  = $(shell $(PYTHIA)/bin/pythia8-config --ldflags --libs) -pthread

# --------------------------------------------------
# Directories
# --------------------------------------------------
SRC_DIR = src
BIN_DIR = bin

# Targets (executables)
TARGETS = chargedSingleSpec
BINS    = $(addprefix $(BIN_DIR)/, $(TARGETS))

# Default target
all: $(BIN_DIR) $(BINS)

# Ensure bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build rule: src/foo.cc -> bin/foo
$(BIN_DIR)/%: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

# --------------------------------------------------
# Run helpers
# --------------------------------------------------
run: $(BIN_DIR)/chargedSingleSpec
	./$(BIN_DIR)/chargedSingleSpec $(ARGS)

# --------------------------------------------------
# Clean
# --------------------------------------------------
clean:
	rm -rf $(BIN_DIR)



