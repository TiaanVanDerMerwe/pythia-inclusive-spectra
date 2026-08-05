/**
 * @file exampleSingleSpec.cc
 * @brief Generates proton-proton collision events with continuous biasing using Pythia8
 * 
 * @details This program generates pp collision events across a single pTHat bins to efficiently
 * sample the full pT spectrum. It implements:
 * - continuous differential cross section biasing with overlap removal for soft/hard QCD transitions
 * - Event weighting for biased phase space sampling
 * - Primary particle selection (excluding weak decay products)
 * - Detector acceptance cuts (eta, pT thresholds)
 * 
 * Output: CSV file containing:
 * 1. Event table with ALL events and their weights
 * 2. Charged hadron kinematics (only pT > threshold)
 * and metadata required for proper cross-section normalization.
 * 
 * @author Tiaan van der Merwe tiaanvdmerwe361@gmail.com
 * @date 08/01/2026
 */
#include "Pythia8/Pythia.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <iomanip>

using namespace Pythia8;

/**
 * @brief Filters primary particles by rejecting weak decay products
 * 
 * @details Primary particles are defined as particles that:
 * 1. Have sufficiently long lifetimes (tau0 >= 10 mm/c)
 * 2. Do not originate from weak decays of long-lived hadrons
 * 
 * This filter is essential for matching experimental detector capabilities,
 * which typically cannot distinguish decay products from the primary vertex.
 * 
 * @param p The particle to be tested
 * @param event The full event record (needed to access mother particles)
 * 
 * @return true if particle is primary, false if it's a secondary decay product
 * 
 * @note Particles with tau0 = 0 (stable particles) are always considered primary
 */
bool primaryFilter(const Pythia8::Particle& p,
                   const Pythia8::Event& event) 
{
  // Reject particles with short lifetimes (weak decay products)
  // tau0 < 10 mm/c indicates secondary particles
  if (p.tau0() < 10.0 && p.tau0() != 0.0) 
    return false;

  // Check mother particles to identify decay products
  for (int iMom = p.mother1(); iMom <= p.mother2(); ++iMom) {
      if (iMom <= 0 || iMom >= event.size()) 
        continue;

      const Particle& mother = event[iMom];

      // Reject if mother is a long-lived hadron (indicates weak decay chain)
      if (mother.isHadron() && mother.tau0() > 10.0)
        return false;
  }
  return true;
}


std::string makeFilename(double COM, double POWER, double SEED)
{
    // Format time
    std::ostringstream oss;
    oss << "pythiaData/" << std::to_string((int)COM) << "/charged/" << std::to_string((int)SEED) << "/particles_pow" << std::to_string((int)POWER) << ".csv";

    return oss.str();
}

/**
 * @brief Main event generation program with pTHat binning
 * 
 * @details Program flow:
 * 1. Collect user input (COM energy, eta acceptance, events per bin)
 * 2. Configure Pythia8 with minimal verbosity
 * 3. For the pTHat bin:
 *    - Generate events with continuous biasing
 *    - Store ALL event weights (for proper normalization)
 *    - Select charged primary hadrons within acceptance (pT > 5 GeV)
 *    - Write event table and particle data
 * 
 * @return 0 on success, 1 on initialization failure
 * 
 * @note Output files: {COM}/charged/particles_pow{POWER}.csv
 * @warning Requires output directory to exist before running
 */
int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------
    // User input: collision parameters from command line
    // ------------------------------------------------------------------
    
  if (argc != 5) {
      std::cerr << "Usage: " << argv[0] 
                << " <COM_energy> <pseudorapidity> <num_events> <biasing_power>\n";
      return 1;
  }
    
  double COM = std::atof(argv[1]);
  double ETA = std::atof(argv[2]);
  int NEVENTS = std::atoi(argv[3]);
  double POWER = std::atof(argv[4]);
  const double PT_THRESHOLD = 5.0;  // Only write hadrons above this pT
  double SEED = 650765638;

  if (COM <= 0 || ETA <= 0 || NEVENTS <= 0) {
    std::cerr << "Invalid parameters\n";
    return 1;
  }

  Pythia pythia;
  Settings& settings = pythia.settings;
  const Info& info   = pythia.info;

  // ------------------------------------------------------------------
  // Reduce runtime verbosity
  // ------------------------------------------------------------------
  pythia.readString("Init:showProcesses = off");
  pythia.readString("Init:showMultipartonInteractions = off");
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showChangedParticleData = off");

  pythia.readString("Random:setSeed = on");
  settings.parm("Random:seed", SEED);

  pythia.readString("Next:numberCount = 1000");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  // ------------------------------------------------------------------
  // Beam setup: proton–proton collisions
  // ------------------------------------------------------------------
  pythia.settings.parm("Beams:eCM", COM);
  pythia.readString("Tune:pp = 14");
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");

  // ------------------------------------------------------------------
  // HardQCD on
  // ------------------------------------------------------------------
  pythia.readString("HardQCD:all = on");
  pythia.readString("PartonLevel:MPI = on");
  pythia.readString("PartonLevel:ISR = on");
  pythia.readString("PartonLevel:FSR = on");
  pythia.readString("HadronLevel:Hadronize = on");
  pythia.readString("HadronLevel:Decay = on");

  // ------------------------------------------------------------------
  // Biasing settings
  // ------------------------------------------------------------------
  std::vector<double> pTlimit = {3., -1.};
  int nBin = pTlimit.size() - 1;
  int nEvents = NEVENTS / nBin;

  pythia.readString("PhaseSpace:bias2Selection = on");
  settings.parm("PhaseSpace:bias2SelectionPow", POWER);
  settings.parm("PhaseSpace:bias2SelectionRef", 15.);

  // ------------------------------------------------------------------
  // Output file setup
  // ------------------------------------------------------------------
  std::string fname = makeFilename(COM, POWER, SEED);
  std::ofstream out(fname);
  out << std::scientific << std::setprecision(6);
  
  if (!out.is_open()) {
    std::cerr << "Cannot open output file\n";
    return 1;
  }
  
  const size_t bufferSize = 64 * 1024 * 1024;
  std::vector<char> fileBuffer(bufferSize);
  out.rdbuf()->pubsetbuf(fileBuffer.data(), bufferSize);
  
  // ------------------------------------------------------------------
  // Write metadata header
  // ------------------------------------------------------------------
  out << "# ETA: " << ETA << "\n";
  out << "# NEVENTS: " << NEVENTS << "\n";
  out << "# MODE: "  << 3 << "\n";
  out << "# POWER: " << POWER << "\n";
  out << "# PREF: "  << 15. << "\n";

  // Buffers for event table and hadron data
  std::ostringstream eventBuffer;
  std::ostringstream hadronBuffer;

  eventBuffer << std::scientific << std::setprecision(6);
  hadronBuffer << std::scientific << std::setprecision(6);

  int eventCount = 0;
  int hadronCount = 0;
  int globalEvent = 0;

  // ------------------------------------------------------------------
  // pTHat bin loop
  // ------------------------------------------------------------------
  for (int iBin = 0; iBin < nBin; ++iBin) {
    
    settings.parm("PhaseSpace:pTHatMin", pTlimit[iBin]);
    settings.parm("PhaseSpace:pTHatMax", pTlimit[iBin + 1]);

    if (!pythia.init()) {
      std::cerr << "Pythia initialization failed\n";
      return 1;
    }

    double sumWeights = 0.0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

      if (!pythia.next()) continue;
      globalEvent++;

      double eventWeight = info.weight();
      sumWeights += eventWeight;

      // Store ALL events in event table
      eventBuffer << globalEvent << "," << eventWeight << "\n";
      ++eventCount;

      // Loop over all particles in the event record
      for (int i = 0; i < pythia.event.size(); ++i) {

        const Particle& p = pythia.event[i];
        
        // ----------------------------------------------------------
        // Particle selection criteria
        // ----------------------------------------------------------

        if (p.pT() < PT_THRESHOLD) continue;
        if (!p.isFinal()) continue;
        if (!p.isCharged()) continue;
        if (!p.isHadron())  continue;
        if (std::abs(p.eta()) > ETA) continue;
        if (!primaryFilter(p, pythia.event)) continue;
        
        hadronBuffer << globalEvent << "," << iBin << "," 
                     << p.pT() << "," << eventWeight << "\n";
        
        ++hadronCount;
        
      }
    }

    // ------------------------------------------------------------------
    // Write bin metadata
    // ------------------------------------------------------------------
    out << "# sigmaGEN: " << info.sigmaGen() << "\n";
    out << "# weightSum: "  << sumWeights << "\n";
  }

  // ------------------------------------------------------------------
  // Write event table and hadron data to file
  // ------------------------------------------------------------------
  out << "\n# Event table\n";
  out << eventBuffer.str();
  
  out << "\n# Hadron data\n";
  out << "event,bin,pT,weight\n";
  out << hadronBuffer.str();

  std::cout << "\n=== All pTHat bins completed successfully ===\n";
  std::cout << "Total events: " << globalEvent << "\n";
  std::cout << "High-pT hadrons written: " << hadronCount << "\n";
  
  out.close();
  return 0;
}