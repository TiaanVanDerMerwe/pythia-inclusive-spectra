#include "Pythia8/Pythia.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <chrono>
#include <iomanip>

using namespace Pythia8;

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


std::string makeFilename()
{
    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);

    // Format time
    std::ostringstream oss;
    oss << "data/events_HadronsPT_mode3_pow7_singlebin_"
        << std::put_time(&tm, "%H-%M-%S")
        << ".csv";

    return oss.str();
}

int main() {
  // Hard QCD for pTHat ≥ 20 GeV.
  // with pTHat biasing applied only in the hard region.
  double NEVENTS, POWER, PREF;
  
  std::cout << "Number of events: ";
  std::cin >> NEVENTS;
  
  std::cout << "Biasing power: ";
  std::cin >> POWER;
  
  std::cout << "Biasing momentum: ";
  std::cin >> PREF;

  Pythia pythia;
  Settings& settings = pythia.settings;
  const Info& info   = pythia.info;

  pythia.readString("Init:showProcesses = off");
  pythia.readString("Init:showMultipartonInteractions = off");
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showChangedParticleData = off");

  // Initialize for LHC at 7 TeV.
  pythia.readString("Beams:eCM = 7000.");
  pythia.readString("Tune:pp = 14");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 700");

  std::vector<double> pTlimit = {0., 0.};

  int nBin = pTlimit.size()-1;

  std::string fname = makeFilename();
  std::ofstream out(fname);
  
  // Set larger buffer for output file (e.g., 64 MB)
  const size_t bufferSize = 64 * 1024 * 1024;
  std::vector<char> fileBuffer(bufferSize);
  out.rdbuf()->pubsetbuf(fileBuffer.data(), bufferSize);
  
  out << "# NEVENTS: " << NEVENTS << "\n";
  out << "# MODE: "  << 3 << "\n";
  out << "# POWER: " << POWER << "\n";
  out << "# PREF: "  << PREF << "\n";
  out << "event,bin,pT,weight\n";

  // Use stringstream for buffered writing with scientific notation
  std::ostringstream buffer;
  buffer << std::scientific << std::setprecision(6);
  
  const int FLUSH_INTERVAL = 10000;  // Flush every N particles
  int particleCount = 0;

  int globalEvent = 0;
  int nEvent = NEVENTS;

  for (int iBin = 0; iBin < nBin; ++iBin) {

    pythia.readString("HardQCD:all = on");
    pythia.readString("SoftQCD:nonDiffractive = off");

    // settings.parm("PhaseSpace:pTHatMin", pTlimit[iBin]);
    // settings.parm("PhaseSpace:pTHatMax", pTlimit[iBin+1]);

    pythia.readString("PhaseSpace:bias2Selection = on");
    settings.parm("PhaseSpace:bias2SelectionPow", POWER);
    settings.parm("PhaseSpace:bias2SelectionRef", PREF);
    
    if (!pythia.init()) return 1;

    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

      if (!pythia.next()) continue;

      // Cache event weight to avoid repeated function calls
      double eventWeight = info.weight();

      // Loop over all particles in the event record
      for (int i = 0; i < pythia.event.size(); ++i) {

        const Particle& p = pythia.event[i];
        
        // Only final-state particles (not intermediate partons/resonances)
        if (!p.isFinal()) continue;
        
        // Only charged particles (detectable in tracking systems)
        if (!p.isCharged()) continue;
        
        // Only hadrons (exclude leptons, photons)
        if (!p.isHadron())  continue;

        // Reduce clutter
        if (p.pT() < 5.0) continue;

        // Detector acceptance cut: reject particles outside tracking volume
        if (std::abs(p.eta()) > 2.4) continue;

        if (!primaryFilter(p, pythia.event)) continue;

        // Write to buffer instead of file
        buffer << globalEvent << "," << iBin << "," 
               << p.pT() << "," << eventWeight << "\n";
        
        ++particleCount;
        
        // Periodically flush buffer to file
        if (particleCount >= FLUSH_INTERVAL) {
          out << buffer.str();
          buffer.str("");  // Clear the buffer
          buffer.clear();  // Clear any error flags
          particleCount = 0;
        }
      }

      ++globalEvent;
    }

    // Flush any remaining data in buffer after each bin
    if (particleCount > 0) {
      out << buffer.str();
      buffer.str("");
      buffer.clear();
      particleCount = 0;
    }

    out << "# sigmaGEN: " << info.sigmaGen() << "\n";
    out << "# weightSum: "  << info.weightSum() << "\n";
  }

  // Final flush
  if (particleCount > 0) {
    out << buffer.str();
  }

  out.close();
  return 0;
  
}