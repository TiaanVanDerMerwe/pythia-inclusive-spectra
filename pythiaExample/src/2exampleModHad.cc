#include "Pythia8/Pythia.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>

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

int main() {
  // Hybrid soft–hard matching.
  // - Soft QCD (non-diffractive) for pTHat < 20 GeV.
  // - Hard QCD for pTHat ≥ 20 GeV.
  // with pTHat biasing applied only in the hard region.

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
  pythia.readString("Random:seed = 5000");

  std::vector<double> pTlimit = {0., 20., 600., 0.};
  int nBin = pTlimit.size() - 1;

  std::string fname = "data/events_HadronsPT_mode5.csv";
  std::ofstream out(fname);
  
  // Set larger buffer for output file (e.g., 64 MB)
  const size_t bufferSize = 64 * 1024 * 1024;
  std::vector<char> fileBuffer(bufferSize);
  out.rdbuf()->pubsetbuf(fileBuffer.data(), bufferSize);
  
  out << "event,bin,pT,weight\n";

  // Use stringstream for buffered writing
  std::ostringstream buffer;
  buffer.precision(6);  // Set precision for floating point output
  buffer << std::fixed;
  
  const int FLUSH_INTERVAL = 10000;  // Flush every N particles
  int particleCount = 0;

  int globalEvent = 0;

  int transitionBin = 0;
  for (int i = 0; i < pTlimit.size(); ++i) {
    if (pTlimit[i] >= 20.0) {
      transitionBin = i;
      break;
    }
  }

  std::cout << "Number of bins: " << nBin << std::endl;
  std::cout << "Number of soft bins: " << transitionBin << std::endl;
  
  std::vector<double> NEVENTS = {487500, 487500, 25000};

  for (int iBin = 0; iBin < nBin; ++iBin) {
    int nEvent = NEVENTS[iBin];
    if (iBin < transitionBin) {
      pythia.readString("HardQCD:all = off");
      pythia.readString("SoftQCD:nonDiffractive = on");
    } else {
      pythia.readString("HardQCD:all = on");
      pythia.readString("SoftQCD:nonDiffractive = off");
    }

    settings.parm("PhaseSpace:pTHatMin", pTlimit[iBin]);
    settings.parm("PhaseSpace:pTHatMax", pTlimit[iBin + 1]);
    if (iBin == transitionBin) {
      pythia.readString("PhaseSpace:bias2Selection = on");
      settings.parm("PhaseSpace:bias2SelectionPow", 5.);
      settings.parm("PhaseSpace:bias2SelectionRef", pTlimit[iBin]);
    }

    if (!pythia.init()) return 1;

    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

      if (!pythia.next()) continue;

      if (iBin < transitionBin && info.isNonDiffractive()) {
        double pTMin = pTlimit[iBin];
        double pTMax = pTlimit[iBin + 1];
        if (info.pTHat() < pTMin || info.pTHat() > pTMax)
          continue;
      }

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
    out << "# weightSum:"  << info.weightSum() << "\n";
  }

  // Final flush
  if (particleCount > 0) {
    out << buffer.str();
  }

  out.close();
  return 0;
}