#include "Pythia8/Pythia.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace Pythia8;

int main(int argc, char* argv[]) {

  // Different modes are illustrated for setting the pT ranges.
  // Event-generation modes:
  //
  // 1 & 2: Hard QCD only.
  //        Fixed pTHat bins defined in the main program.
  //
  // 3 :    Hard QCD only.
  //        Full pTHat range generated in a single run,
  //        with biasing of the hard-process selection by a pTHat^4 factor.
  //
  // 4 :    Soft QCD only (non-diffractive).
  //        Matching between low- and high-pT. (Non-diffractive)
  //
  // 5 :    Hybrid soft–hard matching.
  //        - Soft QCD (non-diffractive) for pTHat < 20 GeV.
  //        - Hard QCD for pTHat ≥ 20 GeV.
  //        Uses the same pTHat binning as mode 1,
  //        with pTHat^4 biasing applied only in the hard region.

  if (argc != 2) {
    std::cerr << "Usage: ./main322 <mode> \n";
    return 1;
  }

  int mode   = std::stoi(argv[1]);
  std::vector<double> nEvents = {450000, 450000, 50000};

  bool completeEvents = false;
  bool smallOutput    = true;

  Pythia pythia;
  Settings& settings = pythia.settings;
  const Info& info   = pythia.info;

  if (smallOutput) {
    pythia.readString("Init:showProcesses = off");
    pythia.readString("Init:showMultipartonInteractions = off");
    pythia.readString("Init:showChangedSettings = off");
    pythia.readString("Init:showChangedParticleData = off");
    pythia.readString("Next:numberCount = 1000000000");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");
  }

  // Mode 1 and 3: set up five pT bins - last one open-ended.
  std::vector<double> pTlimit = {0., 5., 20., 100., 150., 250., 400., 600., 0.};

  // Modes 4 & 5: set up pT bins for range [0, 100]. The lowest bin
  // is generated with soft processes, to regularize pT -> 0 blowup.
  // Warning: if pTlimitLow[1] is picked too low there will be a
  // visible discontinuity, since soft processes are generated with
  // dampening and "Sudakov" for pT -> 0, while hard processes are not.
  std::vector<double> pTlimitLow = {0., 5., 20., 40., 70., 100., 150., 250., 400., 600., 0.};
  std::vector<double> pTlimitTwo = {0., 20., 600., 0.};

  int nBin = pTlimit.size() - 1;
  if (mode == 3) nBin = 1;
  else if (mode == 4) nBin = pTlimitLow.size() - 1;
  else if (mode == 5) nBin = pTlimitTwo.size() - 1;

  std::string fname = "data/events_" + std::to_string((int)mode) + ".csv";
  std::ofstream out(fname);
  out << "event,bin,pTHat,weight\n";

  int globalEvent = 0;
  
  int transitionBin = 0;
  if (mode > 3) {
    for (int i = 0; i < pTlimitTwo.size(); ++i) {
      if (pTlimitTwo[i] >= 20.0) {
        transitionBin = i;
        break;
      }
    }
  }

  for (int iBin = 0; iBin < nBin; ++iBin) {
    int nEvent = nEvents[iBin];

    // Normally HardQCD, but in two cases nonDiffractive.
    // Need MPI on in nonDiffractive to get first interaction, but not else.
    if (mode > 3 && iBin < transitionBin) {
      pythia.readString("HardQCD:all = off");
      pythia.readString("SoftQCD:nonDiffractive = on");
      if (!completeEvents) {
      pythia.readString("PartonLevel:all = on");
        pythia.readString("PartonLevel:ISR = off");
        pythia.readString("PartonLevel:FSR = off");
        pythia.readString("HadronLevel:all = off");
      }
    } else {
      pythia.readString("HardQCD:all = on");
      pythia.readString("SoftQCD:nonDiffractive = off");
      if (!completeEvents) pythia.readString("PartonLevel:all = off");
    }

    // Mode 1: hardcoded here. Use settings.parm for non-string input.
    if (mode == 1) {
      settings.parm("PhaseSpace:pTHatMin", pTlimit[iBin]);
      settings.parm("PhaseSpace:pTHatMax", pTlimit[iBin + 1]);
    }

    // Mode 3: The whole range in one step, but pT-weighted.
    else if (mode == 3) {
      settings.parm("PhaseSpace:pTHatMin", pTlimit[0]);
      settings.parm("PhaseSpace:pTHatMax", 0.);
      pythia.readString("PhaseSpace:bias2Selection = on");
      settings.parm("PhaseSpace:bias2SelectionPow", 5.);
      pythia.readString("PhaseSpace:bias2SelectionRef = 100.");
    }

    // Mode 4: hardcoded here. Use settings.parm for non-string input.
    else if (mode == 4) {
      settings.parm("PhaseSpace:pTHatMin", pTlimitLow[iBin]);
      settings.parm("PhaseSpace:pTHatMax", pTlimitLow[iBin + 1]);
    }

    // Mode 5: hardcoded here. Use settings.parm for non-string input.
    // Hard processes in one step, but pT-weighted.
    else if (mode == 5) {
        settings.parm("PhaseSpace:pTHatMin", pTlimitTwo[iBin]);
        settings.parm("PhaseSpace:pTHatMax", pTlimitTwo[iBin + 1]);
      if (iBin == transitionBin) {
        pythia.readString("PhaseSpace:bias2Selection = on");
        settings.parm("PhaseSpace:bias2SelectionPow", 5);
        pythia.readString("PhaseSpace:bias2SelectionRef = 20.");
      }
    }

    // Initialize for LHC at 14 TeV.
    pythia.readString("Beams:eCM = 14000.");
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = 0");

    if (!pythia.init()) return 1;

    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

      if (!pythia.next()) continue;

      // Transverse momentum of the outgoing partons in the hardest 2→2 subprocess
      double pTHat = info.pTHat();

      // Soft events have no upper pT limit. They therefore overlap
      // with hard events, and the overlap must be removed by hand.
      // No overlap for elastic/diffraction, which is only part of soft.
      if (mode > 3 && info.isNonDiffractive()) {
      double pTMin, pTMax;
      if (mode == 4) {
        pTMin = pTlimitLow[iBin];
        pTMax = pTlimitLow[iBin + 1];
      } else if (mode == 5) {
        pTMin = pTlimitTwo[iBin];
        pTMax = pTlimitTwo[iBin + 1];
      }
      if (pTHat < pTMin || pTHat > pTMax)
        continue;
      }

      out << globalEvent++ << ","
          << iBin << ","
          << pTHat << ","
          << info.weight() << "\n";
    }

    out << "# sigmaGeN: " << info.sigmaGen() << "\n";
    out << "# weightSum:"  << info.weightSum() << "\n";
  }

  out.close();
  return 0;
}
