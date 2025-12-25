#include <Pythia8/Pythia.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  long long nev = 50000;
  int seed = 12345;

  // Parse args: --nev N --seed S
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--nev" && i + 1 < argc) nev = std::stoll(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
  }

  Pythia8::Pythia pythia;

  // Proton-proton at 13 TeV
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eCM = 13000.");

  // Simple di-lepton production (Z/gamma* -> e+ e-)
  pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
  pythia.readString("23:onMode = off");
  pythia.readString("23:onIfAny = 11");

  // Deterministic RNG
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));

  // Reduce Print Overhead
  pythia.readString("Next:numberShowEvent = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowLHA = 0");

  pythia.init();

  // Prevent compiler from optimizing everything away
  volatile double checksum = 0.0;

  for (long long i = 0; i < nev; i++) {
    if (!pythia.next()) { i--; continue; }

    // Find final-state e- and e+
    int ieMinus = -1, iePlus = -1;
    for (int j = 0; j < pythia.event.size(); j++) {
      const auto& pj = pythia.event[j];
      if (!pj.isFinal()) continue;
      if (pj.id() == 11)  ieMinus = j;
      if (pj.id() == -11) iePlus  = j;
    }

    if (ieMinus >= 0 && iePlus >= 0) {
      auto p = pythia.event[ieMinus].p() + pythia.event[iePlus].p();
      checksum += p.pT() + p.eta() + p.phi() + p.mCalc();
    }

    if (i % 10000 == 0 && i > 0) {
      std::cerr << "generated " << i << "/" << nev << "\n";
    }
  }

  std::cerr << "checksum=" << checksum << "\n";
  return 0;
}
