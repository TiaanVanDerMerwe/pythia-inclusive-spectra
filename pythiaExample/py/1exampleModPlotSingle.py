import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

# ------------------------------------------------------------
# Configuration (MUST match C++)
# ------------------------------------------------------------

MODE = int(input("Enter mode: ").strip())
POW = "4"
MODE_LABELS = {1: "Mode 1: Hardcoded bins", 
               3: rf"Mode 3: Biased $p_T^{{{int(POW)}}}$", 
               4: "Mode 4: Low-pT matching",
               5: "Mode 5: Low-pT + bias"}
MODE_COLORS = {1: 'blue', 3: 'red', 4: 'green', 5: 'purple'}


NRANGE = 120
PTRANGE = 600
BIN_EDGES = np.linspace(0.0, PTRANGE, NRANGE + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

# Store results for all modes
results = {}

fname = f"data/events_{MODE}.csv"

# Lists to accumulate all events across datasets
all_pTHat = []
all_weights = []
all_sigmaNorm = []  # Store normalization for each event

with open(fname, "r") as f:
    current_events = []
    sigmaGen = None
    weightSum = None
    
    for line in f:
        line = line.strip()
        if not line or line.startswith("event"):
            continue
            
        if line.startswith("# sigmaGeN"):
            sigmaGen = float(line.split(":")[1])
            continue
            
        if line.startswith("# weightSum"):
            weightSum = float(line.split(":")[1])
            
            # Process accumulated events for this dataset
            if current_events and sigmaGen is not None and weightSum is not None:
                events_array = np.array(current_events, dtype=float)
                pTHat_vals = events_array[:, 2]
                weight_vals = events_array[:, 3]
                
                # Calculate normalization for this dataset
                sigmaNorm = (sigmaGen / weightSum) * (NRANGE / PTRANGE)
                
                # Store events with their normalization
                all_pTHat.extend(pTHat_vals)
                all_weights.extend(weight_vals)
                all_sigmaNorm.extend([sigmaNorm] * len(pTHat_vals))
                
                # Reset for next dataset
                current_events = []
                sigmaGen = None
                weightSum = None
            continue
            
        # Regular event line
        current_events.append(line.split(","))

# Convert to numpy arrays
all_pTHat = np.array(all_pTHat)
all_weights = np.array(all_weights)
all_sigmaNorm = np.array(all_sigmaNorm)

print(f"Total events loaded: {len(all_pTHat)}")
print(f"pTHat range: [{all_pTHat.min():.2f}, {all_pTHat.max():.2f}]")

# Master histograms (sum over bins)
pTraw   = np.zeros(NRANGE)
pTnorm  = np.zeros(NRANGE)
pTvar   = np.zeros(NRANGE)
pTpow3  = np.zeros(NRANGE)

# Bin each event manually
for pt, w, norm in zip(all_pTHat, all_weights, all_sigmaNorm):

    bin_idx = np.searchsorted(BIN_EDGES, pt, side="right") - 1

    if bin_idx < 0 or bin_idx >= NRANGE:
        continue

    pTraw[bin_idx]  += 1
    pTnorm[bin_idx] += norm * w
    pTvar[bin_idx]  += (norm * w) ** 2
    pTpow3[bin_idx] += norm * w * pt**3

pTnormerr = np.sqrt(pTvar)

# Diagnostic output
# print("\nBins 45-50 (90-100 GeV):")
# print(f"  pTraw[45:50] = {pTraw[45:50]}")
# print(f"  pTnorm[45:50] = {pTnorm[45:50]}")
# print(f"  pTvar[45:50] = {pTvar[45:50]}")
# print(f"  pTnormerr[45:50] = {pTnormerr[45:50]}")

# ------------------------------
# Plotting (all modes together)
# ------------------------------

BIN_EDGES_CPP = [0., 20., 600.]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Plot 1: unweighted counts (all modes)
axes[0].step(BIN_CENTERS, pTraw, where="mid", 
            label=MODE_LABELS[MODE], color=MODE_COLORS[MODE], linewidth=1.5)
for edge in BIN_EDGES_CPP:
    axes[0].axvline(edge, color=MODE_COLORS[MODE], linestyle='--', alpha=0.5, linewidth=1)
axes[0].set_xlabel(r"$\hat{p}_T$ [GeV]")
axes[0].set_ylabel("Counts")
axes[0].set_title("pTHat distribution, unweighted")
axes[0].legend()
axes[0].set_yscale("log")
axes[0].set_xlim(0,600)
axes[0].grid(alpha=0.3)

# Plot 2: weighted spectrum (all modes)
axes[1].step(BIN_CENTERS, pTnorm, where="mid", 
             label=MODE_LABELS[MODE], color=MODE_COLORS[MODE], linewidth=1.5)
axes[1].errorbar(BIN_CENTERS, pTnorm, yerr=pTnormerr,
                 fmt="none", ecolor=MODE_COLORS[MODE],
                 elinewidth=1.2, capsize=2, alpha=0.9)
for edge in BIN_EDGES_CPP:
    axes[1].axvline(edge, color=MODE_COLORS[MODE], linestyle='--', alpha=0.5, linewidth=1)
axes[1].set_xlabel(r"$\hat{p}_T$ [GeV]")
axes[1].set_ylabel(r"$d\sigma/d\hat{p}_T$")
axes[1].set_title("pTHat distribution, weighted")
axes[1].legend()
axes[1].set_yscale("log")
axes[1].set_xlim(0,600)
axes[1].grid(alpha=0.3)

# Plot 3: pT^3 × weighted (all modes)
axes[2].step(BIN_CENTERS, pTpow3, where="mid", 
             label=MODE_LABELS[MODE], color=MODE_COLORS[MODE], linewidth=1.5)      
for edge in BIN_EDGES_CPP:
    axes[2].axvline(edge, color=MODE_COLORS[MODE], linestyle='--', alpha=0.5, linewidth=1)
axes[2].set_xlabel(r"$\hat{p}_T$ [GeV]")
axes[2].set_ylabel(r"$\hat{p}_T^3 \times d\sigma/d\hat{p}_T$")
axes[2].set_title("pTHat distribution, pT³ x weighted")
axes[2].legend()
axes[2].set_yscale("log")
axes[2].set_xlim(0,600)
axes[2].grid(alpha=0.3)

# Plot 4: relative uncertainty
with np.errstate(divide='ignore', invalid='ignore'):
    rel_unc = pTnormerr / pTnorm
    rel_unc[~np.isfinite(rel_unc)] = 0

axes[3].step(BIN_CENTERS, np.ones_like(BIN_CENTERS), where="mid", 
             label=MODE_LABELS[MODE], color=MODE_COLORS[MODE], linewidth=1.5)
axes[3].errorbar(BIN_CENTERS, np.ones_like(BIN_CENTERS), yerr=rel_unc,
                 fmt="none", ecolor=MODE_COLORS[MODE],
                 elinewidth=1.2, capsize=2, alpha=0.9)
for edge in BIN_EDGES_CPP:
    axes[3].axvline(edge, color=MODE_COLORS[MODE], linestyle='--', alpha=0.5, linewidth=1)
axes[3].set_xlabel(r"$\hat{p}_T$ [GeV]")
axes[3].set_ylabel(r"Relative Uncertainty")
axes[3].set_title("Relative uncertainty distribution")
axes[3].set_ylim(0.75,1.25)
axes[3].set_xlim(0,600)
axes[3].legend()
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/single/plot_" + str(MODE) + ".png", dpi=300)
plt.show()

print("Done! Plot saved as plots/single/plot_" + str(MODE) + ".png")