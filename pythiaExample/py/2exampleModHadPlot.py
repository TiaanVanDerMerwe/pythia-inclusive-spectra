import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys
from io import StringIO

# ------------------------------------------------------------
# Configuration (MUST match C++)
# ------------------------------------------------------------
NRANGE   = 60
PTRANGE  = 605.0
BIN_EDGES   = np.linspace(5.0, PTRANGE, NRANGE + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH   = BIN_EDGES[1] - BIN_EDGES[0]

fname = "data/events_Hadrons_pow4_pref300_N1e5.csv"

# Create timestamp for both plot and log file
now = datetime.now()
current_time = now.strftime("%H:%M:%S")

# Set up output capture
output_capture = StringIO()
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Redirect stdout to both console and StringIO
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_capture)

# ------------------------------------------------------------
# Read metadata and process file
# ------------------------------------------------------------
def process_file(filename):
    """Read metadata and process file in one pass."""
    metadata = {}
    pt_values = []
    weights = []
    sigmaGen = None
    weightSum = None
    
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("event"):
                continue
            
            if line.startswith("#"):
                if ":" not in line:
                    continue
                    
                line = line.lstrip("#").strip()
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if key in ['NEVENTS', 'MODE', 'POWER', 'PREF']:
                    metadata[key] = value
                elif key == "sigmaGEN":
                    sigmaGen = float(value)
                elif key == "weightSum":
                    weightSum = float(value)
                continue
            
            # Parse event line
            parts = line.split(",", 4)
            pt_values.append(float(parts[2]))
            weights.append(float(parts[3]))
    
    sigmaNorm = (sigmaGen / weightSum) * (NRANGE / PTRANGE)
    events = np.column_stack((pt_values, weights))
    
    return metadata, events, sigmaNorm

# Process file
print("Loading data...")
metadata, events, sigmaNorm = process_file(fname)

MODE = metadata.get('MODE', 'N/A')
NEVENTS = metadata.get('NEVENTS', 'N/A')
POWER = metadata.get('POWER', 'N/A')
PREF = metadata.get('PREF', 'N/A')

print(f"Metadata from file:")
print(f"  Mode: {MODE}")
print(f"  Number of events: {NEVENTS}")
print(f"  Biasing power: {POWER}")
print(f"  Biasing momentum: {PREF}")
print()

# ------------------------------------------------------------
# Process events
# ------------------------------------------------------------
pTnorm  = np.zeros(NRANGE, dtype=np.float64)
pTvar   = np.zeros(NRANGE, dtype=np.float64)
pT_rawcount = np.zeros(NRANGE, dtype=np.int64)

# Extract pTHat and weight columns
pTHat_vals = events[:, 0]
weight_vals = events[:, 1]

# Statistics
total_events = len(events)
pt_min = pTHat_vals.min()
pt_max = pTHat_vals.max()

# Vectorized binning using searchsorted
bin_indices = np.searchsorted(BIN_EDGES, pTHat_vals, side='right') - 1

# Filter valid bins
valid_mask = (bin_indices >= 0) & (bin_indices < NRANGE)
bin_indices = bin_indices[valid_mask]
weight_vals = weight_vals[valid_mask]

# Pre-compute normalized weights
weights_norm = sigmaNorm * weight_vals

# Vectorized accumulation using np.bincount
pTnorm = np.bincount(bin_indices, weights=weights_norm, minlength=NRANGE)[:NRANGE]
pTvar = np.bincount(bin_indices, weights=weights_norm**2, minlength=NRANGE)[:NRANGE]
pT_rawcount = np.bincount(bin_indices, minlength=NRANGE)[:NRANGE]

print(f"Processed {total_events:,} events, sigmaNorm={sigmaNorm:.6e}")

pTnormerr = np.sqrt(pTvar)

# Calculate effective statistics: N_eff = (sum w)^2 / (sum w^2)
with np.errstate(divide="ignore", invalid="ignore"):
    N_eff = (pTnorm**2) / pTvar
    N_eff[~np.isfinite(N_eff)] = 0.0

print(f"\nTotal events loaded: {total_events:,}")
print(f"pTHat range: [{pt_min:.2f}, {pt_max:.2f}]")

# Print effective statistics per bin
print("\n" + "="*80)
print("Effective Statistics per Bin:")
print("="*80)
print(f"{'Bin':<5} {'pT Center':<12} {'pT Range':<20} {'N_eff':<15} {'Value':<15} {'Error':<15}")
print("-"*80)
for i in range(NRANGE):
    pt_low = BIN_EDGES[i]
    pt_high = BIN_EDGES[i+1]
    pt_center = BIN_CENTERS[i]
    n_eff = N_eff[i]
    value = pTnorm[i]
    error = pTnormerr[i]
    
    print(f"{i:<5} {pt_center:>10.2f}   [{pt_low:>6.2f}, {pt_high:>6.2f})   "
          f"{n_eff:>12.2f}   {value:>12.6e}   {error:>12.6e}")

print("-"*80)
print(f"Total effective events across all bins: {N_eff.sum():.2f}")
print(f"Mean effective events per bin: {N_eff.mean():.2f}")
print(f"Median effective events per bin: {np.median(N_eff):.2f}")
print("="*80)

# Restore stdout and save output to file
sys.stdout = original_stdout
log_content = output_capture.getvalue()

# Save log file
log_filename = f"plots/plot_withCuts_pT_{current_time}.txt"
with open(log_filename, 'w') as f:
    f.write(log_content)

print(log_content)  # Print to console
print(f"\nLog saved as {log_filename}")

# ------------------------------------------------------------
# Plotting (2 subplots)
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: weighted spectrum
axes[0].step(BIN_CENTERS, pTnorm, where="mid", linewidth=1.5, label="Total")
axes[0].errorbar(BIN_CENTERS, pTnorm, yerr=pTnormerr,
                 fmt="none", elinewidth=1.2, capsize=2, alpha=0.9)
axes[0].set_yscale("log")
axes[0].set_xlim(0, 600)
axes[0].set_xlabel(r"$p_T$ [GeV]")
axes[0].set_ylabel(r"$d\sigma/dp_T$")
axes[0].set_title("pT distribution, weighted")
axes[0].grid(alpha=0.3)

# Plot 2: relative uncertainty
with np.errstate(divide="ignore", invalid="ignore"):
    rel_unc = pTnormerr / pTnorm
    rel_unc[~np.isfinite(rel_unc)] = 0.0

axes[1].step(BIN_CENTERS, np.ones_like(BIN_CENTERS), where="mid", linewidth=1.5,
             label="Total")
axes[1].errorbar(BIN_CENTERS, np.ones_like(BIN_CENTERS), yerr=rel_unc,
                 fmt="none", elinewidth=1.2, capsize=2, alpha=0.9)
axes[1].set_xlim(0, 600)
axes[1].set_xlabel(r"$p_T$ [GeV]")
axes[1].set_ylabel("Relative Uncertainty")
axes[1].set_title("Relative uncertainty distribution")
axes[1].grid(alpha=0.3)

# Add parameter information as figure title
fig.suptitle(f"Mode: {MODE} | N events: {NEVENTS} | Bias power: {POWER} | Bias $p_{{ref}}$: {PREF} GeV",
             fontsize=12, y=1.00)

plt.tight_layout()

plot_filename = f"plots/plot_withCuts_pT_{current_time}.png"
plt.savefig(plot_filename, dpi=300)
plt.show()

print(f"\nDone! Plot saved as {plot_filename}")