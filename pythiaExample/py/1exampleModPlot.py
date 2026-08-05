import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Valid comparisons:
    # mode 1 and mode 4 at high pT
    # mode 3 and mode 5 at high pT
    # mode 4 and mode 5 at low pT
# ------------------------------------------------------------
# Configuration (MUST match C++)
# ------------------------------------------------------------
MODES = list(map(int, input("Compare modes: ").split()))
if len(MODES) > 2:
    RATIO = list(map(int, input("Which two in the ratio: ").split()))
else:
    RATIO = MODES
MODE_LABELS = {1: "Mode 1: Hardcoded bins", 
               3: "Mode 3: Biased pT⁴", 
               4: "Mode 4: Low-pT matching",
               5: "Mode 5: Low-pT + bias"}
MODE_COLORS = {1: 'blue', 3: 'red', 4: 'green', 5: 'purple'}

# Store results for all modes
results = {}

for MODE in MODES:
    NRANGE = 120
    PTRANGE = 600.0
    BIN_EDGES = np.linspace(0.0, PTRANGE, NRANGE + 1)

    BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
    BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]


    # -----------------------------
    # Loop over pTHat bins (CSV files)
    # -----------------------------
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
    pTnorm  = np.zeros(NRANGE)
    pTvar   = np.zeros(NRANGE)

    # Bin each event manually
    for pt, w, norm in zip(all_pTHat, all_weights, all_sigmaNorm):

        bin_idx = np.searchsorted(BIN_EDGES, pt, side="right") - 1

        if bin_idx < 0 or bin_idx >= NRANGE:
            continue

        pTnorm[bin_idx] += norm * w
        pTvar[bin_idx]  += (norm * w) ** 2

    pTnormerr = np.sqrt(pTvar)

    # Store results for this mode
    results[MODE] = {
        'bin_edges': BIN_EDGES,
        'bin_centres': BIN_CENTERS,
        'pTnorm': pTnorm,
        'pTerr': pTnormerr 
    }

BIN_EDGES_CPP = [0., 7., 20., 100., 150., 250., 400., 600.]

# -----------------------------
# Create figure with 2x2 grid + bottom ratio plot
# -----------------------------
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Create subplots
ax1 = fig.add_subplot(gs[0, 0])  # Top left: Overlay 20-100 GeV
ax2 = fig.add_subplot(gs[0, 1])  # Top right: Overlay 0-20 GeV
ax3 = fig.add_subplot(gs[1, 0])  # Middle left: Offset 20-100 GeV
ax4 = fig.add_subplot(gs[1, 1])  # Middle right: Offset 0-20 GeV
ax5 = fig.add_subplot(gs[2, :])  # Bottom: Ratio plot (spanning both columns)

# -----------------------------
# Plot 1: Overlayed 20-600 GeV (Top Left)
# -----------------------------
for MODE in MODES:
    x = results[MODE]['bin_edges']
    y = results[MODE]['pTnorm']
    e = results[MODE]['pTerr']
    ex = results[MODE]['bin_centres']
    y_plot = np.append(y, y[-1])

    # Step histogram
    ax1.step(
        x, y_plot,
        where="post",
        label=MODE_LABELS[MODE],
        color=MODE_COLORS[MODE],
        linewidth=1.5
    )

    # Error bars (Poisson weighted)
    ax1.errorbar(
        ex, y,
        yerr=e,
        fmt="none",          # no markers
        ecolor=MODE_COLORS[MODE],
        elinewidth=1.2,
        capsize=2,
        alpha=0.9
    )
for edge in BIN_EDGES_CPP:
    ax1.axvline(edge, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_xlabel(r"$\hat{p}_T$ [GeV]")
ax1.set_ylabel(r"$d\sigma/d\hat{p}_T$")
ax1.set_title("pTHat distribution (20-600 GeV)")
ax1.set_yscale("log")
ax1.set_xlim(20, 600)
ax1.grid(alpha=0.3)
ax1.legend()

# -----------------------------
# Plot 2: Overlayed 0-20 GeV (Top Right)
# -----------------------------
for MODE in MODES:
    x = results[MODE]['bin_edges']
    y = results[MODE]['pTnorm']
    e = results[MODE]['pTerr']
    ex = results[MODE]['bin_centres']
    y_plot = np.append(y, y[-1])

    # Step histogram
    ax2.step(
        x, y_plot,
        where="post",
        label=MODE_LABELS[MODE],
        color=MODE_COLORS[MODE],
        linewidth=1.5
    )

    # Error bars (Poisson weighted)
    ax2.errorbar(
        ex, y,
        yerr=e,
        fmt="none",          # no markers
        ecolor=MODE_COLORS[MODE],
        elinewidth=1.2,
        capsize=2,
        alpha=0.9
    )
for edge in BIN_EDGES_CPP:
    ax2.axvline(edge, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_xlabel(r"$\hat{p}_T$ [GeV]")
ax2.set_ylabel(r"$d\sigma/d\hat{p}_T$")
ax2.set_title("pTHat distribution (0-20 GeV)")
ax2.set_yscale("log")
ax2.set_xlim(0, 20)
ax2.grid(alpha=0.3)
ax2.legend()

# -----------------------------
# Plot 3: Offset 20-600 GeV (Middle Left)
# -----------------------------
for i, MODE in enumerate(MODES):
    x = results[MODE]['bin_edges']
    y = results[MODE]['pTnorm']
    e = results[MODE]['pTerr']
    ex = results[MODE]['bin_centres']
    y_plot = np.append(y, y[-1])

    # Multiply by factor of 10^i for visibility
    factor = 10**i
    y_offset = y * factor
    y_plot_offset = y_plot * factor
    e_offset = e * factor

    ax3.step(
        x, y_plot_offset, 
        where="post",
        label=f"{MODE_LABELS[MODE]} x {factor}",
        color=MODE_COLORS[MODE],
        linewidth=1.5
    )

    ax3.errorbar(
        ex, y_offset, 
        yerr=e_offset, 
        fmt="none",
        ecolor=MODE_COLORS[MODE],
        elinewidth=1.2,
        capsize=2,
        alpha=0.9
    )
for edge in BIN_EDGES_CPP:
    ax3.axvline(edge, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.set_xlabel(r"$\hat{p}_T$ [GeV]")
ax3.set_ylabel(r"$d\sigma/d\hat{p}_T$ x factor")
ax3.set_title("pTHat distribution, offset (20-600 GeV)")
ax3.set_yscale("log")
ax3.set_xlim(20, 600)
ax3.grid(alpha=0.3)
ax3.legend()

# -----------------------------
# Plot 4: Offset 0-20 GeV (Middle Right)
# -----------------------------
for i, MODE in enumerate(MODES):
    x = results[MODE]['bin_edges']
    y = results[MODE]['pTnorm']
    e = results[MODE]['pTerr']
    ex = results[MODE]['bin_centres']
    y_plot = np.append(y, y[-1])

    factor = 10**i
    y_offset = y * factor
    y_plot_offset = y_plot * factor
    e_offset = e * factor

    ax4.step(
        x, y_plot_offset, 
        where="post",
        label=f"{MODE_LABELS[MODE]} x {factor}",
        color=MODE_COLORS[MODE],
        linewidth=1.5
    )

    ax4.errorbar(
        ex, y_offset, 
        yerr=e_offset, 
        fmt="none",
        ecolor=MODE_COLORS[MODE],
        elinewidth=1.2,
        capsize=2,
        alpha=0.9
    )
for edge in BIN_EDGES_CPP:
    ax4.axvline(edge, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4.set_xlabel(r"$\hat{p}_T$ [GeV]")
ax4.set_ylabel(r"$d\sigma/d\hat{p}_T$ x factor")
ax4.set_title("pTHat distribution, offset (0-20 GeV)")
ax4.set_yscale("log")
ax4.set_xlim(0, 20)
ax4.grid(alpha=0.3)
ax4.legend()

# -----------------------------
# Plot 3: Ratio plot (Bottom, spanning both columns)
# -----------------------------
# Compute ratio: RATIO[1] / RATIO[0]
mode_numerator = RATIO[1]
mode_denominator = RATIO[0]

y_num = results[mode_numerator]['pTnorm']
e_num = results[mode_numerator]['pTerr']
y_den = results[mode_denominator]['pTnorm']
e_den = results[mode_denominator]['pTerr']
ex = results[mode_numerator]['bin_centres']

# Compute ratio and error propagation
# Only compute ratio where denominator is non-zero
ratio = np.zeros_like(y_num)
ratio_err = np.zeros_like(y_num)

mask = (y_den > 0) & (y_num > 0)
ratio[mask] = y_num[mask] / y_den[mask]

# Error propagation: sigma_ratio = ratio * sqrt((sigma_num/num)^2 + (sigma_den/den)^2)
ratio_err[mask] = ratio[mask] * np.sqrt(
    (e_num[mask]/y_num[mask])**2 + (e_den[mask]/y_den[mask])**2
)

# Plot ratio
ax5.errorbar(
    ex[mask], ratio[mask],
    yerr=ratio_err[mask],
    fmt='o',
    color='black',
    markersize=3,
    elinewidth=1.2,
    capsize=2,
    alpha=0.9,
    label=f"{MODE_LABELS[mode_numerator]} / {MODE_LABELS[mode_denominator]}"
)

# Add horizontal line at y=1
ax5.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7)
for edge in BIN_EDGES_CPP:
    ax5.axvline(edge, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax5.set_xlabel(r"$\hat{p}_T$ [GeV]")
ax5.set_ylabel(f"Ratio")
ax5.set_title(f"Ratio: {MODE_LABELS[mode_numerator]} / {MODE_LABELS[mode_denominator]}")
ax5.grid(alpha=0.3)
ax5.set_ylim(0,2)
ax5.set_xlim(0, 600)
ax5.legend()
plt.tight_layout()

if len(MODES) == 4:
    plt.savefig("plots/compare/plot_panel_all.png", dpi=300)
else:
    plt.savefig("plots/compare/plot_panel_" + str(MODES[0]) + "_" + str(MODES[1]) +".png", dpi=300)
plt.show()

print(f"Done! Plot saved with panels including ratio plot.")
