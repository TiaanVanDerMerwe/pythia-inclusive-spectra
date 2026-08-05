import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from typing import Tuple, Dict, List

# ============================================================
# Helpers
# ============================================================

def power_law(x, A, n):
    return A * x**(-n)

def read_pythia_csv(filename, debug=True):
    """
    Streaming reader for Pythia CSV with:
    - Global metadata header
    - Event table (all events with weights)
    - Hadron data (only pT > threshold)
    """
    
    global_info = {}
    metadata = {}
    event_weights = {}
    hadrons = {"event": [], "bin": [], "pT": [], "weight": []}
    
    section = None  # Track current section: None, 'events', or 'hadrons'
    
    with open(filename, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line:
                continue
            
            # Parse metadata and section headers
            if line.startswith("#"):
                content = line[1:].strip()

                # Check for section markers
                if "Event table" in content:
                    section = 'events'
                    continue
                elif "Hadron data" in content:
                    section = 'hadrons'
                    continue
                
                # Parse key-value metadata
                if ":" in content:
                    key, _, value = content.partition(":")
                    key = key.strip()
                    value = value.strip()
                    
                    # Global parameters
                    if key == "ETA":
                        global_info["eta"] = float(value)
                    elif key == "NEVENTS":
                        global_info["nEvent"] = int(value)
                    elif key == "MODE":
                        global_info["mode"] = int(value)
                    elif key == "POWER":
                        global_info["power"] = float(value)
                    elif key == "PREF":
                        global_info["pref"] = float(value)
                    elif key == "sigmaGEN":
                        metadata["sigmaGEN"] = float(value)
                    elif key == "weightSum":
                        metadata["weightSum"] = float(value)
                
                continue
            
            # Skip CSV column headers
            if line.lower().startswith("event"):
                continue
            
            # Parse data based on current section
            parts = [p.strip() for p in line.split(",")]
            
            if section == 'events' and len(parts) == 2:
                try:
                    event_id = int(parts[0])
                    weight = float(parts[1])
                    event_weights[event_id] = weight
                except ValueError:
                    if debug:
                        print(f"WARNING: Failed to parse event line {line_num}: {line}")
            
            elif section == 'hadrons' and len(parts) == 4:
                try:
                    hadrons["event"].append(int(parts[0]))
                    hadrons["bin"].append(int(parts[1]))
                    hadrons["pT"].append(float(parts[2]))
                    hadrons["weight"].append(float(parts[3]))
                except ValueError:
                    if debug:
                        print(f"WARNING: Failed to parse hadron line {line_num}: {line}")
    
    # Convert to DataFrame
    hadrons_df = pd.DataFrame(hadrons)
    
    if debug:
        print(f"\n=== Parsing Summary ===")
        print(f"Total events: {len(event_weights)}")
        print(f"Hadrons (pT > threshold): {len(hadrons_df)}")
        print(f"Events with high-pT hadrons: {hadrons_df['event'].nunique()}")
        print(f"Global info: {global_info}")
        print(f"Metadata: {metadata}")

    return hadrons_df, metadata, global_info, event_weights

def compute_spectrum_with_poisson(hadrons_df, metadata, global_info, pT_bins):
    """
    Compute differential spectrum with Poisson uncertainties.
    
    Returns:
        central: Normalized spectrum values
        error: Poisson uncertainties
    """
    # Validate inputs
    if len(hadrons_df) == 0:
        print("ERROR: No hadrons to process!")
        n_bins = len(pT_bins) - 1
        return np.zeros(n_bins), np.zeros(n_bins)
    
    # Setup binning
    n_bins = len(pT_bins) - 1
    pT_centers = 0.5 * (pT_bins[1:] + pT_bins[:-1])
    dpT = np.diff(pT_bins)
    deta = 2 * global_info["eta"]
    
    # Extract normalization parameters - only one pTHat bin
    sigma_gen = metadata["sigmaGEN"]
    weight_sum = metadata["weightSum"]
    
    # Compute per-bin normalization factor
    # d²N/(2π pT dpT dη) normalized by σ_gen/Σw
    norm = sigma_gen / (weight_sum * 2 * np.pi * pT_centers * dpT * deta)
    
    # Extract hadron data
    pT_arr = hadrons_df["pT"].to_numpy()
    w_arr = hadrons_df["weight"].to_numpy()
    
    # Bin hadrons by pT
    bin_indices = np.digitize(pT_arr, pT_bins) - 1
    valid_mask = (bin_indices >= 0) & (bin_indices < n_bins)
    
    # Compute weighted histogram and variance
    hist_w = np.bincount(
        bin_indices[valid_mask], 
        weights=w_arr[valid_mask], 
        minlength=n_bins
    )

    hist_w2 = np.bincount(
        bin_indices[valid_mask], 
        weights=w_arr[valid_mask]**2, 
        minlength=n_bins
    )
    
    # Apply differential normalization
    spectrum = hist_w * norm
    spectrum_var = hist_w2 * norm**2
    
    # Normalize to inelastic cross-section
    central = spectrum / SIGMA_INEL[COM]
    error = np.sqrt(spectrum_var) / SIGMA_INEL[COM]
    
    return central, error

def jackknife_uncertainty(hadrons_df, event_weights, metadata, global_info, pT_bins, n_blocks, shuffle=True):
    """
    Compute jackknife uncertainties by systematically removing blocks of events.
    Uses analytical subtraction method for efficiency (single pTHat bin).
    
    Parameters:
        hadrons_df: Hadron dataframe
        event_weights: Dict mapping event_id -> weight
        metadata: Metadata dict
        global_info: Global info dict
        pT_bins: pT bin edges
        n_blocks: Number of jackknife blocks
        shuffle: Whether to randomize event-to-block assignment
    
    Returns:
        central: Central value (full sample)
        error: Jackknife uncertainty estimate
    """
    print(f"\n=== Computing Jackknife with {n_blocks} blocks ===")
    
    # Get all unique events
    all_events = np.array(sorted(event_weights.keys()))
    n_events = len(all_events)
    
    if n_blocks > n_events:
        print(f"WARNING: n_blocks ({n_blocks}) > n_events ({n_events}), setting n_blocks = n_events")
        n_blocks = n_events
    
    # Calculate block size
    block_size = (n_events + n_blocks - 1) // n_blocks
    
    print(f"Total events: {n_events}")
    print(f"Block size: ~{block_size} events per block")
    
    if n_blocks < 10:
        print("WARNING: Too few jackknife blocks — errors may be unreliable")
    
    # Assign events to blocks
    if shuffle:
        np.random.seed(42)  # For reproducibility
        shuffled_events = np.random.permutation(all_events)
    else:
        shuffled_events = all_events
    

    # Create mapping: event_id -> block_id
    event_to_block = {}
    for i, event_id in enumerate(shuffled_events):
        block_id = i // block_size
        event_to_block[event_id] = block_id
    
    # ============================================================
    # Setup binning
    # ============================================================
    n_bins = len(pT_bins) - 1
    pT_centers = 0.5 * (pT_bins[1:] + pT_bins[:-1])
    dpT = np.diff(pT_bins)
    deta = 2 * global_info["eta"]
    
    # Extract normalization parameters
    sigma_gen = metadata["sigmaGEN"]
    weight_sum = metadata["weightSum"]
    
    # ============================================================
    # Precompute block histograms and weight sums
    # ============================================================
    
    # Shape: [n_blocks, n_pt_bins]
    block_histograms = np.zeros((n_blocks, n_bins))
    block_weight_sums = np.zeros(n_blocks)
    
    # Map hadron events to blocks
    hadron_blocks = hadrons_df["event"].map(event_to_block).to_numpy()
    pT_arr = hadrons_df["pT"].to_numpy()
    w_arr = hadrons_df["weight"].to_numpy()

    # Find pT bin indices
    bin_indices = np.searchsorted(pT_bins, pT_arr, side="right") - 1
    valid = (bin_indices >= 0) & (bin_indices < n_bins)
    
    # Filter to valid entries
    valid_blocks = hadron_blocks[valid]
    valid_bins = bin_indices[valid]
    valid_weights = w_arr[valid]

    # Accumulate into block histograms
    print("Building block histograms...")
    for i in range(len(valid_blocks)):
        block_id = valid_blocks[i]
        pt_bin = valid_bins[i]
        weight = valid_weights[i]
        block_histograms[block_id, pt_bin] += weight

    # Accumulate event weight sums per block using event_weights dict
    print("Computing block weight sums...")
    for event_id, weight in event_weights.items():
        block_id = event_to_block.get(event_id)
        if block_id is not None:
            block_weight_sums[block_id] += weight
    
    # ============================================================
    # Compute full spectrum (sum over all blocks)
    # ============================================================
    
    # Sum over all blocks to get total histogram
    total_histogram = np.sum(block_histograms, axis=0)
    
    # Apply normalization
    norm = sigma_gen / weight_sum
    normalization = norm / (2 * np.pi * pT_centers * dpT * deta)
    full_spectrum = total_histogram * normalization
    
    # Normalize by inelastic cross section
    central = full_spectrum / SIGMA_INEL[COM]
    
    print(f"Full spectrum computed:")
    print(f"  Non-zero bins: {np.sum(central > 0)}/{n_bins}")
    
    # ============================================================
    # Compute jackknife samples via analytical subtraction
    # ============================================================
    
    jackknife_samples = np.zeros((n_blocks, n_bins))
    
    print("Computing jackknife samples...")
    # Compute each jackknife sample by subtracting one block from total
    for block_id in range(n_blocks):
        # Subtract this block's contribution from total histogram
        jk_histogram = total_histogram - block_histograms[block_id]
        
        # Correct jackknife weightSum
        jk_weight_sum = weight_sum - block_weight_sums[block_id]
        
        # Safety check (important for small samples)
        if jk_weight_sum <= 0:
            print(f"  WARNING: Block {block_id} has jk_weight_sum <= 0, skipping")
            continue
        
        # Apply normalization
        jk_norm = sigma_gen / jk_weight_sum
        jk_normalization = jk_norm / (2 * np.pi * pT_centers * dpT * deta)
        jk_spectrum = jk_histogram * jk_normalization
        
        # Normalize by inelastic cross section
        jackknife_samples[block_id] = jk_spectrum / SIGMA_INEL[COM]
        
        if (block_id + 1) % max(1, n_blocks // 10) == 0:
            print(f"  Processed {block_id+1}/{n_blocks} jackknife samples")
    
    # diagnose_jackknife_blocks(
    #     hadrons_df, event_weights, event_to_block, 
    #     pT_bins, n_blocks, block_histograms, block_weight_sums
    #   )
    # ============================================================
    # Calculate jackknife uncertainty
    # ============================================================
    
    # Jackknife estimator variance:
    # Var(θ) = (n-1)/n * Σ(θ_i - θ_mean)^2
    
    jk_mean = np.mean(jackknife_samples, axis=0)
    jk_variance = ((n_blocks - 1) / n_blocks) * np.sum((jackknife_samples - jk_mean)**2, axis=0)
    error = np.sqrt(jk_variance)
    
    print(f"\nJackknife uncertainties computed:")
    valid_bins = central > 0
    if np.any(valid_bins):
        print(f"  Average relative error: {np.mean(error[valid_bins] / central[valid_bins]):.3f}")
    print(f"  Bins with non-zero error: {np.sum(error > 0)}/{n_bins}")
    
    return central, error

def diagnose_jackknife_blocks(hadrons_df, event_weights, event_to_block, 
                               pT_bins, n_blocks, block_histograms, block_weight_sums):
    """
    Diagnose why jackknife might underestimate with few blocks.
    Compares within-block vs between-block variance.
    """
    print(f"\n=== JACKKNIFE DIAGNOSTICS ({n_blocks} blocks) ===")
    
    n_bins = len(pT_bins) - 1
    
    print("\nWithin vs Between-block variance:")
    
    # Pick a few representative pT bins
    test_bins = [4]
    
    for bin_idx in test_bins:
        if bin_idx >= n_bins:
            continue
            
        pT_center = 0.5 * (pT_bins[bin_idx] + pT_bins[bin_idx+1])
        
        # Get all hadrons in this pT bin
        mask = (hadrons_df["pT"] >= pT_bins[bin_idx]) & \
               (hadrons_df["pT"] < pT_bins[bin_idx+1])
        
        bin_hadrons = hadrons_df[mask]
        print(bin_hadrons.sort_values(by="weight", ascending=True))

        if len(bin_hadrons) == 0:
            continue
        
        # Map to blocks
        hadron_blocks = bin_hadrons["event"].map(event_to_block).to_numpy()
        hadron_weights = bin_hadrons["weight"].to_numpy()
        
        # Within-block variance (Poisson: sum of w²)
        within_var = 0
        for block_id in range(n_blocks):
            block_mask = hadron_blocks == block_id
            if block_mask.sum() > 0:
                block_weights = hadron_weights[block_mask]
                within_var += np.sum(block_weights**2)
        
        # Between-block variance (from jackknife perspective)
        block_totals = block_histograms[:, bin_idx]
        between_var = np.var(block_totals) * n_blocks  # Scale up
        
        print(f"   pT bin {pT_center:.1f} GeV:")
        print(f"      Within-block var:  {within_var:.2e}")
        print(f"      Between-block var: {between_var:.2e}")
        print(f"      Ratio (between/within): {between_var/within_var:.3f}")

# ============================================================
# MAIN ANALYSIS
# ============================================================

print(f"\n{'='*60}")
print(f"Starting analysis")
print(f"{'='*60}\n")

# ============================================================
# Configuration
# ============================================================

debug = True

COM = input("Enter COM energy: ").strip()
POW = input("Enter power: ").strip()

SIGMA_INEL = {
    "900": 50.3,
    "2760": 62.8,
    "5020": 67.6,
    "5360": 1.0,   # CMS plots dsigma/dpT directly
    "7000": 68.0,
    "13000": 71.3
}

# Jackknife configuration
NUMBER_OF_BLOCKS = [2000000, 1000000, 500000, 250000]  # Block size for jackknife
colors = ["blueviolet","red", "gold", "green"]
SHUFFLE_BLOCKS = True    # Randomize block assignment to avoid correlations

# ============================================================
# LOAD EXPERIMENTAL DATA (HEPData)
# ============================================================
cms_file = f"experimentData/CMS_{COM}_charged.csv"
print(f"\nLoading CMS file: {cms_file}")

df_hep = pd.read_csv(cms_file, comment="#")
pt_low = df_hep['PT LOW'].to_numpy()
pt_high = df_hep['PT HIGH'].to_numpy()

pT_bins = np.append(pt_low, pt_high[-1])
pT_hep = (pt_low + pt_high) / 2

n_bins = len(pT_bins) - 1

y_hep = df_hep["E*D3(N)/DP**3"].to_numpy()
stat_err = df_hep["stat +"].values
sys_err = df_hep["sys +"].values

yerr_hep = np.sqrt(stat_err**2 + sys_err**2)

# ============================================================
# LOAD PYTHIA DATA
# ============================================================
pythia_file = f"pythiaData/{COM}/charged/650765638/particles_pow{POW}.csv"
print(f"\nLoading Pythia file: {pythia_file}")
hadrons_df, metadata, global_info, event_weights = read_pythia_csv(pythia_file)

# ============================================================
# COMPUTE POISSON ERRORS
# ============================================================

poisson_central, poisson_error = compute_spectrum_with_poisson(
    hadrons_df, metadata, global_info, pT_bins
)

# ============================================================
# COMPUTE JACKKNIFE ERRORS (multiple block sizes)
# ============================================================

jk_results = []
jk_errors = []

for n_blocks in NUMBER_OF_BLOCKS:
    jk_central, jk_error = jackknife_uncertainty(
        hadrons_df, event_weights, metadata, global_info, pT_bins, 
        n_blocks=n_blocks, shuffle=SHUFFLE_BLOCKS
    )
    jk_results.append(jk_central)
    jk_errors.append(jk_error)

# ============================================================
# FIT HEPDATA SPECTRUM
# ============================================================

popt_hep, pcov_hep = curve_fit(
    power_law, pT_hep, y_hep,
    sigma=yerr_hep if yerr_hep.sum() > 0 else None,
    absolute_sigma=True,
    maxfev=10000
)

A_hep, n_hep = popt_hep
A_hep_err, n_hep_err = np.sqrt(np.diag(pcov_hep))

print(f"\n=== HEPData fit ===") 
print(f"A = {A_hep:.3e} ± {A_hep_err:.3e}")
print(f"n = {n_hep:.3f} ± {n_hep_err:.3f}")

# ============================================================
# FIT PYTHIA POISSON SPECTRUM
# ============================================================

mask = poisson_error > 0
if mask.sum() < 3:
    print(f"Warning: Too few valid bins for fitting ({mask.sum()} bins)")
    popt_pythia_poisson = None
else:
    popt_pythia_poisson, pcov_pythia_poisson = curve_fit(
        power_law,
        pT_hep[mask],
        poisson_central[mask],
        sigma=poisson_error[mask],
        absolute_sigma=True,
        maxfev=10000
    )
    
    A_pythia_poisson, n_pythia_poisson = popt_pythia_poisson
    A_pythia_poisson_err, n_pythia_poisson_err = np.sqrt(np.diag(pcov_pythia_poisson))
    
    print(f"\n=== Pythia Poisson fit ===") 
    print(f"A = {A_pythia_poisson:.3e} ± {A_pythia_poisson_err:.3e}")
    print(f"n = {n_pythia_poisson:.3f} ± {n_pythia_poisson_err:.3f}")

# ============================================================
# FIT PYTHIA JK SPECTRUM
# ============================================================

A_pythia_jk = []
A_pythia_jk_err = []
n_pythia_jk = []
n_pythia_jk_err = []

for i, n_blocks in enumerate(NUMBER_OF_BLOCKS):
    mask = jk_errors[i] > 0
    if mask.sum() < 3:
        print(f"Warning: Too few valid bins for fitting ({mask.sum()} bins)")
        popt_pythia_jk = None
    else:
        popt_pythia_jk, pcov_pythia_jk = curve_fit(
            power_law,
            pT_hep[mask],
            jk_results[i][mask],
            sigma=jk_errors[i][mask],
            absolute_sigma=True,
            maxfev=10000
        )
        
        A_pythia_jk.append(popt_pythia_jk[0])
        n_pythia_jk.append(popt_pythia_jk[1])
        A_pythia_jk_err.append(np.sqrt(np.diag(pcov_pythia_poisson))[0])
        n_pythia_jk_err.append(np.sqrt(np.diag(pcov_pythia_poisson))[1])
        
        print(f"\n=== Pythia Jackknife fit {n_blocks} blocks ===") 
        print(f"A = {A_pythia_jk[i]:.3e} ± {A_pythia_jk_err[i]:.3e}")
        print(f"n = {n_pythia_jk[i]:.3f} ± {n_pythia_jk_err[i]:.3f}")

# ============================================================
# PLOTTING
# ============================================================

print("\nGenerating comparison plot...")

fig = plt.figure(figsize=(7, 10), constrained_layout=True)
gs = GridSpec(
    3, 1,
    figure=fig,
    height_ratios=[3, 2, 2],
    hspace=0.15,
    top=0.95,
    bottom=0.06  # optional
)

# --- Panel 1: Spectra comparison ---
ax0 = fig.add_subplot(gs[0])

# Generate smooth fit curves
x_fit = np.logspace(np.log10(min(pT_hep)), np.log10(max(pT_hep)), 200)

# Plot experimental data
ax0.errorbar(
    pT_hep,
    y_hep,
    yerr=yerr_hep if yerr_hep.sum() > 0 else None,
    fmt="o",
    color="black",
    capsize=2,
    markersize=6
)

# Plot Poisson Pythia
ax0.errorbar(
    pT_hep,
    poisson_central * 0.1,
    yerr= poisson_error * 0.1,
    fmt="o",
    color="red",
    capsize=2,
    markersize=4
)

# Plot Jackknives Pythia
for i, n_blocks in enumerate(NUMBER_OF_BLOCKS):
    mult = (i + 1) * 0.01
    ax0.errorbar(
        pT_hep,
        jk_results[i] * mult,
        yerr = jk_errors[i] * mult,
        fmt="o",
        color=colors[i],
        capsize=2,
        markersize=4
    )

    #  Pythia fit line
    if popt_pythia_jk is not None:
        ax0.plot(x_fit, power_law(x_fit, A_pythia_jk[i], n_pythia_jk[i]) * mult, ls = "--", color = colors[i],
                label=f"Pythia Jackknife fit {n_blocks} blocks x {mult:.3f}: n = {n_pythia_jk[i]:.3f} ± {n_pythia_jk_err[i]:.3f}")

# HEPData fit line
ax0.plot(x_fit, power_law(x_fit, *popt_hep), 'k--',
         label=f"HEPData fit x 1: n={n_hep:.3f} ± {n_hep_err:.3f}")

# Poisson Pythia fit line
if popt_pythia_poisson is not None:
    ax0.plot(x_fit, power_law(x_fit, A_pythia_poisson, n_pythia_poisson) * 0.1, 'r--',
             label=f"Pythia Poisson fit x 1: n = {n_pythia_poisson:.3f} ± {n_pythia_poisson_err:.3f}")
        
# Format axes
ax0.set_yscale("log")
ax0.set_xlim(min(pT_hep)/1.1, max(pT_hep)*1.1)

# Safe y-limits calculation
valid_pythia = poisson_central[poisson_central > 0]
valid_hep = y_hep[y_hep > 0]

if len(valid_pythia) == 0:
    print("WARNING: No valid Pythia data points for plotting")
    ymin = valid_hep.min() * 0.5
    ymax = valid_hep.max() * 2
else:
    ymin = min(valid_hep.min(), valid_pythia.min()) * 0.5
    ymax = max(valid_hep.max(), valid_pythia.max()) * 2

ax0.set_ylim(0.01 * ymin, ymax)
ax0.set_ylabel(r"$\frac{1}{2\pi p_T}\,\frac{1}{N_{\mathrm{inel}}}\,\frac{d^2 N}{dp_T d\eta}$ [GeV$^{-2}$]", fontsize=12)
ax0.grid(True, which="both", ls="--", alpha=0.3)
ax0.legend(title=r"CMS pp, $\sqrt{s}=$" + COM + " GeV, charged", fontsize=9, loc='upper right')
ax0.tick_params(axis="x", which="both", labelbottom=False)

# --- Panel 2: Data/MC Jaccknife ratio ---
ax1 = fig.add_subplot(gs[1], sharex=ax0)

# Calculate ratio (Data/MC) - avoid division by zero
n_block = NUMBER_OF_BLOCKS[0]
jk_result = jk_results[0]
jk_error = jk_errors[0]

ratio = np.where(jk_result > 0, y_hep / jk_result, np.nan)

# Propagate only MC Jackknife uncertainties
ratio_err = np.where(jk_result > 0, 
                      y_hep * jk_error / jk_result**2, 
                      np.nan)

ax1.errorbar(
    pT_hep,
    ratio,
    yerr=ratio_err,
    color="red",
    alpha=0.6,
    label="Data/MC",
    capsize=2,
    drawstyle="steps-mid"
)

ax1.axhline(1, color="black", linestyle="--", linewidth=1.5, label="Perfect agreement")
ax1.set_ylim(0.5, 1/0.5)
ax1.set_ylabel(f"Data / MC (Jackknife {n_block} blocks)", fontsize=10)
ax1.set_xlabel(r"$p_T$ [GeV/c]", fontsize=12)
ax1.set_xscale("log")
ax1.grid(True, which="both", ls="--", alpha=0.3)
ax1.legend(fontsize=8, loc='upper right')
ax1.tick_params(axis="x", which="both", labelbottom=False)

# # --- Panel 3: Jaccknife/Poisson ratio ---
ax2 = fig.add_subplot(gs[2], sharex=ax0)

all_ratios = []

# Calculate ratio (Data/MC) - avoid division by zero
for i, n_blocks in enumerate(NUMBER_OF_BLOCKS):
    jk_error   = jk_errors[i]
    error_ratio = np.where(poisson_error > 0, jk_error / poisson_error, np.nan)
    all_ratios.append(error_ratio)

    ax2.plot(
        pT_hep,
        error_ratio,
        color=colors[i],
        marker="o",
        alpha=0.6,
        label=f"Jackknife/Poisson {n_blocks} block",
    )

# Convert to array and ignore NaNs
all_ratios = np.concatenate(all_ratios)
rmin = np.nanmin(all_ratios)
rmax = np.nanmax(all_ratios)

print(rmin)
print(rmax)


ax2.axhline(1, color="black", linestyle="--", linewidth=1.5, label="Perfect agreement")
ax2.set_ylim(0.5, 1.5)
ax2.set_ylabel(r"$\sigma_{\mathrm{Jackknife}} / \sigma_{\mathrm{Poisson}}$", fontsize=10)
ax2.set_xlabel(r"$p_T$ [GeV/c]", fontsize=12)
ax2.set_xscale("log")
ax2.grid(True, which="both", ls="--", alpha=0.3)
ax2.legend(fontsize=8, loc='upper right')

output_file = f"plots/jk/650765638/jackknife_comparison_{COM}GeV_pow" + POW + "_large.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as: {output_file}")
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)