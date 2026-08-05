import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuration (MUST match C++)
# ------------------------------------------------------------
NRANGE   = 60
PTRANGE  = 605.0
BIN_EDGES   = np.linspace(5.0, PTRANGE, NRANGE + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH   = BIN_EDGES[1] - BIN_EDGES[0]

fname = "data/events_HadronsPT_mode5.csv"

CHUNK_SIZE = 50000  # Process this many events at a time within each dataset

# Jackknife configuration
EVENTS_PER_BLOCK = 250  # Block size for jackknife
SHUFFLE_BLOCKS = True    # Randomize block assignment to avoid correlations

# ------------------------------------------------------------
# Fixed stream-based file processing with chunking WITHIN datasets
# ------------------------------------------------------------
def process_file_streaming(filename, chunk_size=CHUNK_SIZE):
    """
    Stream file and yield chunks, but ONLY within dataset boundaries.
    
    File format:
    # sigmaGEN: X
    event1_data
    event2_data
    ...
    # weightSum: Y    <- normalization calculated HERE, applies to events ABOVE
    # sigmaGEN: X2
    event1_data
    ...
    """
    with open(filename, "r") as f:
        chunk_buffer = []
        dataset_buffer = []  # Accumulate entire dataset before knowing normalization
        sigmaGen = None
        dataset_id = 0
        
        for line in f:
            line = line.strip()
            if not line or line.startswith("event"):
                continue
            
            if line.startswith("# sigmaGEN"):
                sigmaGen = float(line.split(":")[1])
                continue
            
            if line.startswith("# weightSum"):
                weightSum = float(line.split(":")[1])
                
                # NOW we can calculate normalization for the dataset we just read
                if sigmaGen is not None and weightSum is not None and dataset_buffer:
                    sigmaNorm = (sigmaGen / weightSum) * (NRANGE / PTRANGE)
                    
                    # Yield the accumulated dataset in chunks with dataset ID
                    for i in range(0, len(dataset_buffer), chunk_size):
                        chunk = dataset_buffer[i:i + chunk_size]
                        yield chunk, sigmaNorm, dataset_id
                    
                    # Reset for next dataset
                    dataset_buffer = []
                    sigmaGen = None
                    dataset_id += 1
                continue
            
            # Parse event line and accumulate in dataset buffer
            parts = line.split(",")
            dataset_buffer.append((float(parts[2]), float(parts[3])))  # pTHat, weight
        
        # Don't forget the last dataset!
        if dataset_buffer and sigmaGen is not None:
            # For the last dataset, we might not have seen weightSum yet
            # This handles incomplete files, but normally weightSum should be present
            print("Warning: Last dataset has no weightSum header")

# ------------------------------------------------------------
# Count total events per dataset (first pass)
# ------------------------------------------------------------
print("First pass: counting events per dataset...")
dataset_event_counts = {}

for chunk_events, sigmaNorm, dataset_id in process_file_streaming(fname, CHUNK_SIZE):
    if dataset_id not in dataset_event_counts:
        dataset_event_counts[dataset_id] = 0
    dataset_event_counts[dataset_id] += len(chunk_events)

print(f"Found {len(dataset_event_counts)} datasets")
for ds_id, count in dataset_event_counts.items():
    print(f"  Dataset {ds_id}: {count:,} events")

# ------------------------------------------------------------
# Create shuffled block assignments for each dataset
# ------------------------------------------------------------
print("\nCreating block assignments...")
dataset_block_assignments = {}

np.random.seed(42)  # For reproducibility

for ds_id, n_events in dataset_event_counts.items():
    n_blocks = max(2, n_events // EVENTS_PER_BLOCK)
    
    if SHUFFLE_BLOCKS:
        # Random permutation to avoid correlation with event ordering
        shuffled_indices = np.random.permutation(n_events)
        block_ids = shuffled_indices % n_blocks
    else:
        # Sequential assignment (can be biased if events are ordered)
        block_ids = np.arange(n_events, dtype=np.int32) % n_blocks
    
    dataset_block_assignments[ds_id] = {
        'block_ids': block_ids,
        'n_blocks': n_blocks,
        'block_sizes': np.bincount(block_ids, minlength=n_blocks)
    }
    
    print(f"  Dataset {ds_id}: {n_blocks} blocks (avg {n_events/n_blocks:.1f} events/block)")

# ------------------------------------------------------------
# Second pass: accumulate full spectrum and per-block contributions
# ------------------------------------------------------------
print("\nSecond pass: computing full spectrum and block contributions...")

# Global accumulators
pTnorm  = np.zeros(NRANGE, dtype=np.float64)
pTvar   = np.zeros(NRANGE, dtype=np.float64)

# Dictionary to track each dataset's contributions
dataset_contributions = {}

# Dictionary to track per-block histograms for jackknife
dataset_block_histograms = {}

total_events = 0
total_chunks = 0
pt_min = float('inf')
pt_max = float('-inf')

# Process chunks with progress indicator
for chunk_events, sigmaNorm, dataset_id in process_file_streaming(fname, CHUNK_SIZE):
    # Initialize dataset tracking if new
    if dataset_id not in dataset_contributions:
        dataset_contributions[dataset_id] = {
            'pTnorm': np.zeros(NRANGE, dtype=np.float64),
            'pTvar': np.zeros(NRANGE, dtype=np.float64),
            'events': 0,
            'sigmaNorm': sigmaNorm
        }
        
        # Initialize per-block histograms for this dataset
        n_blocks = dataset_block_assignments[dataset_id]['n_blocks']
        dataset_block_histograms[dataset_id] = np.zeros((n_blocks, NRANGE), dtype=np.float64)
    
    # Get block assignment info for this dataset
    block_info = dataset_block_assignments[dataset_id]
    
    # Convert chunk to numpy arrays (vectorized)
    chunk_array = np.array(chunk_events, dtype=np.float64)
    pTHat_vals = chunk_array[:, 0]
    weight_vals = chunk_array[:, 1]
    
    # Get event indices within this dataset (for block assignment)
    chunk_start_idx = dataset_contributions[dataset_id]['events']
    chunk_end_idx = chunk_start_idx + len(chunk_events)
    
    # Get block IDs for this chunk
    chunk_block_ids = block_info['block_ids'][chunk_start_idx:chunk_end_idx]
    
    # Update statistics
    chunk_size = len(pTHat_vals)
    total_events += chunk_size
    total_chunks += 1
    dataset_contributions[dataset_id]['events'] += chunk_size
    pt_min = min(pt_min, pTHat_vals.min())
    pt_max = max(pt_max, pTHat_vals.max())
    
    # Progress indicator for large files
    if total_chunks % 10 == 0:
        print(f"Processed {total_events:,} events ({total_chunks} chunks)...", end='\r')
    
    # Vectorized binning using searchsorted
    bin_indices = np.searchsorted(BIN_EDGES, pTHat_vals, side='right') - 1
    
    # Filter valid bins
    valid_mask = (bin_indices >= 0) & (bin_indices < NRANGE)
    bin_indices = bin_indices[valid_mask]
    pTHat_vals = pTHat_vals[valid_mask]
    weight_vals = weight_vals[valid_mask]
    chunk_block_ids = chunk_block_ids[valid_mask]
    
    # Pre-compute normalized weights for THIS ENTIRE DATASET
    weights_norm = sigmaNorm * weight_vals
    
    # Vectorized accumulation using np.bincount
    bin_values = np.bincount(bin_indices, weights=weights_norm, minlength=NRANGE)[:NRANGE]
    bin_var = np.bincount(bin_indices, weights=weights_norm**2, minlength=NRANGE)[:NRANGE]
    
    # Add to total
    pTnorm += bin_values
    pTvar += bin_var
    
    # Add to dataset-specific tracking
    dataset_contributions[dataset_id]['pTnorm'] += bin_values
    dataset_contributions[dataset_id]['pTvar'] += bin_var
    
    # Accumulate per-block histograms (vectorized)
    # This is the KEY for jackknife: track contribution from each block
    np.add.at(dataset_block_histograms[dataset_id], (chunk_block_ids, bin_indices), weights_norm)

pTnormerr = np.sqrt(pTvar)

# Calculate effective statistics: N_eff = (sum w)^2 / (sum w^2)
with np.errstate(divide='ignore', invalid='ignore'):
    N_eff = pTnorm**2 / pTvar
    N_eff[~np.isfinite(N_eff)] = 0.0

print(f"\nTotal events loaded: {total_events:,}")
print(f"Total chunks processed: {total_chunks}")
print(f"Number of datasets: {len(dataset_contributions)}")
print(f"pTHat range: [{pt_min:.2f}, {pt_max:.2f}]")
print(f"Memory-efficient processing: chunks within dataset boundaries")

# Print dataset summary
print("\nDataset Summary:")
for ds_id in sorted(dataset_contributions.keys()):
    ds = dataset_contributions[ds_id]
    n_blocks = dataset_block_assignments[ds_id]['n_blocks']
    print(f"  Dataset {ds_id}: {ds['events']:,} events, {n_blocks} blocks, sigmaNorm={ds['sigmaNorm']:.6e}")

# ------------------------------------------------------------
# JACKKNIFE UNCERTAINTY CALCULATION (streaming variance)
# ------------------------------------------------------------
print("\nCalculating Jackknife uncertainties...")

# Streaming accumulation for jackknife variance
sum_theta = np.zeros(NRANGE, dtype=np.float64)
sum_theta2 = np.zeros(NRANGE, dtype=np.float64)
total_blocks = 0

# Process each dataset's blocks
for ds_id in sorted(dataset_contributions.keys()):
    block_info = dataset_block_assignments[ds_id]
    block_hist = dataset_block_histograms[ds_id]
    ds_contrib = dataset_contributions[ds_id]
    
    n_blocks = block_info['n_blocks']
    n_events = ds_contrib['events']
    
    # For each block in this dataset
    for k in range(n_blocks):
        # Leave-one-out: full spectrum minus this block's contribution
        theta_k = pTnorm - block_hist[k]
        
        # Rescale to account for missing events (important!)
        # This corrects the estimate as if we had the full number of events
        block_size = block_info['block_sizes'][k]
        rescale = total_events / (total_events - block_size)
        theta_k *= rescale
        
        # Streaming variance accumulation
        sum_theta += theta_k
        sum_theta2 += theta_k * theta_k
        total_blocks += 1

# Compute jackknife variance
mean_theta = sum_theta / total_blocks
jackknife_var = ((total_blocks - 1) / total_blocks) * (sum_theta2 - total_blocks * mean_theta * mean_theta)
jackknife_err = np.sqrt(np.maximum(jackknife_var, 0))  # Ensure non-negative

print(f"Total jackknife blocks: {total_blocks}")

# ------------------------------------------------------------
# POISSON UNCERTAINTY CALCULATION
# ------------------------------------------------------------
print("Calculating Poisson uncertainties...")

# For Poisson, uncertainty is sqrt(N_eff)
with np.errstate(divide='ignore', invalid='ignore'):
    poisson_err = pTnorm / np.sqrt(N_eff)
    poisson_err[~np.isfinite(poisson_err)] = 0.0

# ------------------------------------------------------------
# COMPARISON STATISTICS
# ------------------------------------------------------------
print("\nUncertainty Comparison:")
print(f"{'Bin':>4} {'pT_center':>10} {'Value':>15} {'Poisson':>15} {'Jackknife':>15} {'Ratio (J/P)':>15}")
print("-" * 95)

for i in range(NRANGE):
    ratio = jackknife_err[i] / poisson_err[i] if poisson_err[i] > 0 else 0.0
    print(f"{i:4d} {BIN_CENTERS[i]:10.2f} {pTnorm[i]:15.6e} {poisson_err[i]:15.6e} {jackknife_err[i]:15.6e} {ratio:15.4f}")

# Calculate summary statistics
valid_bins = (poisson_err > 0) & (jackknife_err > 0)
ratio_valid = jackknife_err[valid_bins] / poisson_err[valid_bins]

print(f"\nRatio Statistics (Jackknife/Poisson):")
print(f"  Mean ratio: {np.mean(ratio_valid):.4f}")
print(f"  Median ratio: {np.median(ratio_valid):.4f}")
print(f"  Std dev: {np.std(ratio_valid):.4f}")
print(f"  Min ratio: {np.min(ratio_valid):.4f}")
print(f"  Max ratio: {np.max(ratio_valid):.4f}")

# Flag suspicious bins
print("\nDiagnostic: Bins with σ_JK/σ_Poisson ≈ 1.0 (possible single-event dominance):")
problem_bins = []
for i in range(NRANGE):
    if poisson_err[i] > 0:
        ratio = jackknife_err[i] / poisson_err[i]
        rel_unc = jackknife_err[i] / pTnorm[i] * 100 if pTnorm[i] > 0 else 0
        
        if abs(ratio - 1.0) < 0.05 and rel_unc > 5:  # Close to 1.0 and large uncertainty
            problem_bins.append((i, BIN_CENTERS[i], ratio, rel_unc))
            print(f"  Bin {i:2d} (pT={BIN_CENTERS[i]:6.2f}): ratio={ratio:.4f}, rel_unc={rel_unc:.2f}%")

if problem_bins:
    print(f"\nFound {len(problem_bins)} bins likely dominated by 1-3 events")
else:
    print("  None found")

# ------------------------------------------------------------
# Plotting (4 subplots)
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Plot 1: weighted spectrum with both error estimates
axes[0].step(BIN_CENTERS, pTnorm, where="mid", linewidth=1.5, label="Total", color='black')
axes[0].errorbar(BIN_CENTERS - 2, pTnorm, yerr=poisson_err,
                 fmt="none", elinewidth=1.2, capsize=2, alpha=0.7, 
                 label="Poisson errors", color='blue')
axes[0].errorbar(BIN_CENTERS + 2, pTnorm, yerr=jackknife_err,
                 fmt="none", elinewidth=1.2, capsize=2, alpha=0.7,
                 label="Jackknife errors", color='red')
axes[0].set_yscale("log")
axes[0].set_xlim(0, 600)
axes[0].set_xlabel(r"$p_T$ [GeV]")
axes[0].set_ylabel(r"$d\sigma/dp_T$")
axes[0].set_title(f"pT distribution (block size={EVENTS_PER_BLOCK}, shuffle={SHUFFLE_BLOCKS})")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: relative uncertainties comparison
with np.errstate(divide="ignore", invalid="ignore"):
    rel_unc_poisson = poisson_err / pTnorm
    rel_unc_jackknife = jackknife_err / pTnorm
    rel_unc_poisson[~np.isfinite(rel_unc_poisson)] = 0.0
    rel_unc_jackknife[~np.isfinite(rel_unc_jackknife)] = 0.0

axes[1].plot(BIN_CENTERS, rel_unc_poisson, 'o-', linewidth=1.2, label='Poisson', color='blue')
axes[1].plot(BIN_CENTERS, rel_unc_jackknife, 's-', linewidth=1.2, label='Jackknife', color='red')
axes[1].set_xlim(0, 600)
axes[1].set_xlabel(r"$p_T$ [GeV]")
axes[1].set_ylabel("Relative Uncertainty")
axes[1].set_title("Relative uncertainty: Poisson vs Jackknife")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: Ratio of Jackknife/Poisson uncertainties
axes[2].plot(BIN_CENTERS, jackknife_err / poisson_err, 'o-', linewidth=1.2, color='purple')
axes[2].axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Unity')
axes[2].set_xlim(0, 600)
axes[2].set_xlabel(r"$p_T$ [GeV]")
axes[2].set_ylabel("Jackknife / Poisson")
axes[2].set_title("Ratio of uncertainties (Jackknife/Poisson)")
axes[2].legend()
axes[2].grid(alpha=0.3)

# Plot 4: Absolute difference
axes[3].plot(BIN_CENTERS, jackknife_err - poisson_err, 'o-', linewidth=1.2, color='green')
axes[3].axhline(y=0.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[3].set_xlim(0, 600)
axes[3].set_xlabel(r"$p_T$ [GeV]")
axes[3].set_ylabel("Jackknife - Poisson")
axes[3].set_title("Absolute difference in uncertainties")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/plot_pT_jackknife_comparison.png", dpi=300)
plt.show()

print("\nDone! Plot saved as plots/plot_pT_jackknife_comparison.png")