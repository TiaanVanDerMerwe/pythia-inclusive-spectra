# PYTHIA Single-Particle Spectrum

A Monte Carlo framework for generating and analysing inclusive
charged-hadron transverse momentum spectra in proton--proton ((pp))
collisions using **PYTHIA 8.317**.

This project was developed as the first validation step toward more
advanced jet and dihadron correlation studies. It generates
charged-particle spectra from simulated (pp) events, estimates
statistical uncertainties using both Poisson statistics and the
jackknife resampling method, and compares the resulting distributions
with experimental measurements.

## Features

-   Event generation with **PYTHIA 8.317**
-   Configurable centre-of-mass energy and event statistics
-   Inclusive charged-hadron (p_T) spectra
-   Statistical uncertainty estimation using
    -   Poisson uncertainties
    -   Jackknife resampling
-   Comparison with published experimental data
-   Publication-quality plotting scripts

## Repository Structure

``` text
.
├── generation/          # Event generation scripts
├── analysis/            # Spectrum calculation and uncertainty estimation
├── plotting/            # Plotting and comparison scripts
├── data/                # Experimental reference data
├── figures/             # Generated figures
└── README.md
```

*(The exact directory names may differ depending on the current
repository layout.)*

## Physics Motivation

Inclusive charged-particle spectra provide one of the most fundamental
tests of an event generator. Before studying more complex observables
such as dihadron correlations or medium-modified jets, it is important
to verify that the generator reproduces basic hadron production over a
wide kinematic range.

This repository serves as that validation step.

## Statistical Uncertainties

Two independent methods are implemented:

-   **Poisson uncertainties**, computed directly from the particle
    yields.
-   **Jackknife resampling**, obtained by repeatedly omitting subsets of
    events and recomputing the spectrum, providing an estimate of the
    statistical variance of the analysis procedure.

## Requirements

-   C++17
-   PYTHIA 8.317
-   Python 3
-   NumPy
-   Matplotlib
-   pandas

## Example Workflow

1.  Generate (pp) events with PYTHIA.
2.  Produce charged-particle spectra.
3.  Calculate statistical uncertainties.
4.  Compare the simulated spectra with experimental measurements.
5.  Produce publication-quality figures.

## Future Applications

This repository forms the foundation for subsequent studies of

-   dihadron correlations,
-   jet observables,
-   parton energy-loss models,
-   and medium-modified hadron production in heavy-ion collisions.

## Author

Tiaan van der Merwe

MSc Physics, University of Cape Town
