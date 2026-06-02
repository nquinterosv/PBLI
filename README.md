# PBLI

This repository contains MATLAB code and lightweight synthetic data for illustrating the Particle-Based Likelihood Inference (PBLI) framework for the SimpleDIVA model.

The code is intentionally simple. It is not meant to be a full MATLAB toolbox, but rather a clear example of how the PBLI workflow was implemented for the synthetic validation experiments.

## Files

- `pert.mat`  
  Perturbation sequence for the pitch-shift UP condition.

- `PBLI_synthetic_dataset.mat`  
  Lightweight synthetic dataset containing 1,000 simulated SimpleDIVA trajectories, the perturbation sequence, the target value, the ground-truth parameter vector, and the noise levels used in the simulations.

- `run_example_PBLI.m`  
  Example script. It loads the synthetic setup, generates one noisy realization, runs PBLI using both the dual-window and late-window likelihoods, and plots the fitted trajectories and posterior distributions.

- `run_all_montecarlo_PBLI.m`  
  Full Monte Carlo script used to regenerate the complete PBLI simulation results. The output is:

  ```matlab
  PBLI_montecarlo_results.mat
  ```

  This output file is very large, approximately 12 GB, because it stores the PBLI results for every Monte Carlo realization, noise level, and likelihood configuration. For this reason, it is not included in this GitHub repository. It is archived separately in Zenodo at [https://doi.org/10.5281/zenodo.20493995](doi.org/10.5281/zenodo.20493995)

## How to run

Keep all files in the same MATLAB folder.

To run the example:

```matlab
run_example_PBLI
```

To regenerate the full PBLI Monte Carlo results:

```matlab
run_all_montecarlo_PBLI
```

The full Monte Carlo script is computationally heavy. For a quick test, reduce these values inside the script:

```matlab
cfg.nMC = 5;
cfg.nParticles = 10000;
```

Then return them to the paper-scale values when needed.

## What is stored in `PBLI_synthetic_dataset.mat`

This file stores only the synthetic SimpleDIVA trajectories and the information needed to interpret them. It does not store PBLI inference results.

Main contents:

- `syntheticDataset.pert`: perturbation vector.
- `syntheticDataset.trials`: trial vector.
- `syntheticDataset.f0T`: normalized target value.
- `syntheticDataset.thetaTrue`: ground-truth parameter vector `[alpha_A, alpha_S, lambda_FF]`.
- `syntheticDataset.parameterNames`: names of the three SimpleDIVA parameters.
- `syntheticDataset.noiseSD_cents`: noise levels used in the simulations.
- `syntheticDataset.nMC`: number of Monte Carlo realizations.
- `syntheticDataset.rngSeed`: random seed used to generate the dataset.
- `syntheticDataset.early_clean`: noise-free early-window trajectory.
- `syntheticDataset.late_clean`: noise-free late-window trajectory.
- `syntheticDataset.feedforward_clean`: noise-free feedforward state trajectory.
- `syntheticDataset.early_noisy`: noisy early-window trajectories.
- `syntheticDataset.late_noisy`: noisy late-window trajectories.

The noisy trajectory arrays are stored as:

```matlab
nMC x nNoiseLevels x nTrials
```

with `nMC = 1000`.

The observation noise is Gaussian in cents and is applied multiplicatively to normalized linear-frequency trajectories:

```matlab
y_noisy = y_clean * 2^(epsilon_cents/1200)
```

## Full Monte Carlo results

The full PBLI output file,

```matlab
PBLI_montecarlo_results.mat
```

is not stored in GitHub because of its size. It is archived separately in Zenodo.

That file contains the complete PBLI outputs for all Monte Carlo realizations, including MAP estimates, credible intervals, posterior samples, and trajectory envelopes.

## Note

The simulations are synthetic twin experiments: the data are generated from the same SimpleDIVA model used for inference. This was done intentionally to study parameter recovery and identifiability in a controlled setting before moving to empirical-noise or experimental-data extensions.
