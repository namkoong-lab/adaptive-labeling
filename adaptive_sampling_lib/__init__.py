"""Shared library for the Oct-2026 revision experiments of
"Adaptive Measurement via Posterior Variance Minimization".

Modules
-------
paths       : where results, data and model weights live (environment variables, shared-storage defaults)
data        : synthetic d-dimensional polyadic GP problems, eICU / California-housing / OpenML / MIMIC-IV loaders
subset      : Gumbel soft top-k relaxation and hard (Plackett-Luce) subset sampling
targets     : MSE and log-loss targets + posterior-risk utilities
uq_gp       : weighted-Cholesky GP uncertainty module (closed form, differentiable in weights)
uq_ensemble : vectorized Ensemble+ module with a differentiable unrolled Adam inner loop
uq_tabpfn   : TabPFN v2.5 uncertainty module with weighted context (attention bias)
uq_kumo     : Kumo-Tabular uncertainty module with weighted context (attention bias)
planners    : pathwise / REINFORCE one-step planners and active-learning baselines
predictor   : predictor networks (the model whose performance is being estimated)
runner      : multi-horizon acquisition loop with JSON logging
"""
