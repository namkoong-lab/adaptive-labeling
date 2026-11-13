"""Shared library for the Oct-2026 revision experiments of
"Adaptive Measurement via Posterior Variance Minimization".

Modules
-------
data        : synthetic d-dimensional polyadic GP problems, eICU / California-housing loaders
subset      : Gumbel soft top-k relaxation and hard (Plackett-Luce) subset sampling
targets     : MSE and log-loss targets + posterior-risk utilities
uq_gp       : weighted-Cholesky GP uncertainty module (closed form, differentiable in weights)
uq_ensemble : vectorized Ensemble+ module with a differentiable unrolled Adam inner loop
uq_tabpfn   : TabPFN (v3 / v3.5) uncertainty module with weighted context (attention bias)
planners    : pathwise / REINFORCE one-step planners and active-learning baselines
predictor   : predictor networks (the model whose performance is being estimated)
runner      : multi-horizon acquisition loop with JSON logging
"""
