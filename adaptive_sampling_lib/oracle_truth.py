"""Denoised ("noise-free") value of the target for synthetic GP data.

The realised target Q_real = mean_x (y(x) - psi(x))^2 over the evaluation labels differs from the
population quantity Q(theta) = mean_x E_{Y|theta,x}[(Y - psi(x))^2] by one draw of the label noise
(of size ~ 2 sigma sqrt(Q/N)), an offset common to every method.  For synthetic data the generating
GP is known exactly, so the Bayes-optimal denoised value is the posterior mean of Q under the *oracle*
posterior conditioned on ALL labels (initial + pool + evaluation):
    Q_orc = mean_x [ (m_orc(x) - psi(x))^2 + v_orc(x) ] + sigma^2,
with residual uncertainty std_orc(Q) (reported; typically ~5x smaller than the label-noise floor).
"""
from __future__ import annotations

import torch

from .targets import MSETarget
from .uq_gp import GPModule


def oracle_target(problem, psi: torch.Tensor, lengthscale: float, outputscale: float, noise_std: float, n_draws: int = 2000,
                  mean_constant: float = 0.0, seed: int = 0):
    """-> dict(q_real, q_oracle, q_oracle_std, q_plugin) for a synthetic GP problem and predictor outputs psi [N_eval]."""
    dev = problem.eval_x.device
    gp = GPModule(lengthscale, outputscale, noise_std ** 2, mean_constant, 1e-4, device=dev)
    target = MSETarget(psi, noise_var=noise_std ** 2)
    X = torch.cat([problem.train_x, problem.pool_x, problem.eval_x], 0)
    y = torch.cat([problem.train_y, problem.pool_y, problem.eval_y], 0)
    with torch.no_grad():
        st = gp.condition(X, y)
        q_orc = float(gp.target_mean(st, problem.eval_x, target))
        g = torch.Generator(device=dev).manual_seed(seed)
        draws = gp.posterior_draws(st, problem.eval_x, n_draws, g)
        q_std = float(target.q(draws).std(unbiased=True))
        m, _ = gp.posterior_mean_std(st, problem.eval_x) if hasattr(gp, "posterior_mean_std") else (draws.mean(0), None)
        q_plugin = float(((m.to(psi) - psi) ** 2).mean() + noise_std ** 2)
    return {"q_real": target.true_value(problem.eval_y), "q_oracle": q_orc, "q_oracle_std": q_std, "q_plugin": q_plugin}
