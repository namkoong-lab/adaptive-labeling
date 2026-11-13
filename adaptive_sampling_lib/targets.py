"""Targets Q(theta) evaluated on the unlabelled evaluation set, and the posterior-risk estimate
R_MC = sample variance of Q over posterior draws (eq. (41))."""
import torch


class MSETarget:
    """Population MSE of a fixed predictor.  Draws F [.., S, N] are posterior draws of the latent
    function theta(x) at the eval points; Q_s = mean_x (F_s(x) - psi(x))^2 + noise_var.  When the
    UQ module returns draws of noisy labels (TabPFN fantasies), pass noise_var=0."""

    def __init__(self, predictions: torch.Tensor, noise_var: float = 0.0):
        self.pred = predictions.detach()  # [N]
        self.noise_var = float(noise_var)

    def q(self, draws: torch.Tensor) -> torch.Tensor:  # [..., S, N] -> [..., S]
        return ((draws - self.pred.to(draws)) ** 2).mean(-1) + self.noise_var

    def true_value(self, eval_y: torch.Tensor) -> float:
        return float(((eval_y.to(self.pred) - self.pred) ** 2).mean())

    def mean_from_moments(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """E[Q] from marginal posterior(-predictive) moments of the label at each eval point:
        mean [..., N], var [..., N] -> [...]."""
        return ((mean - self.pred.to(mean)) ** 2 + var).mean(-1) + self.noise_var


class LogLossTarget:
    """Population log loss (cross-entropy) of a fixed classifier with log-probabilities
    log_psi [N, C].  Draws P [.., S, N, C] are posterior draws of class probabilities
    theta_c(x); Q_s = mean_x sum_c P_s(c|x) * (-log psi_c(x))."""

    def __init__(self, log_probs: torch.Tensor):
        self.log_psi = log_probs.detach()  # [N, C]

    def q(self, draws: torch.Tensor) -> torch.Tensor:
        return -(draws * self.log_psi.to(draws)).sum(-1).mean(-1)

    def true_value(self, eval_y: torch.Tensor) -> float:
        lp = self.log_psi.gather(1, eval_y.long().to(self.log_psi.device).unsqueeze(1)).squeeze(1)
        return float(-lp.mean())

    def mean_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """E[Q] from marginal predictive class probabilities [..., N, C] -> [...]."""
        return -(probs * self.log_psi.to(probs)).sum(-1).mean(-1)


def risk_and_estimate(Q: torch.Tensor):
    """Q: [..., S] -> (variance over S (unbiased), mean over S)."""
    return Q.var(dim=-1, unbiased=True), Q.mean(dim=-1)
