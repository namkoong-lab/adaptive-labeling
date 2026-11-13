"""The fixed predictor psi whose population performance is being estimated (an MLP trained on
the biased initial labelled set; port of gp_pipeline_regression/predictor_network.py)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), output_size=1, dropout=0.1):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_predictor(train_x, train_y, task="regression", hidden_dims=(64, 32), dropout=0.1, lr=1e-3, weight_decay=1e-4,
                    n_epochs=100, batch_size=32, val_split=0.2, patience=10, seed=42, device="cpu", n_classes=2):
    """Returns the trained network in eval mode.  task: 'regression' (MSE) or 'classification' (CE)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x = train_x.detach().float().cpu()
    y = train_y.detach().cpu()
    n = x.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_split)
    tr, va = perm[n_val:], perm[:n_val]
    out_size = 1 if task == "regression" else n_classes
    model = MLP(x.shape[1], hidden_dims, out_size, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if task == "regression":
        crit = lambda p, t: ((p.squeeze(-1) - t.float()) ** 2).mean()
    else:
        crit = lambda p, t: nn.functional.cross_entropy(p, t.long())
    xd, yd = x.to(device), y.to(device)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(n_epochs):
        model.train()
        order = tr[torch.randperm(tr.numel(), generator=g)]
        for i in range(0, order.numel(), batch_size):
            b = order[i: i + batch_size]
            opt.zero_grad()
            loss = crit(model(xd[b]), yd[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(xd[va]), yd[va]).item() if va.numel() > 0 else loss.item()
        if vl < best - 1e-12:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predictor_outputs(model, x, task="regression"):
    model.eval()
    out = model(x.float().to(next(model.parameters()).device))
    if task == "regression":
        return out.squeeze(-1)
    return torch.log_softmax(out, dim=-1).clamp(min=-30.0)  # log-probs [N, C]
