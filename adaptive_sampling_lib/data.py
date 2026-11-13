"""Problem construction: synthetic d-dimensional polyadic GP data (the paper's 51-cluster
selection-bias construction generalized to R^d), eICU regression/classification and
California housing with k-means-cluster selection bias.

A *problem* is a labelled initial training set (one cluster), a pool whose labels are
hidden from the planner, and an unlabelled evaluation set on which the target
(MSE / log-loss of a fixed predictor) is defined.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch


@dataclass
class Problem:
    train_x: torch.Tensor
    train_y: torch.Tensor
    pool_x: torch.Tensor
    pool_y: torch.Tensor
    eval_x: torch.Tensor
    eval_y: torch.Tensor
    train_cluster: torch.Tensor | None = None
    pool_cluster: torch.Tensor | None = None
    eval_cluster: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)
    eval_p: torch.Tensor | None = None  # synthetic classification: true class-1 probability at the evaluation inputs

    def to(self, device, dtype=None):
        def cv(t, is_x):
            if t is None:
                return None
            t = t.to(device)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            return t

        return Problem(
            cv(self.train_x, True), cv(self.train_y, False), cv(self.pool_x, True), cv(self.pool_y, False),
            cv(self.eval_x, True), cv(self.eval_y, False),
            None if self.train_cluster is None else self.train_cluster.to(device),
            None if self.pool_cluster is None else self.pool_cluster.to(device),
            None if self.eval_cluster is None else self.eval_cluster.to(device),
            dict(self.meta),
            None if self.eval_p is None else self.eval_p.to(device),
        )

    @property
    def input_dim(self):
        return self.train_x.shape[1]

    def summary(self):
        return {
            "n_train": int(self.train_x.shape[0]), "n_pool": int(self.pool_x.shape[0]),
            "n_eval": int(self.eval_x.shape[0]), "input_dim": int(self.input_dim), **self.meta,
        }


# --------------------------------------------------------------------------------------
# Kernels and GP prior sampling
# --------------------------------------------------------------------------------------
def rbf_kernel(x1, x2, lengthscale, outputscale):
    d2 = torch.cdist(x1, x2, p=2) ** 2
    return outputscale * torch.exp(-0.5 * d2 / lengthscale ** 2)


def sample_gp_prior(x, lengthscale, outputscale, noise_std, mean_constant=0.0, generator=None, jitter=1e-6):
    """Joint draw of noisy labels y = f(x) + eps from a zero/constant-mean RBF GP prior."""
    x64 = x.double()
    K = rbf_kernel(x64, x64, lengthscale, outputscale) + (noise_std ** 2 + jitter) * torch.eye(x.shape[0], dtype=torch.float64)
    L = torch.linalg.cholesky(K)
    z = torch.randn(x.shape[0], dtype=torch.float64, generator=generator)
    return (mean_constant + L @ z).float()


# --------------------------------------------------------------------------------------
# Synthetic polyadic (clustered) data in R^d
# --------------------------------------------------------------------------------------
def polyadic_cluster_layout(n_train, n_pool, n_anchor, eval_per_cluster=None):
    """The paper's cluster-assignment layout (if_logits=True branch of polyadic_sampler_new):
    train: all from cluster 0; pool: half from cluster 0, rest equally over clusters 1..K-1;
    eval: n_train/10 from cluster 0, n_train/10 from each of clusters 1..(K-1)/2, 1 from each of the rest.
    For n_train=100, n_pool=500, K=51 this gives |pool|=500, |eval|=285 as in the paper.
    eval_per_cluster overrides the n_train/10 count of the heavy evaluation clusters."""
    epc = n_train // 10 if eval_per_cluster is None else int(eval_per_cluster)
    train_idx = torch.zeros(n_train, dtype=torch.long)
    pool_parts = [torch.zeros(n_pool // 2, dtype=torch.long)]
    per = n_pool // (2 * (n_anchor - 1))
    for i in range(1, n_anchor):
        pool_parts.append(torch.full((per,), i, dtype=torch.long))
    pool_idx = torch.cat(pool_parts)
    eval_parts = [torch.zeros(epc, dtype=torch.long)]
    half = (n_anchor - 1) // 2
    for i in range(1, 1 + half):
        eval_parts.append(torch.full((epc,), i, dtype=torch.long))
    for i in range(1 + half, n_anchor):
        eval_parts.append(torch.full((1,), i, dtype=torch.long))
    eval_idx = torch.cat(eval_parts)
    return train_idx, pool_idx, eval_idx


def make_synthetic_gp_problem(
    seed: int,
    input_dim: int = 1,
    n_train: int = 100,
    n_pool: int = 500,
    n_anchor: int = 51,
    lengthscale: float = 1.0,
    outputscale: float = 0.69,
    noise_std: float = 0.1,
    mean_constant: float = 0.0,
    stdev_scale: float = 0.5,
    stdev_pool_scale: float = 0.5,
    scaling_factor: float | None = None,
    eval_per_cluster: int | None = None,
    layout: str = "paper",
    center_scale: float = 1.0,
    cluster_std: float = 0.35,
    lengthscale_per_sqrt_d: float = 0.5,
) -> Problem:
    """d-dimensional generalization of the paper's synthetic setting.

    Anchor k (k=0..K-1) is placed at norm sqrt(d)*k along a random direction; points around
    anchor k are anchor + stdev*N(0, I_d) (pool spread shrunk by stdev_pool_scale); everything
    is divided by `scaling_factor`, which defaults to sqrt(d).  With that default the cluster
    radius (~stdev) and the distance between anchors k and k+1 near the origin (~1) are
    independent of d, so the GP lengthscale=1 setting of the 1-D paper experiment keeps the
    same geometry in every dimension.  For d=1 this is exactly the paper's construction.
    """
    g = torch.Generator().manual_seed(seed)
    d = input_dim
    train_idx, pool_idx, eval_idx = polyadic_cluster_layout(n_train, n_pool, n_anchor, eval_per_cluster)
    n_eval = eval_idx.numel()
    if layout == "paper":
        if scaling_factor is None:
            scaling_factor = math.sqrt(d)
        anchor = torch.randn(n_anchor, d, generator=g)
        anchor = anchor / anchor.norm(dim=1, keepdim=True) * (math.sqrt(d) * torch.arange(n_anchor).unsqueeze(1))
        train_x = (anchor[train_idx] + torch.randn(n_train, d, generator=g) * stdev_scale) / scaling_factor
        eval_x = (anchor[eval_idx] + torch.randn(n_eval, d, generator=g) * stdev_scale) / scaling_factor
        pool_x = (anchor[pool_idx] + torch.randn(pool_idx.numel(), d, generator=g) * stdev_scale * stdev_pool_scale) / scaling_factor
    elif layout == "gmm":
        # Compact Gaussian-mixture layout with moderate covariate shift: cluster centres
        # ~ center_scale * N(0, I_d), isotropic cluster std `cluster_std`; the GP lengthscale is
        # lengthscale_per_sqrt_d * sqrt(d) so that the function is smooth within a cluster and
        # nearly independent across clusters in every dimension.  The population (pool, eval)
        # is roughly N(0, (center_scale^2 + cluster_std^2) I), so the single training cluster is a
        # few standard deviations away from the rest of the population rather than ~100 as in
        # the paper layout -- the regime foundation models such as TabPFN are meant for.
        anchor = center_scale * torch.randn(n_anchor, d, generator=g)
        train_x = anchor[train_idx] + torch.randn(n_train, d, generator=g) * cluster_std
        eval_x = anchor[eval_idx] + torch.randn(n_eval, d, generator=g) * cluster_std
        pool_x = anchor[pool_idx] + torch.randn(pool_idx.numel(), d, generator=g) * cluster_std * stdev_pool_scale * 2
        lengthscale = lengthscale_per_sqrt_d * math.sqrt(d)
        scaling_factor = 1.0
    else:
        raise ValueError(layout)
    x_all = torch.cat([train_x, eval_x, pool_x], 0)
    y_all = sample_gp_prior(x_all, lengthscale, outputscale, noise_std, mean_constant, generator=g)
    train_y = y_all[: n_train]
    eval_y = y_all[n_train: n_train + n_eval]
    pool_y = y_all[n_train + n_eval:]
    meta = dict(dataset=f"synthetic_gp_{layout}", seed_dataset=seed, input_dim=d, n_anchor=n_anchor, lengthscale=lengthscale,
                outputscale=outputscale, noise_std=noise_std, stdev_scale=stdev_scale, stdev_pool_scale=stdev_pool_scale,
                scaling_factor=scaling_factor, layout=layout, center_scale=center_scale, cluster_std=cluster_std)
    return Problem(train_x, train_y, pool_x, pool_y, eval_x, eval_y, train_idx, pool_idx, eval_idx, meta)


# --------------------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------------------
def make_synthetic_logit_problem(seed: int, input_dim: int = 10, n_train: int = 100, n_pool: int = 500, n_anchor: int = 51,
                                 outputscale: float = 4.0, layout: str = "gmm", center_scale: float = 0.5, cluster_std: float = 0.5,
                                 lengthscale_per_sqrt_d: float = 0.5, eval_per_cluster: int | None = None) -> Problem:
    """Binary classification with known ground truth: a latent GP draw f on the same clustered layout as
    make_synthetic_gp_problem, p(y=1|x) = sigmoid(f(x)), labels y ~ Bernoulli(p).  The true probabilities at the
    evaluation inputs are kept in Problem.eval_p, so the population log loss of any classifier is exact."""
    base = make_synthetic_gp_problem(seed, input_dim, n_train, n_pool, n_anchor, outputscale=outputscale, noise_std=0.0,
                                     layout=layout, center_scale=center_scale, cluster_std=cluster_std,
                                     lengthscale_per_sqrt_d=lengthscale_per_sqrt_d, eval_per_cluster=eval_per_cluster)
    g = torch.Generator().manual_seed(10_000 + seed)
    probs = [torch.sigmoid(y) for y in (base.train_y, base.pool_y, base.eval_y)]  # the (noise-free) latent draw -> probabilities
    ys = [(torch.rand(pr.shape, generator=g) < pr).long() for pr in probs]
    meta = dict(base.meta)
    meta.update(dataset=f"synthetic_logit_{layout}", task="classification")
    return Problem(base.train_x, ys[0], base.pool_x, ys[1], base.eval_x, ys[2], base.train_cluster, base.pool_cluster, base.eval_cluster, meta, eval_p=probs[2])


def make_polyadic_gp_problem(seed: int = 0, n_train: int = 100, n_pool: int = 500, n_eval: int = 285, n_anchor: int = 51,
                             input_dim: int = 1, stdev_scale: float = 0.5, stdev_pool_scale: float = 0.5,
                             scaling_factor: float = 1.0, lengthscale: float = 1.0, outputscale: float = 0.69,
                             noise_std: float = 0.1) -> Problem:
    """The synthetic GP problem of the paper's first version (its polyadic sampler, 'if_logits' layout), reproduced bitwise:
    the same random-number sequence (seed, a 10-point dummy draw, then the inputs) in float64 and the same gpytorch calls
    (gpytorch is needed only here).  Layout: anchors at distances 0, 1, ..., n_anchor-1 (times sqrt(d)) along random
    directions; the initial labels from anchor 0; half of the pool from anchor 0 and the other half spread evenly over the
    rest; the evaluation set n_train/10 points at each of anchors 0 .. (n_anchor-1)/2 and one point at each remaining anchor.

    The labels are NOT draws from the GP prior with the given hyperparameters.  The original code called the model in
    gpytorch's eval mode after setting its training data to these same inputs with zero targets, which returns the
    posterior given y = 0 at every input; for more than 800 points gpytorch also samples it through an approximate
    low-rank root.  The labels are therefore close to zero (std about 0.03-0.08 for seed 0, against about 0.65 for a prior
    draw).  make_synthetic_gp_problem draws from the prior.  Used by experiments/gp/gradient_quality.py --setup polyadic."""
    import gpytorch

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # the original run set this before generating its data
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        mean_module, kernel = gpytorch.means.ConstantMean(), gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module.constant = torch.tensor([0.0])
        kernel.base_kernel.lengthscale = lengthscale
        kernel.outputscale = outputscale
        likelihood.noise_covar.noise = noise_std ** 2

        class _GP(gpytorch.models.ExactGP):
            def __init__(self, x, y):
                super().__init__(x, y, likelihood)
                self.mean_module, self.covar_module = mean_module, kernel

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

        model = _GP(torch.randn(10, input_dim), torch.zeros(10))  # the original's dummy training data (consumes the RNG)
        # inputs (x_sampler, if_logits branch)
        anchor = torch.randn(n_anchor, input_dim)
        anchor = anchor / torch.sqrt((anchor ** 2).sum(1, keepdim=True)) * (math.sqrt(input_dim) * torch.arange(n_anchor)).unsqueeze(1)
        train_idx = torch.randint(0, 1, (n_train,))
        pool_idx = torch.randint(0, 1, (int(n_pool / 2),))
        for i in range(1, n_anchor):
            pool_idx = torch.cat([pool_idx, torch.randint(i, i + 1, (int(n_pool / (2 * (n_anchor - 1))),))])
        eval_idx = torch.randint(0, 1, (int(n_train / 10),))
        for i in range(1, 1 + int((n_anchor - 1) / 2)):
            eval_idx = torch.cat([eval_idx, torch.randint(i, i + 1, (int(n_train / 10),))])
        for i in range(1 + int((n_anchor - 1) / 2), n_anchor):
            eval_idx = torch.cat([eval_idx, torch.randint(i, i + 1, (1,))])
        assert eval_idx.numel() == n_eval and pool_idx.numel() == n_pool, "the original layout fixes the set sizes"
        train_x = (anchor[train_idx] + torch.randn(n_train, input_dim) * stdev_scale) / scaling_factor
        eval_x = (anchor[eval_idx] + torch.randn(n_eval, input_dim) * stdev_scale) / scaling_factor
        pool_x = (anchor[pool_idx] + torch.randn(n_pool, input_dim) * stdev_scale * stdev_pool_scale) / scaling_factor
        # labels: one joint draw from the GP prior with observation noise (y_sampler)
        x = torch.cat([train_x, eval_x, pool_x], 0)
        model.eval()
        likelihood.eval()
        model.set_train_data(inputs=x, targets=torch.zeros(x.shape[0]), strict=False)
        with torch.no_grad():
            y = likelihood(model(x)).sample()
    finally:
        torch.set_default_dtype(prev)
    train_y, eval_y, pool_y = y[:n_train], y[n_train: n_train + n_eval], y[n_train + n_eval:]
    meta = dict(dataset="synthetic_polyadic", seed=seed, lengthscale=lengthscale, outputscale=outputscale, noise_std=noise_std)
    return Problem(train_x, train_y, pool_x, pool_y, eval_x, eval_y, train_idx, pool_idx, eval_idx, meta)


from adaptive_sampling_lib.paths import DATA_CACHE, EICU_DIR  # noqa: E402

EICU_REG_DIR = os.path.join(EICU_DIR, "0518_data") + os.sep
EICU_RAW_TRAIN = os.path.join(EICU_DIR, "eicu_train_final.csv")
EICU_RAW_TEST = os.path.join(EICU_DIR, "eicu_test_final.csv")


def _read_xy(csv_dir, name):
    x = pd.read_csv(os.path.join(csv_dir, f"{name}_x.csv")).to_numpy()[:, 1:]
    y = pd.read_csv(os.path.join(csv_dir, f"{name}_y.csv")).to_numpy()[:, 1]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_eicu_regression_problem(csv_dir: str = EICU_REG_DIR) -> Problem:
    """The paper's eICU regression split (100 train from one k-means cluster, 388 pool, 500 eval of which the runs use the
    first 285; 10 standardized features; label = random-forest predicted in-hospital mortality probability)."""
    train_x, train_y = _read_xy(csv_dir, "train")
    pool_x, pool_y = _read_xy(csv_dir, "pool")
    eval_x, eval_y = _read_xy(csv_dir, "test")
    meta = dict(dataset="eicu_regression", csv_dir=csv_dir)
    return Problem(train_x, train_y, pool_x, pool_y, eval_x, eval_y, None, None, None, meta)


def make_clustered_problem(
    X: np.ndarray, y: np.ndarray, seed: int, n_clusters: int = 51, n_train: int = 100, n_pool: int = 500,
    n_eval: int = 285, train_cluster: str | int = "random", standardize_x: bool = True, standardize_y: bool = False,
    pool_eval_layout: str = "uniform", meta: dict | None = None, n_eval_clusters: int = 8, eval_scale: int = 1,
) -> Problem:
    """Selection-bias construction on a real dataset: k-means the covariates into `n_clusters`
    groups, take the initial labelled data from a single cluster, and draw the pool and the
    evaluation set from the whole population (all clusters)."""
    from sklearn.cluster import KMeans

    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if standardize_x:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
    if standardize_y:
        y = (y - y.mean()) / y.std()
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit(X)
    labels = km.labels_
    counts = np.bincount(labels, minlength=n_clusters)
    if train_cluster == "random":
        eligible = np.where(counts >= n_train)[0]
        c0 = int(rng.choice(eligible))
    elif train_cluster == "largest":
        c0 = int(np.argmax(counts))
    else:
        c0 = int(train_cluster)
    idx_c0 = np.where(labels == c0)[0]
    train_idx = rng.choice(idx_c0, n_train, replace=False)
    remaining = np.setdiff1d(np.arange(len(y)), train_idx)
    rng.shuffle(remaining)
    if pool_eval_layout == "uniform":
        pool_idx = remaining[:n_pool]
        eval_idx = remaining[n_pool: n_pool + n_eval]
    elif pool_eval_layout == "skewed":
        # The paper's synthetic construction transported to real data.
        # * Evaluation set = the target population, with heterogeneous density across covariate space:
        #   a random half of the non-training clusters is "heavy" (n_train/10 points each), the other half
        #   "light" (1 point each), plus n_train/10 points from the training cluster.
        # * Pool = what is available for labelling, distributed differently from the target population:
        #   half of it from the training cluster, the rest spread evenly over all other clusters.
        others = np.array([c for c in range(n_clusters) if c != c0])
        rng.shuffle(others)
        heavy = others[: len(others) // 2]
        epc = max(1, n_train // 10) * int(eval_scale)  # eval_scale > 1: same heavy/light structure with eval_scale x more points
        quota_eval = {int(c0): epc, **{int(c): epc for c in heavy}, **{int(c): int(eval_scale) for c in others[len(others) // 2:]}}
        per_other = max(1, (n_pool - n_pool // 2) // len(others))
        quota_pool = {int(c0): n_pool // 2, **{int(c): per_other for c in others}}
        pool_rows = {int(c): remaining[labels[remaining] == c] for c in range(n_clusters)}  # already shuffled

        def take(quota, total, exclude):
            chosen, spare = [], []
            for c, q in quota.items():
                rows = np.setdiff1d(pool_rows[c], exclude, assume_unique=True)
                rows = rows[rng.permutation(len(rows))]
                chosen.append(rows[:q])
                spare.append(rows[q:])
            chosen = np.concatenate(chosen)
            if len(chosen) < total:  # clusters that are too small: fill from the spare rows of the quota clusters
                spare = np.concatenate(spare)
                chosen = np.concatenate([chosen, spare[rng.permutation(len(spare))][: total - len(chosen)]])
            return chosen[:total]

        eval_idx = take(quota_eval, n_eval, np.array([], dtype=int))
        pool_idx = take(quota_pool, n_pool, eval_idx)
        eval_clusters = heavy
    else:
        raise ValueError(pool_eval_layout)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    li = lambda a: torch.tensor(labels[a], dtype=torch.long)
    m = dict(meta or {})
    m.update(dict(seed_dataset=seed, n_clusters=n_clusters, train_cluster=c0, pool_eval_layout=pool_eval_layout, eval_scale=int(eval_scale),
                  heavy_eval_clusters=(sorted(int(c) for c in eval_clusters) if pool_eval_layout == "skewed" else None)))
    return Problem(t(X[train_idx]), t(y[train_idx]), t(X[pool_idx]), t(y[pool_idx]), t(X[eval_idx]), t(y[eval_idx]),
                   li(train_idx), li(pool_idx), li(eval_idx), m)


def load_california_housing_arrays():
    from sklearn.datasets import fetch_california_housing

    os.makedirs(DATA_CACHE, exist_ok=True)
    ds = fetch_california_housing(data_home=DATA_CACHE)
    X = ds.data.astype(np.float64)
    y = np.log(ds.target.astype(np.float64))  # log median house value (in 100k$)
    return X, y


def make_california_problem(seed: int, **kw) -> Problem:
    X, y = load_california_housing_arrays()
    return make_clustered_problem(X, y, seed, standardize_y=True, meta=dict(dataset="california_housing"), **kw)


def load_eicu_raw_arrays(path: str = EICU_RAW_TRAIN, max_rows: int | None = None):
    df = pd.read_csv(path)
    df = df.dropna()
    if max_rows is not None:
        df = df.iloc[:max_rows]
    y = df["EVENT_LABEL"].to_numpy().astype(np.int64)
    X = df.drop(columns=["EVENT_LABEL"]).to_numpy().astype(np.float64)
    return X, y, [c for c in df.columns if c != "EVENT_LABEL"]


def make_eicu_classification_problem(seed: int, n_features: int | None = None, **kw) -> Problem:
    """eICU in-hospital mortality (binary) with cluster selection bias.  If n_features is given,
    keep the n most important features of a random-forest fit (as in the paper's regression
    construction, which kept 10)."""
    X, y, cols = load_eicu_raw_arrays()
    if n_features is not None and n_features < X.shape[1]:
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=8).fit(X, y)
        keep = np.argsort(-rf.feature_importances_)[:n_features]
        X = X[:, keep]
        cols = [cols[i] for i in keep]
    prob = make_clustered_problem(X, y.astype(np.float64), seed, meta=dict(dataset="eicu_classification", features=cols), **kw)
    prob.train_y = prob.train_y.long()
    prob.pool_y = prob.pool_y.long()
    prob.eval_y = prob.eval_y.long()
    return prob


def make_eicu_regression_clustered_problem(seed: int, **kw) -> Problem:
    """Fresh eICU regression draws (RF mortality probability as label) with a new random train
    cluster per seed, built from the raw eICU file; used when more than one dataset seed is wanted."""
    X, y, cols = load_eicu_raw_arrays()
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=8).fit(X, y)
    keep = np.argsort(-rf.feature_importances_)[:10]
    p = rf.predict_proba(X)[:, 1]
    return make_clustered_problem(X[:, keep], p, seed, meta=dict(dataset="eicu_regression_rf", features=[cols[i] for i in keep]), **kw)


# --------------------------------------------------------------------------------------
# OpenML datasets (numeric features only; categorical columns are ordinal-encoded)
# --------------------------------------------------------------------------------------
def load_openml_arrays(data_id: int, task: str = "regression", max_rows: int | None = 60000, seed: int = 0):
    """Returns (X float64 [n, d], y float64 or int64 [n], feature names).  Rows with missing
    values are dropped; categorical features are ordinal-encoded; a class target is label-encoded."""
    from sklearn.datasets import fetch_openml

    d = fetch_openml(data_id=data_id, data_home=DATA_CACHE, as_frame=True, parser="liac-arff")
    X, y = d.data.copy(), d.target.copy()
    keep = ~X.isna().any(axis=1) & ~y.isna()
    X, y = X[keep], y[keep]
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    X = X.to_numpy(dtype=np.float64)
    if task == "classification":
        y = y.astype("category").cat.codes.to_numpy().astype(np.int64)
    else:
        y = y.to_numpy(dtype=np.float64)
    if max_rows is not None and X.shape[0] > max_rows:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_rows, replace=False)
        X, y = X[idx], y[idx]
    return X, y, list(d.data.columns)


def make_openml_problem(data_id: int, task: str, seed: int, log_target: bool = False, **kw) -> Problem:
    X, y, cols = load_openml_arrays(data_id, task, seed=seed)
    meta = dict(dataset=f"openml_{data_id}", task=task, features=cols)
    if task == "classification":
        prob = make_clustered_problem(X, y.astype(np.float64), seed, meta=meta, **kw)
        prob.train_y, prob.pool_y, prob.eval_y = prob.train_y.long(), prob.pool_y.long(), prob.eval_y.long()
        return prob
    if log_target:
        y = np.log1p(np.clip(y, 0, None))
    return make_clustered_problem(X, y, seed, standardize_y=True, meta=meta, **kw)


# --------------------------------------------------------------------------------------
# MIMIC-IV (first-24h ICU cohort built by adaptive_sampling_lib.mimic_cohort)
# --------------------------------------------------------------------------------------
MIMIC_COHORT_CSV = os.path.join(DATA_CACHE, "mimic_icu_first24h.csv")


def load_mimic_arrays(task: str = "classification", csv: str = MIMIC_COHORT_CSV, min_feature_coverage: float = 0.7,
                      n_features: int | None = None):
    """Returns (X, y, feature names).  Features = first-24h vital aggregates + age/gender with at least
    `min_feature_coverage` non-missing rate; rows with any missing kept feature are dropped (complete
    cases).  task='classification' -> in-hospital mortality; 'icu_los' -> log ICU length of stay (days);
    'hosp_los' -> log hospital length of stay.  n_features: keep the n most important features of a
    random-forest fit for the task (as in the paper's eICU construction)."""
    df = pd.read_csv(csv, index_col=0)
    feat_cols = [c for c in df.columns if c.endswith(("_mean", "_min", "_max"))] + ["gcs_min_total", "age", "gender_male"]
    cov = df[feat_cols].notna().mean()
    feat_cols = [c for c in feat_cols if cov[c] >= min_feature_coverage]
    df = df.dropna(subset=feat_cols)
    X = df[feat_cols].to_numpy(dtype=np.float64)
    if task == "classification":
        y = df["hospital_expire_flag"].to_numpy().astype(np.int64)
    elif task == "icu_los":
        y = np.log(df["icu_los_days"].to_numpy(dtype=np.float64).clip(1e-3, None))
    elif task == "hosp_los":
        y = np.log(df["hosp_los_days"].to_numpy(dtype=np.float64).clip(1e-3, None))
    else:
        raise ValueError(task)
    if n_features is not None and n_features < X.shape[1]:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        rf = (RandomForestClassifier if task == "classification" else RandomForestRegressor)(n_estimators=100, random_state=0, n_jobs=8)
        rf.fit(X, y)
        keep = np.argsort(-rf.feature_importances_)[:n_features]
        X, feat_cols = X[:, keep], [feat_cols[i] for i in keep]
    return X, y, feat_cols


def make_mimic_problem(seed: int, task: str = "classification", n_features: int | None = 10, **kw) -> Problem:
    X, y, cols = load_mimic_arrays(task, n_features=n_features)
    meta = dict(dataset=f"mimic_{task}", features=cols, n_rows_total=int(X.shape[0]))
    if task == "classification":
        prob = make_clustered_problem(X, y.astype(np.float64), seed, meta=meta, **kw)
        prob.train_y, prob.pool_y, prob.eval_y = prob.train_y.long(), prob.pool_y.long(), prob.eval_y.long()
        return prob
    return make_clustered_problem(X, y, seed, standardize_y=True, meta=meta, **kw)
