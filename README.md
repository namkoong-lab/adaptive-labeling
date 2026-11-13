# Adaptive Measurement via Posterior Variance Minimization — experiments

Code for the experiments of the revised paper *Adaptive Measurement via Posterior Variance Minimization*: a planner chooses
which points to label so that the posterior variance of a target quantity (here the error of a fixed predictor on a
shifted population) falls as fast as possible.  The planners are a pathwise (autodiff) policy gradient and REINFORCE with
and without a baseline; the baselines are random and uncertainty sampling.  The belief over the labels is a Gaussian
process, an `Ensemble+` network ensemble, or a tabular foundation model (TabPFN v2.5, Kumo-Tabular).

This branch (`revision-oct2026`) is the internal superset: every experiment of the revision, including those not in the
paper (Ensemble+, TabPFN).  Results, reports and the experiment log are kept outside git.

## Layout

```
adaptive_sampling_lib/       shared library
  planners.py                pathwise and REINFORCE planners, uncertainty baselines
  runner.py                  the multi-epoch acquisition loop; writes one JSON per run
  uq_gp.py uq_ensemble.py uq_tabpfn.py uq_kumo.py    beliefs, each with a differentiable weighted-context update
  data.py targets.py predictor.py subset.py          problems, targets, the evaluated predictor, soft top-k
  launch.py                  runs a job list on several GPUs
  aggregate.py plot_gradient_quality.py              curves, tables and figures from the result JSONs
  paths.py                   where results, data and model weights live (environment variables)
experiments/
  gp/                        GP regression: synthetic in R^d (d = 1 ... 100) and eICU
  ensemble_plus/             Ensemble+ regression, multi-step
  tabpfn/                    TabPFN v2.5 as the belief: regression/ and classification/
  kumo/                      Kumo-Tabular medium as the belief: regression/ and classification/
paper_figures/               the paper's experiment figures and tables from the result JSONs
```

Each experiment folder has a `run.py` (one run), a `make_jobs.py` (a job list; `--grid` gives the grids we ran) and its
checks and analysis scripts.

## Setup

Python 3.12 and a CUDA GPU.  Install the packages in `requirements.txt` (the versions the runs used).

- **Kumo-Tabular** needs NVIDIA's `structured-data-models` 0.2.1 (wheel from
  <https://github.com/NVIDIA/structured-data-models/releases>), installed into `$ADAPTIVE_SAMPLING_MODEL_CACHE/sdm_pkgs`
  (`pip install --target ... structured_data_models-0.2.1-py3-none-any.whl`) or into the environment.  Its weights are
  downloaded from Hugging Face on first use into `$ADAPTIVE_SAMPLING_MODEL_CACHE/hf_hub`.
- **TabPFN v2.5** checkpoints are downloaded by the `tabpfn` package into `$TABPFN_MODEL_CACHE_DIR` on first use.
- **Data.**  California housing and the OpenML datasets (Bike sharing, MAGIC, credit-card default) are downloaded
  automatically into `$ADAPTIVE_SAMPLING_DATA_CACHE`.  eICU and MIMIC-IV are credentialed (PhysioNet) and must be obtained
  separately; the MIMIC-IV cohort is built once with `python -m adaptive_sampling_lib.mimic_cohort`.

| Variable | What | Default |
|---|---|---|
| `ADAPTIVE_SAMPLING_RESULTS` | result JSONs, one folder per experiment | lab shared storage |
| `ADAPTIVE_SAMPLING_DATA_CACHE` | downloaded datasets, built MIMIC-IV cohort | lab shared storage |
| `ADAPTIVE_SAMPLING_EICU_DIR` | eICU files | lab shared storage |
| `ADAPTIVE_SAMPLING_MIMIC_DIR` | raw MIMIC-IV v3.1 tables | lab shared storage |
| `ADAPTIVE_SAMPLING_MODEL_CACHE` | Kumo-Tabular weights and package | lab shared storage |
| `TABPFN_MODEL_CACHE_DIR` | TabPFN checkpoints | lab shared storage |

## Running

```bash
python experiments/kumo/make_jobs.py --task regression --grid final --out jobs.txt
python -m adaptive_sampling_lib.launch jobs.txt --gpus 0 1 --per_gpu 3 --log_dir logs
```

Each job line ends with `## <output file>`; the launcher skips jobs whose output exists, so an interrupted grid can be
restarted.  Each run writes `<results>/<experiment>/<tag>/.../<method>_ds<split>_s<seed>.json` with the per-epoch posterior
variance of the target, its estimate, the true value and the selected batches.  A single run:

```bash
python experiments/gp/run.py --method pathwise --input_dim 10 --seed 1 --seed_dataset 1 --n_steps 1 --tau 0.5
```

All grids use batches of 5 labels; planners use 10 rollouts per gradient estimate, Adam with learning rate 0.1 and the
top-5 logits as the batch.  Baselines do not plan, so they are run once and reused for every compute budget.

## Experiments and their grids

### GP (`experiments/gp`)
Synthetic GP functions on a 51-cluster layout in R^d; 100 initial labels from one cluster, a pool of 500, an evaluation set
of 285, 10 epochs; the belief is a GP with the generating hyperparameters.  `--dataset eicu` runs the paper's eICU split
(label: the random-forest mortality probability) with the GP hyperparameters fitted in the paper.

| Grid | Command |
|---|---|
| Paper: d in {1, 5, 10, 20, 50, 100}, 20 function draws, budgets 1/5/10/25/100 steps, pathwise at tau = 0.5 | `make_jobs.py --grid paper` |
| eICU: ten training seeds, all methods at 1 and 1000 steps per epoch, tau = 0.5 | `make_jobs.py --grid eicu` |
| Rollouts per gradient estimate N in {1, 2, 10}: d = 1, 100 steps per epoch, 20 function draws, tau = 0.5 | `make_jobs.py --grid rollouts` |
| Pathwise at the default tau = 0.1 | `make_jobs.py --methods pathwise --regimes low mid5 mid10 mid25 high` |
| Pathwise at tau = 1.0 | `make_jobs.py --methods pathwise --regimes low mid5 mid10 mid25 high --tag_prefix draws_tau10 --extra "--tau 1.0"` |
| One function draw, training seeds 0-9 | `make_jobs.py --seeds 0-9 --regimes low high` |

`plot_results.py` prints tables and draws the variance and error curves of any of these grids (e.g.
`plot_results.py --dataset eicu --tags eicu_low_compute eicu_high_compute`); `analyze_draws.py` compares the posterior
variance with the realised error across function draws; `smoke_test.py` checks the weighted GP update against the subset
GP and that the planners and baselines run.

Gradient quality (`gradient_quality.py`, one function draw per run, `--seed_dataset k`): at the initial belief and
uniform logits, the expected cosine similarity between each estimator's N-rollout gradient (REINFORCE, REINFORCE with a
mean baseline, pathwise at several tau) and the true gradient of the planners' objective, for N = 1 ... 1000.  The true
gradient is REINFORCE over 10^6 rollouts with the closed-form posterior variance of the target.  `--setup polyadic`
reproduces the first version of this study (its synthetic problem, a constant-zero predictor and the true pool labels in
the rollouts; needs gpytorch); `--predictor` and `--labels` change one factor at a time.  Plot one or several draws
with `python -m adaptive_sampling_lib.plot_gradient_quality --json <files> --taus 0.1 0.5 1`.

### Ensemble+ (`experiments/ensemble_plus`)
Ten smooth (SiLU, heavy-ball) members anchored to their initialisation; synthetic GP functions in R^5, 20 initial labels,
5 epochs, 1 and 50 gradient steps per epoch.

| Grid | Command |
|---|---|
| v5: 21 clusters, pool 200 (reported) | `make_jobs.py --grid v5` |
| v4: 11 clusters, pool 100 | `make_jobs.py --grid v4` |

`smoke_test.py` checks the ensemble and the planners on it; `benchmark_pathwise_speed.py` times one pathwise gradient
estimate against the original per-member implementation.

### TabPFN v2.5 (`experiments/tabpfn`)
Regression: eICU, California housing, Bike sharing, MIMIC-IV ICU length of stay and a synthetic GP layout in R^10
(target: the MSE of a fixed predictor).  Classification: eICU mortality, MAGIC, credit-card default, MIMIC-IV mortality
and a synthetic task (target: the log-loss of a fixed classifier).  Ten splits each, skewed pool and evaluation set,
1/10/50 gradient steps per epoch, tau = 0.5, float32.

| Grid | Command |
|---|---|
| Regression, range-conditioned moments (reported) | `make_jobs.py --task regression --grid final` |
| Classification (reported) | `make_jobs.py --task classification --grid final` |
| Regression, full moments, squashed inputs | `make_jobs.py --task regression --tag_prefix final_full --extra "--input_transform squash3 --fp32 --tau 0.5"` |

Checks and studies in `regression/`: `test_tabpfn_module.py` (weighted update, its gradients, posterior draws),
`gradient_quality.py` (gradient estimators against a reference; plot with `python -m adaptive_sampling_lib.plot_gradient_quality`),
`noise_decomposition.py`, `analyze_splits.py`.

### Kumo-Tabular (`experiments/kumo`)
The TabPFN grids with Kumo-Tabular medium as the belief.  The fractional update keeps each context row's attention to
itself at full weight (`--kumo_self_attn full`) and, for regression, gives the pseudo-updates the labelled data's
preprocessing statistics (`--kumo_fixed_stats both`); both leave the ordinary update unchanged at binary weights.

| Grid | Command |
|---|---|
| Regression (reported) | `make_jobs.py --task regression --grid final` |
| Classification (reported) | `make_jobs.py --task classification --grid final` |
| Regression, original fractional update | `make_jobs.py --task regression --tag_prefix final_kumo_fix --extra "--moments range --tau 0.5 --kumo_fixed_stats both"` |
| Classification, original fractional update | `make_jobs.py --task classification --tag_prefix final_kumo_cls --extra "--tau 0.5"` |
| Average-row-embedding relaxation (pathwise only) | `make_jobs.py --task regression --methods pathwise --tag_prefix final_kumo_ae --extra "--moments range --tau 0.5 --kumo_fixed_stats both --kumo_relax avg_emb"` |

Checks in `regression/` and `classification/`: `check_weighted.py` (the weighted update against the package's forward
pass, binary weights, gradients) and `check_relaxations.py`; `stage1.py` compares relaxations and objectives at the first
epoch, and `agreement.py` compares variance criteria on candidate batches.

## Paper figures (`paper_figures`)

`make_figures.py` draws the GP-in-R^d and Kumo-Tabular figures, the tables and `summary.txt` (the numbers quoted in the
text); `make_gp_extra_figures.py` draws the GP eICU, rollouts-per-estimate and gradient-quality figures.  Both read the
results root and write to `paper_figures/output/` (or `--out <folder>`, e.g. the paper's figure folder).
