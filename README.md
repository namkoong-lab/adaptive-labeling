# Adaptive Measurement via Posterior Variance Minimization

Code for the experiments of the paper *Adaptive Measurement via Posterior Variance Minimization*.  A planner chooses which
points to label so that the posterior variance of a target quantity, here the error of a fixed predictor on a shifted
population, falls as fast as possible.  The planners optimise a one-step lookahead objective with a smoothed pathwise
(autodiff) policy gradient or with REINFORCE (with and without a baseline); the baselines are random and uncertainty
sampling.  The belief over the labels is a Gaussian process or a tabular foundation model (Kumo-Tabular, TabPFN v2.5).

The code of the paper's first version is in the history of this repository (commit `15cb875` and earlier).

## Layout

```
adaptive_sampling_lib/       shared library
  planners.py                pathwise and REINFORCE planners, uncertainty baselines
  runner.py                  the multi-epoch acquisition loop; writes one JSON per run
  uq_gp.py uq_tabpfn.py uq_kumo.py                   beliefs, each with a differentiable weighted-context update
  data.py targets.py predictor.py subset.py          problems, targets, the evaluated predictor, soft top-k
  launch.py                  runs a job list on several GPUs
  aggregate.py plot_runs.py plot_gradient_quality.py curves, tables and figures from the result JSONs
  paths.py                   where results, data and model weights live (environment variables)
experiments/
  gp/                        GP regression: synthetic in R^d (d = 1 ... 100) and eICU; the gradient-quality study
  kumo/                      Kumo-Tabular as the belief: regression/ and classification/
  tabpfn/                    TabPFN v2.5 as the belief: regression/ and classification/ (the Kumo runners build on these)
paper_figures/               the paper's experiment figures and tables from the result JSONs
```

Each experiment folder has a `run.py` (one run), a `make_jobs.py` (a job list; `--grid` gives the paper's grids) and its
checks and analysis scripts.

## Setup

Python 3.12 and a CUDA GPU.  `pip install -r requirements.txt` installs the versions used for the paper.

- **Kumo-Tabular** needs NVIDIA's `structured-data-models` 0.2.1 (wheel from
  <https://github.com/NVIDIA/structured-data-models/releases>), installed into the environment or into
  `$ADAPTIVE_SAMPLING_MODEL_CACHE/sdm_pkgs` (`pip install --target ... structured_data_models-0.2.1-py3-none-any.whl`).
  Its weights are downloaded from Hugging Face on first use into `$ADAPTIVE_SAMPLING_MODEL_CACHE/hf_hub`.
- **TabPFN v2.5** checkpoints are downloaded by the `tabpfn` package into `$TABPFN_MODEL_CACHE_DIR` on first use.
- **Public data.**  California housing and the OpenML datasets (Bike sharing, MAGIC, credit-card default) are downloaded
  automatically into `$ADAPTIVE_SAMPLING_DATA_CACHE`.
- **Credentialed data.**  eICU-CRD and MIMIC-IV are available to credentialed users from PhysioNet and are not distributed
  with this code.
  - eICU: `$ADAPTIVE_SAMPLING_EICU_DIR/eicu_train_final.csv` and `eicu_test_final.csv` (one row per ICU stay, first-day
    features and the in-hospital mortality label; used by the Kumo-Tabular and TabPFN eICU tasks), and the GP experiment's
    fixed split `$ADAPTIVE_SAMPLING_EICU_DIR/0518_data/{train,pool,test}_{x,y}.csv` (10 standardised features; label: the
    random-forest predicted mortality probability).
  - MIMIC-IV v3.1: the raw `hosp/` and `icu/` tables in `$ADAPTIVE_SAMPLING_MIMIC_DIR`; build the cohort once with
    `python -m adaptive_sampling_lib.mimic_cohort`.

| Variable | What | Default |
|---|---|---|
| `ADAPTIVE_SAMPLING_RESULTS` | result JSONs, one folder per experiment | `results/` |
| `ADAPTIVE_SAMPLING_DATA_CACHE` | downloaded datasets, the built MIMIC-IV cohort | `data_cache/` |
| `ADAPTIVE_SAMPLING_EICU_DIR` | eICU files | `data/eicu/` |
| `ADAPTIVE_SAMPLING_MIMIC_DIR` | raw MIMIC-IV v3.1 tables | `data/mimiciv/3.1/` |
| `ADAPTIVE_SAMPLING_MODEL_CACHE` | Kumo-Tabular weights and package | `model_cache/` |
| `TABPFN_MODEL_CACHE_DIR` | TabPFN checkpoints | `model_cache/tabpfn/` |

Defaults are relative to the repository root and ignored by git.

## Running

```bash
python experiments/gp/make_jobs.py --grid paper --out jobs.txt
python -m adaptive_sampling_lib.launch jobs.txt --gpus 0 1 --per_gpu 4 --log_dir logs
```

Each job line ends with `## <output file>`; the launcher skips jobs whose output exists, so an interrupted grid can be
restarted.  Each run writes `<results>/<experiment>/<tag>/<problem>/<method>_ds<split>_s<seed>.json` with the per-epoch
posterior variance of the target, its estimate, the true value and the selected batches.  A single run:

```bash
python experiments/gp/run.py --method pathwise --input_dim 10 --seed 1 --seed_dataset 1 --n_steps 1 --tau 0.5
```

All grids use batches of 5 labels and 10 decision epochs; planners use 10 rollouts per gradient estimate, Adam with
learning rate 0.1, temperature tau = 0.5 for the pathwise planner, and the top-5 logits as the batch.  Low compute is one
gradient step per epoch.  Baselines do not plan, so they are run once and reused for every compute budget.

## The paper's experiments

### GP (`experiments/gp`)

| Experiment | Command |
|---|---|
| Synthetic GP in R^d, d in {1, 5, 10, 20, 50, 100}, 20 function draws, 1/5/10/25/100 gradient steps per epoch | `make_jobs.py --grid paper` |
| eICU, ten training seeds, 1 and 1000 gradient steps per epoch | `make_jobs.py --grid eicu` |
| Rollouts per gradient estimate N in {1, 2, 10} (d = 1, 100 steps per epoch) | `make_jobs.py --grid rollouts` |
| Gradient quality: cosine similarity with the true gradient against N, one function draw per run | `gradient_quality.py --seed_dataset k --gpu 0` |

The synthetic problems have 51 clusters, 100 initial labels from one cluster, a pool of 500 and an evaluation set of 285;
the belief is a GP with the generating hyperparameters.  The eICU problem is a fixed split with the GP hyperparameters
fitted in the paper.  `plot_results.py` prints tables and draws the variance and error curves of any grid (e.g.
`plot_results.py --dataset eicu --tags eicu_low_compute eicu_high_compute`); `analyze_draws.py` compares the posterior
variance with the realised error across function draws; `smoke_test.py` checks the weighted GP update against the GP
conditioned on the selected subset.  The gradient-quality figure:
`python -m adaptive_sampling_lib.plot_gradient_quality --json <results>/gp_high_dim/gradient_quality/*.json --taus 0.1 0.5 10`.

### Kumo-Tabular (`experiments/kumo`)

Four real datasets per task with ten random splits each (k-means selection bias: initial labels from one cluster, the
evaluation set from a few clusters) and a synthetic task; 1, 10 and 50 gradient steps per epoch.  Regression: eICU
mortality risk, California housing, Bike sharing, MIMIC-IV ICU length of stay (target: the MSE of a fixed predictor).
Classification: eICU and MIMIC-IV in-hospital mortality, MAGIC, credit-card default (target: the log-loss of a fixed
classifier).  In the planner's fractional (weighted) update every context row attends to itself at full weight
(`--kumo_self_attn full`) and, for regression, the pseudo-updates use the labelled data's preprocessing statistics
(`--kumo_fixed_stats both`); both leave the update unchanged at binary weights.

| Experiment | Command |
|---|---|
| Regression | `make_jobs.py --task regression --grid final` |
| Classification | `make_jobs.py --task classification --grid final` |

Checks: `check_weighted.py` (the weighted update against the package's forward pass, binary weights, gradients) and
`check_relaxations.py` in `regression/` and `classification/`.

### TabPFN v2.5 (`experiments/tabpfn`)

The same grids with TabPFN v2.5 as the belief (`make_jobs.py --task regression|classification --grid final`).  The
Kumo-Tabular runners reuse these runners' problem construction.  `regression/test_tabpfn_module.py` checks the weighted
update, its gradients and the posterior draws; `regression/gradient_quality.py` is the gradient-quality study with
TabPFN as the belief.

## Figures and tables of the paper (`paper_figures`)

After running the grids above:

```bash
python paper_figures/make_figures.py            # GP in R^d and Kumo-Tabular figures, tables, summary.txt
python paper_figures/make_gp_extra_figures.py   # GP on eICU, rollouts per estimate, gradient quality
```

Both read the results root and write to `paper_figures/output/` (or `--out <folder>`).
