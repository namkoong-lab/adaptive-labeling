#!/bin/bash
# Regenerate tables + figures for all four experiments (figures land in each experiment folder).
PY=/shared/share_mala/daksh/adaptive_sampling_experiments/.venv/bin/python
R=/shared/share_mala/daksh/adaptive_sampling_experiments/results
cd /user/dm3766/Adaptive_sampling
echo "################ 1. GP high-dim"; $PY gp_pipeline_regression_high_dim/plot_results.py --tags low_compute high_compute
echo "################ 2. Ensemble+ multi-step"; $PY -m adaptive_sampling_lib.plot_runs --root $R/ensemble_multistep --tags complex_v4_low_compute complex_v4_high_compute --baseline_tag complex_v4_low_compute --groups . --T 5 --t_list 0 1 3 5 --out ensemble_plus_pipeline_regression_multistep/figures
echo "################ 2b. Ensemble+ multi-step, 21 clusters (v5)"; $PY -m adaptive_sampling_lib.plot_runs --root $R/ensemble_multistep --tags complex21_v5_low_compute complex21_v5_high_compute --baseline_tag complex21_v5_low_compute --groups . --T 5 --t_list 0 1 3 5 --out ensemble_plus_pipeline_regression_multistep/figures_v5
echo "################ 2c. Gradient quality (TabPFN, eICU)"; $PY -c "import json; d=json.load(open('$R/tabpfn_regression/gradient_quality_eicu.json')); [print(k, round(v['mean'],3), '+-', round(v['se'],3)) for k,v in d['cos'].items()]"
echo "################ 3a. TabPFN regression, skewed pool/eval (headline)"; $PY -m adaptive_sampling_lib.plot_runs --root $R/tabpfn_regression --tags skew_pilot_low skew_pilot_high --baseline_tag skew_pilot_low --T 10 --t_list 0 2 5 10 --out tabpfn_pipeline_regression/figures
echo "################ 3b. TabPFN regression, uniform pool/eval"; $PY -m adaptive_sampling_lib.plot_runs --root $R/tabpfn_regression --tags main_low main_high --baseline_tag main_low --T 10 --t_list 0 2 5 10 --out tabpfn_pipeline_regression/figures
echo "################ 4a. TabPFN classification, skewed pool/eval (headline)"; $PY -m adaptive_sampling_lib.plot_runs --root $R/tabpfn_classification --tags skew_pilot_low skew_pilot_high --baseline_tag skew_pilot_low --T 10 --t_list 0 2 5 10 --out tabpfn_pipeline_classification/figures
echo "################ 4b. TabPFN classification, uniform pool/eval"; $PY -m adaptive_sampling_lib.plot_runs --root $R/tabpfn_classification --tags main_low main_high --baseline_tag main_low --T 10 --t_list 0 2 5 10 --out tabpfn_pipeline_classification/figures
