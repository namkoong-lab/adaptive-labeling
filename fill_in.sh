#!/bin/bash
# Re-run every sweep's job list; launchers skip jobs whose output exists, so this fills in failed/missing runs.
# usage: bash fill_in.sh <gpu list, e.g. "2 4">
GPUS=${@:-2 4}
PY=/shared/share_mala/daksh/adaptive_sampling_experiments/.venv/bin/python
cd /user/dm3766/Adaptive_sampling
for jf in gp_pipeline_regression_high_dim/jobs_low.txt gp_pipeline_regression_high_dim/jobs_high_rest2.txt \
          ensemble_plus_pipeline_regression_multistep/jobs_full.txt \
          tabpfn_pipeline_regression/jobs_main_light.txt tabpfn_pipeline_regression/jobs_main_heavy.txt \
          tabpfn_pipeline_classification/jobs_main_light.txt tabpfn_pipeline_classification/jobs_main_heavy.txt; do
  name=$(echo $jf | tr '/' '_' | sed 's/.txt//')
  echo "=== $jf"; $PY -m adaptive_sampling_lib.launch $jf --gpus $GPUS --per_gpu 3 --log_dir logs_fill_in/$name --python $PY
done
