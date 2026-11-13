#!/bin/bash
# Deferred start of the Kumo regression grid on researchgpu04: waits until every job of the classification grid
# (logs_tabpfn/jfinal_refill_and_cls.txt) has been claimed (lock or output), so the classification launcher starts no
# new jobs, then launches the Kumo launcher.  Its per-GPU caps (gpus_gpu04.txt, the classification grid's totals) count
# the running jobs of BOTH pipelines (--cap_by_running _pipeline_), so the agreed totals per GPU are never exceeded.
# Logs go to shared storage (home is at its quota).
cd /user/dm3766/Adaptive_sampling || exit 1
PY=/shared/share_mala/daksh/adaptive_sampling_experiments/.venv/bin/python
LG=/shared/share_mala/daksh/adaptive_sampling_experiments/logs
until $PY - <<'EOF'
import hashlib, os, sys
n = 0
for line in open("logs_tabpfn/jfinal_refill_and_cls.txt"):
    line = line.strip()
    if not line or line.startswith("#") or "tabpfn_classification" not in line:
        continue
    cmd, _, exp = line.partition(" ## "); exp = exp.strip()
    lock = os.path.join("logs_tabpfn/jcls_locks", hashlib.md5((exp or cmd).encode()).hexdigest() + ".lock")
    n += not (os.path.exists(exp) or os.path.exists(lock))
sys.exit(1 if n else 0)
EOF
do sleep 60; done
echo "$(date) classification jobs all claimed; starting the Kumo launcher" >> $LG/start_kumo_gpu04.log
mkdir -p $LG/jkumo_gpu04 $LG/jkumo_locks
exec $PY -m adaptive_sampling_lib.launch kumo_pipeline_regression/jobs_final_rng.txt --gpus_file kumo_pipeline_regression/gpus_gpu04.txt \
    --min_free_mb 16000 --lock_dir $LG/jkumo_locks --cap_by_running _pipeline_ --log_dir $LG/jkumo_gpu04 --python $PY > $LG/launch_jkumo_gpu04.log 2>&1
