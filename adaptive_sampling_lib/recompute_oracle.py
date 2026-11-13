"""Post-hoc: add the denoised (oracle-GP) target value and the squared/oracle errors to existing synthetic-data
result files (GP high-dim and Ensemble+), without re-running anything.

  python -m adaptive_sampling_lib.recompute_oracle --gpu 4 [--roots <dir> ...] [--dry]

For every (experiment, dataset args) group it rebuilds the problem and the predictor exactly as the run did,
checks that the realised target matches the stored one (predictor reproduced), computes the oracle truth and
writes sq_err / actual_target_oracle / abs_err_oracle / sq_err_oracle into every epoch record.
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_sampling_lib.data import make_synthetic_gp_problem  # noqa: E402
from adaptive_sampling_lib.oracle_truth import oracle_target  # noqa: E402
from adaptive_sampling_lib.predictor import predictor_outputs, train_predictor  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, seed_all  # noqa: E402

from adaptive_sampling_lib.paths import RESULTS_ROOT as R  # noqa: E402


def group_key(rec):
    a = rec["args"]
    return (a.get("layout", "paper"), a["input_dim"], a.get("seed_dataset", 0), a["n_train"], a["n_pool"], a["n_anchor"],
            a.get("lengthscale"), a["outputscale"], a["noise_std"], a.get("center_scale"), a.get("cluster_std"), a.get("eval_per_cluster"))


def build(rec, dev):
    a = rec["args"]
    seed_all(a.get("seed_dataset", 0))
    if a.get("layout", "paper") == "paper" and "lengthscale" in a and a.get("eval_per_cluster") is None:  # GP high-dim runner
        prob = make_synthetic_gp_problem(a.get("seed_dataset", 0), a["input_dim"], a["n_train"], a["n_pool"], a["n_anchor"],
                                         a["lengthscale"], a["outputscale"], a["noise_std"])
        ls = a["lengthscale"]
    else:  # Ensemble+ runner (build_problem)
        prob = make_synthetic_gp_problem(a["seed_dataset"], a["input_dim"], a["n_train"], a["n_pool"], a["n_anchor"],
                                         outputscale=a["outputscale"], noise_std=a["noise_std"], layout=a["layout"],
                                         center_scale=a["center_scale"], cluster_std=a["cluster_std"], eval_per_cluster=a["eval_per_cluster"])
        ls = prob.meta["lengthscale"]
    prob = prob.to(dev)
    pred = train_predictor(prob.train_x, prob.train_y, "regression", seed=42, device=dev)
    psi = predictor_outputs(pred, prob.eval_x, "regression")
    return prob, psi, ls, a["outputscale"], a["noise_std"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", default=[f"{R}/gp_high_dim/low_compute", f"{R}/gp_high_dim/high_compute",
                                                 f"{R}/ensemble_multistep/complex_v4_low_compute", f"{R}/ensemble_multistep/complex_v4_high_compute",
                                                 f"{R}/ensemble_multistep/complex21_v5_low_compute", f"{R}/ensemble_multistep/complex21_v5_high_compute"])
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--force", action="store_true", help="recompute even if the file already has the oracle fields")
    args = p.parse_args()
    dev = pick_device(args.gpu)
    cache = {}
    n_done = n_skip = n_mismatch = 0
    for root in args.roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
            try:
                rec = json.load(open(f))
            except Exception:
                continue
            if "horizon" not in rec or "args" not in rec:
                continue
            if not args.force and "sq_err_oracle" in rec["horizon"][0] and "truth" in rec:
                n_skip += 1
                continue
            key = group_key(rec)
            if key not in cache:
                prob, psi, ls, os_, ns = build(rec, dev)
                truth = oracle_target(prob, psi, ls, os_, ns)
                cache[key] = truth
                print(f"group {key[:3]} ... q_real={truth['q_real']:.4f} q_oracle={truth['q_oracle']:.4f} (+-{truth['q_oracle_std']:.4f}) "
                      f"q_plugin={truth['q_plugin']:.4f}", flush=True)
            truth = cache[key]
            stored = rec["horizon"][0]["actual_target"]
            if abs(stored - truth["q_real"]) > 1e-3 * max(1.0, abs(stored)):
                n_mismatch += 1
                print(f"  MISMATCH {f}: stored actual {stored:.5f} vs recomputed {truth['q_real']:.5f} (predictor not reproduced; skipping)", flush=True)
                continue
            for m in rec["horizon"]:
                est = m["est_target"]
                m["sq_err"] = (est - m["actual_target"]) ** 2
                m["actual_target_oracle"] = truth["q_oracle"]
                m["abs_err_oracle"] = abs(est - truth["q_oracle"])
                m["sq_err_oracle"] = (est - truth["q_oracle"]) ** 2
            rec["truth"] = dict(truth)
            if not args.dry:
                tmp = f + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(rec, fh)
                os.replace(tmp, f)
            n_done += 1
    print(f"updated {n_done} files, skipped {n_skip} (already done), {n_mismatch} mismatches")


if __name__ == "__main__":
    main()
