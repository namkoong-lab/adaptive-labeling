"""Job lists for the TabPFN experiments (regression and classification share this script via --task).
  python make_jobs.py --task regression --datasets eicu california bike --seeds 0-2 --out_prefix jobs_pilot
Writes <out_prefix>_light.txt (baselines + REINFORCE) and <out_prefix>_heavy.txt (pathwise)."""
import argparse, os
ROOT = "/shared/share_mala/daksh/adaptive_sampling_experiments/results"
HERE = os.path.dirname(os.path.abspath(__file__))

def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out

p = argparse.ArgumentParser()
p.add_argument("--task", default="regression", choices=["regression", "classification"])
p.add_argument("--datasets", nargs="+", default=["eicu", "california", "bike"])
p.add_argument("--seeds", default="0-2")
p.add_argument("--regimes", nargs="+", default=["low", "high"])
p.add_argument("--n_steps_high", type=int, default=50)
p.add_argument("--methods", nargs="+", default=["random", "uncertainty", "uncertainty_seq", "reinforce", "reinforce_baseline", "pathwise"])
p.add_argument("--tag_prefix", default="pilot")
p.add_argument("--extra", default="")
p.add_argument("--group_suffix", default="", help="e.g. _skewed when --extra contains '--layout skewed'")
p.add_argument("--out_prefix", default="jobs_pilot")
p.add_argument("--dataset_seeds", default=None, help="if given: one run per dataset split k (new train cluster / eval set / predictor) with training seed k")
a = p.parse_args()
run_py = os.path.join(os.path.dirname(HERE), f"tabpfn_pipeline_{a.task}", "run.py")
res_root = os.path.join(ROOT, f"tabpfn_{a.task}")
light, heavy = [], []
for regime in a.regimes:
    n_steps = 1 if regime == "low" else (int(regime[3:]) if regime.startswith("mid") else a.n_steps_high)  # mid<k>: k steps
    tag = f"{a.tag_prefix}_{regime}"
    for ds in a.datasets:
        for m in a.methods:
            if regime != "low" and m in ("random", "uncertainty", "uncertainty_seq"): continue
            pairs = [(0, s) for s in parse_range(a.seeds)] if a.dataset_seeds is None else [(k, k) for k in parse_range(a.dataset_seeds)]
            for dsd, s in pairs:
                out = os.path.join(res_root, tag, f"tabpfn_{ds}{a.group_suffix}", f"{m}_ds{dsd}_s{s}.json")
                cmd = f"{{PY}} {run_py} --dataset {ds} --method {m} --seed {s} --seed_dataset {dsd} --n_steps {n_steps} --tag {tag} --gpu 0 {a.extra} ## {out}"
                (heavy if m == "pathwise" else light).append(cmd)
for name, lst in [("light", light), ("heavy", heavy)]:
    with open(f"{a.out_prefix}_{name}.txt", "w") as f: f.write("\n".join(lst) + "\n")
    print(f"{a.out_prefix}_{name}.txt: {len(lst)} jobs")
