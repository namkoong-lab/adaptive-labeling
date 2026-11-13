"""Do the drift-free gain (a) and the expected posterior variance (b) rank batches the same way, and which one predicts what a
batch actually does?  At t=0 of final-grid splits (Kumo-Tabular medium, the grid's settings), for ~25 candidate hard batches
(every method's first batch in the Kumo final grid and 16 random five-subsets), with the standard hard update:
  gain   E_y[(M_+ - M_t)^2]             over S imagined label vectors (the planners' current objective)
  (a)    Var_y(M_+)                     its drift-free part
  (b)    E_y[Var(Q | D_+)]              posterior variance of Q after the update, Sb label vectors x Sd joint draws (4 chunks)
  real   Var(Q | D + batch, true labels) / Var(Q | D)   (64 joint draws)
and, per state, Spearman correlations between (a), (b), gain and the realised variance (a better batch has larger (a) and
gain but smaller (b) and realised variance; the correlations are signed so that agreement is positive).
  python agreement.py --task regression|classification --gpu 0 [--sets california] [--splits 1-3]"""
import argparse, glob, importlib.util, json, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))  # repo root
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget, LogLossTarget  # noqa: E402
from adaptive_sampling_lib.utils import seed_all  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--task", default="regression"); p.add_argument("--gpu", type=int, default=0)
p.add_argument("--sets", default=None); p.add_argument("--splits", default="1-3"); p.add_argument("--S", type=int, default=32)
p.add_argument("--Sb", type=int, default=16); p.add_argument("--Sd", type=int, default=32)
a = p.parse_args(); dev = torch.device(f"cuda:{a.gpu}"); torch.backends.cuda.matmul.allow_tf32 = True
CLS = a.task == "classification"
from adaptive_sampling_lib.paths import RESULTS_ROOT as ROOT  # noqa: E402
if CLS:
    RUNPY, GRID, SETS = "experiments/kumo/classification/run.py", f"{ROOT}/kumo_classification/final_kumo_cls_{{reg}}/kumo-m_{{ds}}_skewed", "mimic"
else:
    RUNPY, GRID, SETS = "experiments/kumo/regression/run.py", f"{ROOT}/kumo_regression/final_kumo_fix_{{reg}}/kumo-m_{{ds}}_skewed_rng", "california"
spec = importlib.util.spec_from_file_location("runmod", os.path.join(HERE, "..", "..", "..", RUNPY)); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
lo, hi = map(int, a.splits.split("-")); out = []
for ds in (a.sets or SETS).split(","):
    for k in range(lo, hi + 1):
        rnd = json.load(open(f"{GRID.format(reg='low', ds=ds)}/random_ds{k}_s{k}.json"))
        q = argparse.ArgumentParser(); run.add_args(q); args = q.parse_args(["--method", "pathwise"])
        vars(args).update({kk: v for kk, v in rnd["args"].items() if kk in vars(args)}); args.kumo_fixed_stats = "both"
        seed_all(args.seed_dataset); prob = run.base.build_problem(args, dev)
        kind = "classification" if CLS else "regression"
        psi = predictor_outputs(train_predictor(prob.train_x, prob.train_y, kind, seed=42, device=dev, **({"n_classes": 2} if CLS else {})), prob.eval_x, kind)
        target = LogLossTarget(psi) if CLS else MSETarget(psi, noise_var=0.0)
        uq = run.build_uq(args, prob, dev); st = uq.condition(prob.train_x, prob.train_y); n = prob.pool_x.shape[0]
        with torch.no_grad():
            m_t = float(uq.target_mean(st, prob.eval_x, target))
        cands = {}
        for reg in ("low", "mid10", "high"):
            for f in sorted(glob.glob(f"{GRID.format(reg=reg, ds=ds)}/*_ds{k}_s{k}.json")):
                m = os.path.basename(f).split("_ds")[0]
                lab = m if reg == "low" and m in ("random", "uncertainty", "uncertainty_seq") else f"{m} {reg}"
                if reg != "low" and m in ("random", "uncertainty", "uncertainty_seq"):
                    continue
                d = json.load(open(f))
                if "total_time" in d:
                    cands[lab] = d["selected"][0]
        g = torch.Generator().manual_seed(1000 + k)
        for r in range(16):
            cands[f"subset {r}"] = torch.randperm(n, generator=g)[:5].tolist()
        labs = list(cands)
        W = torch.zeros(len(labs), n, device=dev)
        for i, l in enumerate(labs):
            W[i, torch.tensor(cands[l], device=dev)] = 1.0
        Y = uq.sample_pool_labels(st, prob.pool_x, a.S, torch.Generator(device=dev).manual_seed(7)).float()
        with torch.no_grad():
            M = torch.stack([uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, Y[s], W, target, differentiable=False).float() for s in range(a.S)]).cpu().numpy()
            Vb = []
            for s in range(a.Sb):  # (b): posterior variance of Q after each batch under imagined labels s
                dr = uq.rollout_draws(st, prob.eval_x, prob.pool_x, Y[s], W, a.Sd, differentiable=False, generator=torch.Generator(device=dev).manual_seed(500 + s))
                Vb.append(target.q(dr).var(-1).cpu().numpy())
        Vb = np.mean(Vb, 0)
        gain, va = ((M - m_t) ** 2).mean(0), M.var(0)

        def pv(X, y):
            d = uq.posterior_draws(uq.condition(X, y), prob.eval_x, 64, torch.Generator(device=dev).manual_seed(11))
            return float(target.q(d).var())
        v0 = pv(prob.train_x, prob.train_y)
        real = np.array([pv(torch.cat([prob.train_x, prob.pool_x[cands[l]]]), torch.cat([prob.train_y, prob.pool_y[cands[l]]])) / v0 for l in labs])
        cor = {"a_vs_b": spearmanr(va, -Vb).correlation, "gain_vs_a": spearmanr(gain, va).correlation, "a_vs_real": spearmanr(va, -real).correlation,
               "b_vs_real": spearmanr(-Vb, -real).correlation, "gain_vs_real": spearmanr(gain, -real).correlation}
        out.append({"setting": ds, "split": k, "m_t": m_t, "v0": v0, "labels": labs, "gain": gain.tolist(), "a": va.tolist(), "b": Vb.tolist(),
                    "real": real.tolist(), "spearman": cor})
        print(f"[{a.task}] {ds} split {k}: {len(labs)} batches | Spearman (a) vs (b) {cor['a_vs_b']:+.2f}, gain vs (a) {cor['gain_vs_a']:+.2f} | "
              f"predicting the realised variance: (a) {cor['a_vs_real']:+.2f}, (b) {cor['b_vs_real']:+.2f}, gain {cor['gain_vs_real']:+.2f}", flush=True)
        del uq; torch.cuda.empty_cache()
os.makedirs(f"{ROOT}/diagnostics/stage1", exist_ok=True)
outf = f"{ROOT}/diagnostics/stage1/agreement_{a.task}.json"
json.dump(out, open(outf, "w")); open(outf + ".done", "w").write("ok\n"); print("saved", outf)
