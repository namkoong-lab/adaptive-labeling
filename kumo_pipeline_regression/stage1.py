"""Stage 1 of the relaxation x objective test for Kumo-Tabular (regression and classification).  At t=0 of final-grid splits,
where every method starts from the same labelled data, the pathwise planner is rerun at 1, 10 and 50 steps with one
relaxation (--relax weighted / avg_input / avg_emb, --self_attn weighted / full) and one objective (--objective gain /
gain_paired / risk), and every returned batch, together with the grid's random-sampling batch on the same split, is scored
with the standard hard update, the same for every cell:
  gain, drift^2, labvar   E_y[(M_+ - M_t)^2] = drift^2 + Var_y(M_+) over S imagined label vectors (the belief's expectation)
  train_share             share of the batch in the training cluster
  v1_ratio                posterior variance of Q after adding the batch with its TRUE labels / before (64 joint draws,
                          fixed seed), i.e. what the batch actually did to the reported quantity
  python diag_stage1.py --task regression|classification --relax R --self_attn A --objective O --gpu 0
                        [--sets ...] [--splits 1-3] [--steps 1,10,50]
Writes results/diagnostics/stage1/<task>_<relax>_<self_attn>_<objective>_<sets>.json (one file per cell and dataset list,
so jobs of different datasets never write the same file).  It lives in kumo_pipeline_regression/ so that the launchers'
--cap_by_running _pipeline_ counts it with the grid jobs."""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adaptive_sampling_lib.planners import PolicyGradientPlanner  # noqa: E402
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget, LogLossTarget  # noqa: E402
from adaptive_sampling_lib.utils import seed_all  # noqa: E402
p = argparse.ArgumentParser()
p.add_argument("--task", default="regression"); p.add_argument("--relax", default="weighted"); p.add_argument("--self_attn", default="weighted")
p.add_argument("--objective", default="gain"); p.add_argument("--gpu", type=int, default=0); p.add_argument("--sets", default=None)
p.add_argument("--splits", default="1-3"); p.add_argument("--steps", default="1,10,50"); p.add_argument("--S", type=int, default=32)
a = p.parse_args(); dev = torch.device(f"cuda:{a.gpu}"); torch.backends.cuda.matmul.allow_tf32 = True
CLS = a.task == "classification"
ROOT = "/shared/share_mala/daksh/adaptive_sampling_experiments/results"
if CLS:
    RUNPY, GRID, SETS = "kumo_pipeline_classification/run.py", f"{ROOT}/kumo_classification/final_kumo_cls_low/kumo-m_{{ds}}_skewed", "eicu,magic,credit,mimic"
else:
    RUNPY, GRID, SETS = "kumo_pipeline_regression/run.py", f"{ROOT}/kumo_regression/final_kumo_fix_low/kumo-m_{{ds}}_skewed_rng", "eicu_raw,california,bike,mimic_los"
spec = importlib.util.spec_from_file_location("runmod", os.path.join(os.path.dirname(HERE), RUNPY)); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
steps_list = [int(s) for s in a.steps.split(",")]
lo, hi = map(int, a.splits.split("-"))
name = f"{a.task}_{a.relax}_{a.self_attn}_{a.objective}"
os.makedirs(f"{ROOT}/diagnostics/stage1", exist_ok=True)
outf = f"{ROOT}/diagnostics/stage1/{name}_{(a.sets or SETS).replace(',', '+')}.json"
out = json.load(open(outf)) if os.path.exists(outf) else []
done = {(r["setting"], r["split"]) for r in out}


def posterior_var(uq, X, y, eval_x, target, seed):
    st_ = uq.condition(X, y)
    d = uq.posterior_draws(st_, eval_x, 64, torch.Generator(device=dev).manual_seed(seed))
    return float(target.q(d).var())


for ds in (a.sets or SETS).split(","):
    for k in range(lo, hi + 1):
        if (ds, k) in done:
            continue
        G = GRID.format(ds=ds)
        rnd = json.load(open(f"{G}/random_ds{k}_s{k}.json"))
        q = argparse.ArgumentParser(); run.add_args(q); args = q.parse_args(["--method", "pathwise"])
        vars(args).update({kk: v for kk, v in rnd["args"].items() if kk in vars(args)})
        args.kumo_relax, args.kumo_self_attn, args.kumo_fixed_stats, args.objective = a.relax, a.self_attn, "both", a.objective
        seed_all(args.seed_dataset); prob = run.base.build_problem(args, dev)
        kind = "classification" if CLS else "regression"
        psi = predictor_outputs(train_predictor(prob.train_x, prob.train_y, kind, seed=42, device=dev, **({"n_classes": 2} if CLS else {})), prob.eval_x, kind)
        target = LogLossTarget(psi) if CLS else MSETarget(psi, noise_var=0.0)
        uq = run.build_uq(args, prob, dev); st = uq.condition(prob.train_x, prob.train_y); n = prob.pool_x.shape[0]
        with torch.no_grad():
            m_t = float(uq.target_mean(st, prob.eval_x, target))
        intrain = (prob.pool_cluster == rnd["problem"]["train_cluster"]).to(dev)
        Y = uq.sample_pool_labels(st, prob.pool_x, a.S, torch.Generator(device=dev).manual_seed(7)).float()
        batches, times = {"random": torch.tensor(rnd["selected"][0], device=dev)}, {}
        for steps in steps_list:
            seed_all(k)
            planner = PolicyGradientPlanner(uq, target, 5, n_steps=steps, G=10, n_mc=16, lr=0.1, tau=0.5, estimator="pathwise", objective=a.objective,
                                            grad_chunk=2, imagined_labels="per_step")
            t0 = time.time()
            idx, _ = planner.select(st, prob.pool_x, prob.eval_x, torch.Generator(device=dev).manual_seed(100 + k))
            times[f"PW {steps}"] = time.time() - t0; batches[f"PW {steps}"] = idx.to(dev)
        labs = list(batches)
        W = torch.zeros(len(labs), n, device=dev)
        for g, l in enumerate(labs):
            W[g, batches[l]] = 1.0
        with torch.no_grad():
            M = torch.stack([uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, Y[s], W, target, differentiable=False).float() for s in range(a.S)]).cpu().numpy()
        v0 = posterior_var(uq, prob.train_x, prob.train_y, prob.eval_x, target, 11)
        rec = {"setting": ds, "split": k, "m_t": m_t, "v0": v0, "methods": {}}
        for g, l in enumerate(labs):
            idx = batches[l]
            v1 = posterior_var(uq, torch.cat([prob.train_x, prob.pool_x[idx]]), torch.cat([prob.train_y, prob.pool_y[idx]]), prob.eval_x, target, 11)
            mp = M[:, g]
            rec["methods"][l] = {"gain": float(np.mean((mp - m_t) ** 2)), "drift2": float((mp.mean() - m_t) ** 2), "labvar": float(mp.var()),
                                 "train_share": float(intrain[idx].float().mean()), "v1_ratio": v1 / v0, "seconds": times.get(l)}
        out.append(rec)
        json.dump(out, open(outf, "w"))
        r = rec["methods"]
        print(f"[{name}] {ds:10s} split {k}: gain/random " + " ".join(f"{l} {r[l]['gain'] / r['random']['gain']:.2f}" for l in labs if l != "random")
              + " | var after true labels / before " + " ".join(f"{l} {r[l]['v1_ratio']:.2f}" for l in labs)
              + " | in training cluster " + " ".join(f"{l} {r[l]['train_share']:.1f}" for l in labs), flush=True)
        del uq; torch.cuda.empty_cache()
open(outf + ".done", "w").write("ok\n")  # marker for the launcher (skips finished jobs)
print("done", name)
