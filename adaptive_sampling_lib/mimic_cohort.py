"""Build a MIMIC-IV (v3.1) ICU cohort table from the locally downloaded raw files.

Only the tables that are present locally are used: icu/chartevents (vitals), icu/icustays,
hosp/admissions, hosp/patients.  (hosp/labevents is not downloaded, so no laboratory values.)

Cohort (Karim, Kim & Jeong, Diagnostics 2026, 16(18):2982, 24-h landmark design): first ICU stay of each
adult patient (anchor_age >= 18), ICU stay >= 24 h, and no death on or before the 24-h landmark.
Features (first 24 h of the ICU stay, mean/min/max of the numeric chart values):
  heart rate, systolic/diastolic/mean BP (non-invasive, falling back to arterial), respiratory
  rate, SpO2, temperature (C; F converted), glucose, GCS total (eye+verbal+motor), FiO2,
  plus admission weight, height, age, gender.
Labels: hospital_expire_flag (in-hospital mortality, classification), ICU length of stay in days
(regression, log-transformed downstream), hospital length of stay in days.

  python -m adaptive_sampling_lib.mimic_cohort --out <csv>
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

ROOT = "/shared/share_mala/daksh/mimiciv/physionet.org/files/mimiciv/3.1"
OUT_DEFAULT = "/shared/share_mala/daksh/adaptive_sampling_experiments/data_cache/mimic_icu_first24h.csv"

ITEMS = {
    220045: "hr",
    220179: "sbp_ni", 220180: "dbp_ni", 220181: "mbp_ni",
    220050: "sbp_art", 220051: "dbp_art", 220052: "mbp_art",
    220210: "rr",
    220277: "spo2",
    223762: "temp_c", 223761: "temp_f",
    220621: "glucose", 225664: "glucose", 226537: "glucose",
    220739: "gcs_eye", 223900: "gcs_verbal", 223901: "gcs_motor",
    223835: "fio2",
    226512: "weight_kg", 226531: "weight_lb",
    226730: "height_cm", 226707: "height_in",
}
PLAUSIBLE = {  # crude physiological ranges for outlier removal
    "hr": (20, 300), "sbp": (40, 300), "dbp": (20, 200), "mbp": (30, 250), "rr": (2, 80), "spo2": (50, 100),
    "temp_c": (30, 43), "glucose": (20, 1000), "fio2": (21, 100), "weight_kg": (20, 400), "height_cm": (100, 250),
}


def build(out=OUT_DEFAULT, chunksize=5_000_000):
    t0 = time.time()
    stays = pd.read_csv(f"{ROOT}/icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
    adm = pd.read_csv(f"{ROOT}/hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    pat = pd.read_csv(f"{ROOT}/hosp/patients.csv.gz")
    stays = stays.sort_values("intime").groupby("subject_id", as_index=False).first()  # first ICU stay per patient
    stays = stays[stays["los"] >= 1.0]
    adm["deathtime"] = pd.to_datetime(adm["deathtime"])
    stays = stays.merge(adm[["hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag", "admission_type", "insurance", "race"]], on="hadm_id")
    stays = stays.merge(pat[["subject_id", "gender", "anchor_age"]], on="subject_id")
    stays = stays[stays["anchor_age"] >= 18].copy()
    landmark = stays["intime"] + pd.Timedelta(hours=24)
    stays = stays[~(stays["deathtime"].notna() & (stays["deathtime"] <= landmark))].copy()  # death at/before the landmark
    stays["hosp_los_days"] = (stays["dischtime"] - stays["admittime"]).dt.total_seconds() / 86400
    stays = stays.rename(columns={"los": "icu_los_days"})
    print(f"cohort stays: {len(stays)}  ({time.time() - t0:.0f}s)", flush=True)
    win = stays[["stay_id", "intime"]].set_index("stay_id")
    win["end"] = win["intime"] + pd.Timedelta(hours=24)
    keep_ids = set(win.index)
    agg = {}  # (stay_id, feature) -> [sum, min, max, count]
    parts = []
    n_rows = 0
    reader = pd.read_csv(f"{ROOT}/icu/chartevents.csv.gz", usecols=["stay_id", "charttime", "itemid", "valuenum"],
                         dtype={"stay_id": "int64", "itemid": "int64", "valuenum": "float64"}, chunksize=chunksize)
    for i, ch in enumerate(reader):
        n_rows += len(ch)
        ch = ch[ch["itemid"].isin(ITEMS.keys()) & ch["stay_id"].isin(keep_ids)].dropna(subset=["valuenum"])
        if len(ch) == 0:
            continue
        ch["charttime"] = pd.to_datetime(ch["charttime"])
        w = win.loc[ch["stay_id"].values]
        inwin = (ch["charttime"].values >= w["intime"].values) & (ch["charttime"].values <= w["end"].values)
        ch = ch[inwin]
        ch["feat"] = ch["itemid"].map(ITEMS)
        # unit conversions
        f = ch["feat"] == "temp_f"
        ch.loc[f, "valuenum"] = (ch.loc[f, "valuenum"] - 32) * 5 / 9
        ch.loc[f, "feat"] = "temp_c"
        f = ch["feat"] == "weight_lb"
        ch.loc[f, "valuenum"] = ch.loc[f, "valuenum"] * 0.4536
        ch.loc[f, "feat"] = "weight_kg"
        f = ch["feat"] == "height_in"
        ch.loc[f, "valuenum"] = ch.loc[f, "valuenum"] * 2.54
        ch.loc[f, "feat"] = "height_cm"
        parts.append(ch[["stay_id", "feat", "valuenum"]])
        if i % 10 == 0:
            print(f"  chunk {i}: {n_rows / 1e6:.0f}M rows scanned, kept {sum(len(p) for p in parts) / 1e6:.1f}M  ({time.time() - t0:.0f}s)", flush=True)
    ev = pd.concat(parts, ignore_index=True)
    del parts
    # merge invasive/non-invasive BP: prefer non-invasive, fall back to arterial
    for v in ["sbp", "dbp", "mbp"]:
        ev.loc[ev["feat"] == f"{v}_ni", "feat"] = v
    for v in ["sbp", "dbp", "mbp"]:
        has_ni = set(ev.loc[ev["feat"] == v, "stay_id"])
        art = ev["feat"] == f"{v}_art"
        ev.loc[art & ~ev["stay_id"].isin(has_ni), "feat"] = v
        ev = ev[ev["feat"] != f"{v}_art"]
    # plausibility filter
    for feat, (lo, hi) in PLAUSIBLE.items():
        m = ev["feat"] == feat
        ev = ev[~m | ((ev["valuenum"] >= lo) & (ev["valuenum"] <= hi))]
    gcs = ev[ev["feat"].str.startswith("gcs")].groupby(["stay_id", "feat"])["valuenum"].min().unstack()
    g = gcs.sum(axis=1, min_count=3).rename("gcs_min_total")
    ev = ev[~ev["feat"].str.startswith("gcs")]
    stats = ev.groupby(["stay_id", "feat"])["valuenum"].agg(["mean", "min", "max"])
    wide = stats.unstack("feat")
    wide.columns = [f"{feat}_{stat}" for stat, feat in wide.columns]
    for c in [c for c in wide.columns if c.startswith(("weight_kg", "height_cm"))]:
        if not c.endswith("_mean"):
            wide = wide.drop(columns=c)
    wide = wide.join(g, how="left")
    df = stays.set_index("stay_id").join(wide, how="inner")
    df["gender_male"] = (df["gender"] == "M").astype(int)
    df = df.rename(columns={"anchor_age": "age"})
    feat_cols = [c for c in df.columns if c.endswith(("_mean", "_min", "_max"))] + ["gcs_min_total", "age", "gender_male"]
    out_df = df[["subject_id", "hadm_id", "hospital_expire_flag", "icu_los_days", "hosp_los_days", "first_careunit", "admission_type", "race"] + feat_cols]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    out_df.to_csv(out)
    print(f"saved {out}: {out_df.shape}, mortality rate {out_df['hospital_expire_flag'].mean():.3f}, "
          f"missing per feature: {out_df[feat_cols].isna().mean().round(3).to_dict()}  ({time.time() - t0:.0f}s)", flush=True)
    return out_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_DEFAULT)
    a = p.parse_args()
    build(a.out)
