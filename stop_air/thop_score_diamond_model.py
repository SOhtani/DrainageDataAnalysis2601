#!/usr/bin/env python3
"""
thop_score_diamond_model.py

Phase 3 of the STOP_AIR initial analysis (see PLAN_STOP_AIR.md).

Scores the 63 STOP_AIR cases (from target_manifest.csv, n_data_rows > 0)
with the DIAMOND frozen 24h model, and computes a preliminary AUROC
against the proxy PAL labels already computed by thop_case_summary.py.

Reuses the exact preprocessing pipeline established for the C-line
(Juntendo RCT) external validation (diamond_external_validate.py):
  1. take each case's Air-leak series from start up to horizon_hours
     (exclude if drainage was removed before reaching the horizon --
     none of our 63 cases should hit this, since they were already
     filtered to duration >= 24h)
  2. pad/truncate to max_rows (from scaler_24h.json)
  3. scale with (x - data_min) / (data_max - data_min) -- the DEV
     cohort's min/max, not refit on this data
  4. run through the frozen model (weights-only .h5, architecture
     rebuilt here to match, since the M1/M2 Mac save bug forced a
     weights-only save format back when this was frozen)

Usage:
  python3 thop_score_diamond_model.py \
      --manifest ~/Desktop/stop_air_target_csv/target_manifest.csv \
      --case_summary ~/Desktop/stop_air_target_csv/stop_air_case_summary.csv \
      --model_dir ~/Documents/DIAMOND/DIAMOND/frozen_24h \
      --horizon_hours 24
"""

import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

TARGET_HOUR = 9
WINDOW_HOURS = 1
RULE_THRESHOLD = 100.0


def parse_dt(s: str) -> datetime.datetime:
    # "3/27/23 10:15" -- same format thop_log_to_csv.py / thop_extract_target.py write
    date_part, time_part = s.split(" ")
    m, d, y = date_part.split("/")
    h, mi = time_part.split(":")
    yy = int(y)
    yy += 2000 if yy < 70 else 1900
    return datetime.datetime(yy, int(m), int(d), int(h), int(mi))


def read_data_csv(path: str):
    """Returns a DataFrame with columns Date (datetime), Air leak (float), sorted."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                dt = parse_dt(row["Date"])
                air_leak = float(row["Air leak [ml/min]"])
            except (KeyError, ValueError):
                continue
            rows.append((dt, air_leak))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["Date", "Air leak"]).sort_values("Date").reset_index(drop=True)
    return df


def extract_horizon_series(df: pd.DataFrame, horizon_hours: int):
    start_time = df["Date"].min()
    end_required = start_time + pd.Timedelta(hours=horizon_hours)
    end_time = df["Date"].max()
    if end_time < end_required:
        return None, "drain_removed_before_horizon"
    sub = df[df["Date"] <= end_required]
    if sub.empty:
        return None, "empty_horizon_slice"
    return sub[["Air leak"]].to_numpy(dtype=np.float32), "ok"


def compute_pod_mean(df: pd.DataFrame, pod_day: int):
    start_time = df["Date"].min()
    target_date = (start_time + pd.Timedelta(days=pod_day)).date()
    target_dt = pd.Timestamp(target_date) + pd.Timedelta(hours=TARGET_HOUR)
    window_start = target_dt - pd.Timedelta(hours=WINDOW_HOURS)
    window_end = target_dt + pd.Timedelta(hours=WINDOW_HOURS)
    sub = df[(df["Date"] >= window_start) & (df["Date"] <= window_end)]
    if sub.empty:
        return None
    return float(sub["Air leak"].mean())


def pad_or_truncate(item: np.ndarray, target_len: int) -> np.ndarray:
    if item.shape[0] < target_len:
        return np.pad(item, ((0, target_len - item.shape[0]), (0, 0)), "constant")
    return item[:target_len]


def create_cnn_model(input_shape):
    """Must match diamond_freeze_dev_model.py's architecture exactly --
    we're loading weights into this, not a saved full model."""
    model = Sequential()
    model.add(Conv1D(64, 3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="target_manifest.csv from thop_extract_target.py")
    ap.add_argument("--case_summary", required=True, help="stop_air_case_summary.csv from thop_case_summary.py")
    ap.add_argument("--model_dir", required=True, help="dir containing frozen_weights_{h}h.h5 and scaler_{h}h.json")
    ap.add_argument("--horizon_hours", type=int, default=24, choices=[24, 36, 48, 72])
    ap.add_argument("-o", "--out", help="output path (default: stop_air_diamond_scores_{h}h.csv next to case_summary)")
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    scaler_path = os.path.join(model_dir, f"scaler_{args.horizon_hours}h.json")
    weights_path = os.path.join(model_dir, f"frozen_weights_{args.horizon_hours}h.h5")
    if not os.path.exists(scaler_path):
        sys.exit(f"ERROR: scaler not found at {scaler_path}")
    if not os.path.exists(weights_path):
        sys.exit(f"ERROR: weights not found at {weights_path}")

    with open(scaler_path) as f:
        scaler_info = json.load(f)
    data_min, data_max, max_rows = scaler_info["data_min"], scaler_info["data_max"], scaler_info["max_rows"]
    print(f"[INFO] loaded scaler: data_min={data_min}, data_max={data_max}, max_rows={max_rows}")

    model = create_cnn_model(input_shape=(max_rows, 1))
    model.load_weights(weights_path)
    print(f"[INFO] loaded weights from {weights_path}")

    manifest_rows = list(csv.DictReader(open(args.manifest, encoding="utf-8-sig")))

    def n_rows(r):
        try:
            return int(r.get("n_data_rows") or 0)
        except ValueError:
            return 0

    usable = [r for r in manifest_rows if n_rows(r) > 0]
    by_folder = defaultdict(list)
    for r in usable:
        by_folder[r["Folder"]].append(r)

    case_labels = {}
    with open(args.case_summary, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            case_labels[row["Folder"]] = row["proxy_pal_label"]

    records = []
    for folder, segs in sorted(by_folder.items()):
        combined = None
        for seg in segs:
            df = read_data_csv(seg["data_csv"])
            if df is None:
                continue
            combined = df if combined is None else pd.concat([combined, df], ignore_index=True)
        if combined is None or combined.empty:
            records.append({"Folder": folder, "status": "no_data", "proxy_pal_label": case_labels.get(folder, "")})
            continue
        combined = combined.sort_values("Date").reset_index(drop=True)

        series, status = extract_horizon_series(combined, args.horizon_hours)
        pod_mean = compute_pod_mean(combined, pod_day=args.horizon_hours // 24)
        rec = {
            "Folder": folder,
            "status": status,
            "proxy_pal_label": case_labels.get(folder, ""),
            "pod_mean_airleak": pod_mean,
            "pod_gt100": (pod_mean is not None and pod_mean > RULE_THRESHOLD),
        }
        if status == "ok":
            rec["series"] = series
        records.append(rec)

    df_all = pd.DataFrame(records)
    included = df_all[df_all["status"] == "ok"].copy().reset_index(drop=True)
    print(f"[INFO] n_cases={len(df_all)}, n_included(horizon到達)={len(included)}")
    print(f"[INFO] status breakdown: {df_all['status'].value_counts().to_dict()}")

    if included.empty:
        sys.exit("ERROR: no case reached the horizon -- nothing to score.")

    x_list = [pad_or_truncate(s, max_rows) for s in included["series"].tolist()]
    x = np.array(x_list, dtype=np.float32)
    denom = (data_max - data_min) if (data_max - data_min) != 0 else 1.0
    x_scaled = (x - data_min) / denom

    prob = model.predict(x_scaled, verbose=0).flatten()
    included["ai_probability"] = prob

    y_true = (included["proxy_pal_label"] == "positive").astype(int).to_numpy()
    y_prob = included["ai_probability"].to_numpy()

    auc = safe_auc(y_true, y_prob)
    print(f"\n[RESULT] STOP_AIR proxy-label AUROC (horizon={args.horizon_hours}h, n={len(included)}): {auc:.3f}")

    # rule-based comparator, same style as C-line reporting
    rule_df = included[included["pod_mean_airleak"].notna()]
    if not rule_df.empty:
        y_true_rule = (rule_df["proxy_pal_label"] == "positive").astype(int).to_numpy()
        y_rule = rule_df["pod_gt100"].astype(int).to_numpy()
        rule_auc = safe_auc(y_true_rule, y_rule)
        print(f"[RESULT] rule-based comparator (POD{args.horizon_hours//24} mean air leak > {RULE_THRESHOLD} mL/min) AUROC: {rule_auc:.3f} (n={len(rule_df)})")

    # sensitivity/specificity at Youden-optimal threshold (descriptive only --
    # this threshold is chosen on this same small sample, so treat as exploratory)
    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden = tpr - fpr
        best_idx = int(np.argmax(youden))
        best_thr = thresholds[best_idx]
        y_pred = (y_prob >= best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        print(f"[INFO] Youden threshold={best_thr:.3f}: sensitivity={sens:.3f}, specificity={spec:.3f}")

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.case_summary)), f"stop_air_diamond_scores_{args.horizon_hours}h.csv")
    included_out = included[["Folder", "proxy_pal_label", "pod_mean_airleak", "pod_gt100", "ai_probability"]]
    included_out.to_csv(out_path, index=False)
    print(f"\n[OK] per-case predictions -> {out_path}")


if __name__ == "__main__":
    main()
