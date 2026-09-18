#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_external_validate.py

目的:
  diamond_freeze_dev_model.py で保存した凍結モデル（legacy_styleパイプライン、
  開発コホート全体で学習）を、順天堂多施設Thopazデータ（Excelエクスポート、
  'Data'シート）に適用し、外部検証AUROC等を算出する。

  ラベルは2種類算出する:
    (a) rule_label: legacy_styleと同じ比較用ルール（POD{day} air leak > 100mL/min）
        → これは「モデルの入力とは独立な比較対象」であり、正式なラベルではない。
    (b) true_pal_label: REDCap由来の真のPAL定義（エアーリーク消失日 - 手術日 >= 5日）
        → これは本研究の主要アウトカム。REDCap側の症例ID・手術日・エアーリーク消失日
          とThopazファイルの症例IDを突合する必要がある（Step4: ID結合可否の確認と連動）。
        → 今日はこの突合ができる範囲でのみ計算する。

⚠️ 今日の暫定仕様:
  - Excel 'Data'シートの列名は資料確認後に確定させる（--airleak_col等で上書き可能にしてある）。
  - 凍結モデルは固定長入力（scaler_*.jsonのmax_rows）を要求するため、
    外部データもその長さに合わせてpadding/truncationする。
    ※ 開発コホートと外部コホートでThopazのログ記録間隔が異なる場合、
      単純な「行数」パディングでは時間軸の意味がズレる可能性がある（要注意点としてスライドに明記）。

使い方（例）:
  python diamond_external_validate.py \
      --model_dir ~/Documents/DIAMOND/DIAMOND/frozen_24h \
      --horizon_hours 24 \
      --external_dir "/Volumes/Extreme Pro/Thop easy data/RCT-2_...Thopaz dataをexportしたexcel（全施設分）" \
      --redcap_csv "/Volumes/Extreme Pro/Red Cap data/ThopazRCT2_raw.csv" \
      --output_dir ~/Documents/DIAMOND/DIAMOND/external_validation_24h
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, average_precision_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


TARGET_HOUR = 9
WINDOW_HOURS = 1
RULE_THRESHOLD = 100.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="diamond_freeze_dev_model.py の output_dir")
    p.add_argument("--horizon_hours", type=int, choices=[24, 48, 72], required=True)
    p.add_argument("--external_dir", required=True, help="順天堂Excelエクスポートのルートフォルダ")
    p.add_argument("--redcap_csv", default=None, help="ThopazRCT2_raw.csv（真のPAL算出用、任意）")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--date_col", default="Date")
    p.add_argument("--airleak_col", default=None, help="未指定なら自動判定を試みる")
    return p.parse_args()


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_airleak_column(df: pd.DataFrame, override: Optional[str]) -> Optional[str]:
    if override and override in df.columns:
        return override
    normalized = {c: normalize_colname(c) for c in df.columns}
    priority_exact = {"airleak", "airleakmlmin", "airleakml", "leak", "flow"}
    for c, n in normalized.items():
        if n in priority_exact:
            return c
    for c, n in normalized.items():
        if "airleak" in n:
            return c
    for c, n in normalized.items():
        if n == "leak":
            return c
    return None


def find_case_files(external_dir: Path) -> List[Path]:
    return sorted(external_dir.rglob("*.xlsx"))


def load_case_series(file_path: Path, date_col: str, airleak_col_override: Optional[str]) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        df = pd.read_excel(file_path, sheet_name="Data")
    except Exception as e:
        return None, f"read_failed: {e}"

    air_col = resolve_airleak_column(df, airleak_col_override)
    if air_col is None:
        return None, "air leak column not found"
    if date_col not in df.columns:
        return None, "date column not found"

    out = df[[date_col, air_col]].copy()
    out.columns = ["Date", "Air leak"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Air leak"] = pd.to_numeric(out["Air leak"], errors="coerce")
    out = out.dropna(subset=["Date", "Air leak"]).sort_values("Date").reset_index(drop=True)
    if out.empty:
        return None, "no valid rows"
    return out, "ok"


def extract_horizon_series(df: pd.DataFrame, horizon_hours: int) -> Tuple[Optional[np.ndarray], str]:
    start_time = df["Date"].min()
    end_required = start_time + pd.Timedelta(hours=horizon_hours)
    end_time = df["Date"].max()
    if end_time < end_required:
        return None, "drain_removed_before_horizon"
    sub = df[df["Date"] <= end_required].copy()
    if sub.empty:
        return None, "empty_horizon_slice"
    return sub[["Air leak"]].to_numpy(dtype=np.float32), "ok"


def compute_pod_mean(df: pd.DataFrame, pod_day: int) -> Optional[float]:
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


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    def sd(a, b):
        return a / b if b else float("nan")
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "sensitivity": sd(tp, tp + fn), "specificity": sd(tn, tn + fp),
        "PPV": sd(tp, tp + fp), "NPV": sd(tn, tn + fn),
        "accuracy": sd(tp + tn, tp + tn + fp + fn),
    }


def load_true_pal_labels(redcap_csv: Path) -> pd.DataFrame:
    """REDCap生データから真のPAL(surgedat, alendatの差>=5日)を算出。
    ⚠️ 症例IDでThopazファイルと突合する必要がある(Step4と連動、突合キー要確認)。"""
    df = pd.read_csv(redcap_csv)
    # 列名は実データで要確認。以下は想定名。
    candidates_surg = [c for c in df.columns if "surgedat" in normalize_colname(c)]
    candidates_alend = [c for c in df.columns if "alendat" in normalize_colname(c)]
    if not candidates_surg or not candidates_alend:
        raise RuntimeError(f"surgedat/alendat列が見つかりません。実際の列名: {df.columns.tolist()[:30]}")
    surg_col, alend_col = candidates_surg[0], candidates_alend[0]
    df["surgedat_parsed"] = pd.to_datetime(df[surg_col], errors="coerce")
    df["alendat_parsed"] = pd.to_datetime(df[alend_col], errors="coerce")
    df["true_pal"] = ((df["alendat_parsed"] - df["surgedat_parsed"]).dt.days >= 5).astype("Int64")
    return df


def main() -> int:
    args = parse_args()
    model_dir = Path(os.path.expanduser(args.model_dir))
    external_dir = Path(os.path.expanduser(args.external_dir))
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / f"scaler_{args.horizon_hours}h.json") as f:
        scaler_info = json.load(f)
    model = tf.keras.models.load_model(model_dir / f"frozen_model_{args.horizon_hours}h.keras")

    data_min, data_max = scaler_info["data_min"], scaler_info["data_max"]
    max_rows = scaler_info["max_rows"]

    pod_day = args.horizon_hours // 24
    files = find_case_files(external_dir)
    print(f"[INFO] found {len(files)} external xlsx files")

    records = []
    for fp in files:
        raw, status = load_case_series(fp, args.date_col, args.airleak_col)
        if raw is None:
            records.append({"case_id": fp.stem, "file_path": str(fp), "status": status})
            continue
        series, status2 = extract_horizon_series(raw, args.horizon_hours)
        pod_mean = compute_pod_mean(raw, pod_day)
        records.append({
            "case_id": fp.stem,
            "file_path": str(fp),
            "status": status2,
            "pod_mean_airleak": pod_mean,
            "pod_gt100": int(pod_mean > RULE_THRESHOLD) if pod_mean is not None else None,
            "series": series,
        })

    df_all = pd.DataFrame(records)
    df_all.to_csv(output_dir / f"external_case_status_{args.horizon_hours}h.csv", index=False)
    included = df_all[df_all["status"] == "ok"].copy().reset_index(drop=True)
    print(f"[INFO] n_files={len(df_all)}, n_included(horizon到達)={len(included)}")
    print(f"[INFO] status breakdown: {df_all['status'].value_counts().to_dict()}")

    if included.empty:
        raise RuntimeError("外部データでhorizonに到達した症例がありません。列名・シート名を確認してください。")

    # スケーリング(開発コホートのmin/maxを適用、外部データでrefitしない)
    x_list = [pad_or_truncate(s, max_rows) for s in included["series"].tolist()]
    x = np.array(x_list, dtype=np.float32)
    denom = (data_max - data_min) if (data_max - data_min) != 0 else 1.0
    x_scaled = (x - data_min) / denom

    prob = model.predict(x_scaled, verbose=0).flatten()
    included["ai_probability"] = prob

    # ルールベース比較（proxy、参考値）
    rule_df = included[included["pod_gt100"].notna()].copy()

    result_summary = {
        "horizon_hours": args.horizon_hours,
        "n_external_files": len(df_all),
        "n_included": len(included),
        "n_excluded": len(df_all) - len(included),
    }

    # 真のPAL突合（redcap_csvが指定されている場合のみ）
    if args.redcap_csv:
        redcap_df = load_true_pal_labels(Path(os.path.expanduser(args.redcap_csv)))
        redcap_df.to_csv(output_dir / "redcap_true_pal_computed.csv", index=False)
        print(f"[INFO] REDCap側 true_pal 算出済み症例数: {redcap_df['true_pal'].notna().sum()} / {len(redcap_df)}")
        print("[TODO] Thopazファイルのcase_idとREDCapの症例IDの突合キーが未確定。"
              "Step4(ID結合可否の確認)の結果を用いて、ここでマージ処理を追加する必要があります。")

    included_out_cols = ["case_id", "file_path", "pod_mean_airleak", "pod_gt100", "ai_probability"]
    included[included_out_cols].to_csv(output_dir / f"external_predictions_{args.horizon_hours}h.csv", index=False)

    with open(output_dir / f"external_result_summary_{args.horizon_hours}h.json", "w") as f:
        json.dump(result_summary, f, indent=2, default=str)

    print(f"[OK] 外部データ読込・モデル適用まで完了。")
    print(f"[OK] 予測値: {output_dir / f'external_predictions_{args.horizon_hours}h.csv'}")
    print("[NEXT] 真のPALラベルとの突合ができ次第、AUROC等を算出してください（本スクリプトのTODO部分）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
