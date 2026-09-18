#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_simulate_return_to_room_cohort.py

目的:
  新規の前向き観察研究（帰室時リーク20-2000 mL/min、単施設・同じ施設、n=100+）と
  組み入れ基準を揃えた「学習コホート」をDIAMOND(既存の無選択・後方視的生データ)から
  事後的に再構成した場合、どれくらいの規模・陽性率になるかを試算する。

  diamond_check_persistent_low_leak_exclusion.py（2h〜24hがずっと低い症例を除外）とは
  別物: こちらは「帰室時（記録開始直後）のリーク値」を組み入れ基準として使う
  （前向き観察研究の実際の適格基準に合わせるため）。

  「帰室時リーク」は記録開始直後のノイズを避けるため、既定で記録開始から
  --anchor_window_minutes（既定30分）の平均値として算出する。

出力:
  - 帰室時リークが[--low, --high]（既定20-2000 mL/min）に収まる症例数・比率
  - その部分集団でのtotalleak(陽性)率
  - （参考）その部分集団でのPOD1ルールAUC（ブートストラップ95%CI付き）
  - 症例別の値はローカルCSVにのみ保存、チャット等には出力しない

使い方（例）:
  python diamond_simulate_return_to_room_cohort.py \
      --input_dir ~/Documents/DIAMOND/DIAMOND/Data-fix \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/scored_ge5 \
      --low 20 --high 2000 --anchor_window_minutes 30
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


LABEL_POSITIVE_DIR = "totalleak"
LABEL_NEGATIVE_DIR = "totalnoleak"
POD1_TARGET_HOUR = 9
POD1_WINDOW_HOURS = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="diamond_roc_pod1_flow_devcohort.pyと同じ生データディレクトリ")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--low", type=float, default=20.0, help="帰室時リークの組み入れ下限(mL/min)")
    p.add_argument("--high", type=float, default=2000.0, help="帰室時リークの組み入れ上限(mL/min)")
    p.add_argument(
        "--anchor_window_minutes", type=float, default=30.0,
        help="「帰室時」を記録開始から何分間の平均で見るか(1点ノイズ回避)",
    )
    p.add_argument("--pod_day", type=int, default=1, help="参考AUC算出に使うPOD日")
    p.add_argument("--bootstrap_n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recursive", action="store_true", default=True)
    return p.parse_args()


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def detect_label_from_path(file_path: Path) -> Optional[int]:
    parts = [p.lower() for p in file_path.parts]
    if LABEL_POSITIVE_DIR in parts:
        return 1
    if LABEL_NEGATIVE_DIR in parts:
        return 0
    return None


def find_candidate_files(input_dir: Path, recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [p for p in iterator if p.is_file()]
    allowed_ext = {".csv", ".txt", ".tsv"}
    return [p for p in files if p.suffix.lower() in allowed_ext]


_LAST_GOOD_FORMAT: Dict[str, Optional[str]] = {"encoding": None, "sep": None}


def try_read_table(file_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """diamond_roc_pod1_flow_devcohort.py(2026-08-19高速化版)と同一ロジック。"""
    global _LAST_GOOD_FORMAT
    if _LAST_GOOD_FORMAT["encoding"] is not None:
        enc, sep = _LAST_GOOD_FORMAT["encoding"], _LAST_GOOD_FORMAT["sep"]
        try:
            if sep == "auto":
                df = pd.read_csv(file_path, encoding=enc, sep=None, engine="python")
            else:
                df = pd.read_csv(file_path, encoding=enc, sep=sep, engine="c")
            if df is not None and len(df.columns) >= 2 and len(df) >= 1:
                return df, enc, sep
        except Exception:
            pass

    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]
    fast_seps = [",", "\t", ";"]
    for enc in encodings:
        for sep in fast_seps:
            try:
                df = pd.read_csv(file_path, encoding=enc, sep=sep, engine="c")
                if df is not None and len(df.columns) >= 2 and len(df) >= 1:
                    _LAST_GOOD_FORMAT["encoding"], _LAST_GOOD_FORMAT["sep"] = enc, sep
                    return df, enc, sep
            except Exception:
                continue
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, sep=None, engine="python")
            if df is not None and len(df.columns) >= 2 and len(df) >= 1:
                _LAST_GOOD_FORMAT["encoding"], _LAST_GOOD_FORMAT["sep"] = enc, "auto"
                return df, enc, "auto"
        except Exception:
            continue
    return None, None, None


def resolve_datetime_column(df: pd.DataFrame) -> Optional[str]:
    normalized = {c: normalize_colname(c) for c in df.columns}
    priority_exact = {"date", "datetime", "timestamp", "time", "recordedtime", "recordtime"}
    for c, n in normalized.items():
        if n in priority_exact:
            return c
    for c, n in normalized.items():
        if "date" in n or "time" in n:
            return c
    return None


def resolve_airleak_column(df: pd.DataFrame) -> Optional[str]:
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


def clean_and_standardize(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    dt_col = resolve_datetime_column(df)
    air_col = resolve_airleak_column(df)
    if dt_col is None:
        return None, "datetime column not found"
    if air_col is None:
        return None, "air leak column not found"
    out = df[[dt_col, air_col]].copy()
    out.columns = ["Date", "Air leak"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Air leak"] = pd.to_numeric(out["Air leak"], errors="coerce")
    out = out.dropna(subset=["Date", "Air leak"]).sort_values("Date").reset_index(drop=True)
    if out.empty:
        return None, "no valid rows"
    return out, "ok"


def compute_pod_mean(df: pd.DataFrame, pod_day: int) -> Optional[float]:
    start_time = df["Date"].min()
    target_date = (start_time + pd.Timedelta(days=pod_day)).date()
    target_dt = pd.Timestamp(target_date) + pd.Timedelta(hours=POD1_TARGET_HOUR)
    window_start = target_dt - pd.Timedelta(hours=POD1_WINDOW_HOURS)
    window_end = target_dt + pd.Timedelta(hours=POD1_WINDOW_HOURS)
    sub = df[(df["Date"] >= window_start) & (df["Date"] <= window_end)]
    if sub.empty:
        return None
    return float(sub["Air leak"].mean())


def compute_return_to_room_leak(df: pd.DataFrame, anchor_window_minutes: float) -> Optional[float]:
    """「帰室時リーク」= 記録開始直後anchor_window_minutes分の平均値(1点ノイズを避ける)。"""
    start_time = df["Date"].min()
    w_end = start_time + pd.Timedelta(minutes=anchor_window_minutes)
    sub = df[(df["Date"] >= start_time) & (df["Date"] <= w_end)]
    if sub.empty:
        return None
    return float(sub["Air leak"].mean())


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_candidate_files(input_dir, args.recursive)
    print(f"[INFO] 候補ファイル数: {len(files)}")

    records = []
    for i, fp in enumerate(files):
        if i > 0 and i % 100 == 0:
            print(f"[INFO] 進捗: {i}/{len(files)}件処理済み")
        label = detect_label_from_path(fp)
        if label is None:
            continue
        raw_df, enc, sep = try_read_table(fp)
        if raw_df is None:
            records.append({"case_id": fp.stem, "label": label, "status": "read_failed"})
            continue
        clean, status = clean_and_standardize(raw_df)
        if clean is None:
            records.append({"case_id": fp.stem, "label": label, "status": status})
            continue
        rtr_leak = compute_return_to_room_leak(clean, args.anchor_window_minutes)
        pod_mean = compute_pod_mean(clean, args.pod_day)
        records.append({
            "case_id": fp.stem, "label": label,
            "status": "ok" if rtr_leak is not None else "anchor_window_empty",
            "return_to_room_leak": rtr_leak,
            "pod_mean_airleak": pod_mean,
        })

    df_all = pd.DataFrame(records)
    status_path = output_dir / "dev_return_to_room_case_status.csv"
    df_all.to_csv(status_path, index=False)
    print(f"[OK] 症例別ステータス(ローカルのみ、チャット非出力): {status_path}")

    valid = df_all[(df_all["status"] == "ok") & df_all["return_to_room_leak"].notna()].copy()
    print(f"\n[INFO] 帰室時リーク算出できた症例: n={len(valid)} / 全候補{len(df_all)}")
    print(f"[INFO] 全体の陽性率(totalleak): {int(valid['label'].sum())}/{len(valid)} "
          f"({valid['label'].mean():.1%})")

    included = valid[(valid["return_to_room_leak"] >= args.low) & (valid["return_to_room_leak"] <= args.high)].copy()
    n_before = len(valid)
    n_after = len(included)
    print(
        f"\n===== 帰室時リーク{args.low:.0f}〜{args.high:.0f} mL/minで組み入れシミュレーション ====="
    )
    print(f"[INFO] {n_before}例 -> {n_after}例が該当 ({n_after / n_before:.1%})")
    n_pos = int(included["label"].sum())
    n_neg = n_after - n_pos
    pos_rate = (n_pos / n_after) if n_after else float("nan")
    print(f"[INFO] この部分集団の内訳: 陽性(totalleak)={n_pos}, 陰性(totalnoleak)={n_neg}, 陽性率={pos_rate:.1%}")
    print(
        "[NOTE] この規模・陽性率が、新規前向き観察研究(n=100+、同じ組み入れ基準)と"
        "比較可能な「再構成後の学習コホート」の目安になる。"
    )

    # 参考: この部分集団でのPOD1ルールAUC(既存の分析との連続性のため)
    scored = included[included["pod_mean_airleak"].notna()].copy()
    if len(scored) > 0 and scored["label"].nunique() == 2:
        y = scored["label"].astype(int).to_numpy()
        x = scored["pod_mean_airleak"].to_numpy(dtype=float)
        auc = float(roc_auc_score(y, x))
        ci_lo, ci_hi = bootstrap_auc_ci(y, x, args.bootstrap_n, args.seed)
        print(
            f"\n[参考] この部分集団(n={len(scored)})でのPOD1ルールAUC(既存分析との比較用): "
            f"{auc:.3f} (95%CI {ci_lo:.3f}-{ci_hi:.3f})"
        )
    else:
        print("\n[参考] この部分集団では陽性/陰性が揃っていないかデータ不足のため、参考AUCは算出しません")

    if HAS_MPL and len(valid) > 0:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        colors = {0: "#2a78d6", 1: "#eb6834"}
        for lbl, name in [(0, "totalnoleak"), (1, "totalleak")]:
            vals = valid.loc[valid["label"] == lbl, "return_to_room_leak"]
            vals = vals.clip(upper=vals.quantile(0.99))  # 外れ値で見づらくならないよう99%tileでクリップ(表示のみ)
            ax.hist(vals, bins=40, alpha=0.6, color=colors[lbl], label=f"{name} (n={len(vals)})")
        ax.axvline(args.low, color="gray", linestyle="--", linewidth=1)
        ax.axvline(args.high, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(f"帰室時リーク(最初{args.anchor_window_minutes:.0f}分平均, mL/min, 上位1%はクリップ表示)")
        ax.set_ylabel("症例数")
        ax.set_title(f"単施設コホート: 帰室時リークの分布と組み入れ帯({args.low:.0f}-{args.high:.0f})")
        ax.legend()
        fig.tight_layout()
        out_path = output_dir / "hist_return_to_room_leak_distribution.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] 分布図を保存: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
