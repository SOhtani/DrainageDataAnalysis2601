#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_step1_duration_feasibility.py

目的（Step 1）:
  「複合アウトカム(PAL)にいきなり飛びつかず、まず定義が明確な連続変数
  （ドレーン留置期間）が、ドレーンデータのトレンドから予測できるかを軽く確認する」
  ための、軽量な特徴量＋相関チェックスクリプト。

  深層学習は使わない（重い・遅い）。全期間のair leakデータから単純な要約統計量を
  作り、実測の留置期間（記録の最終時刻 - 開始時刻）との相関・簡易回帰(R^2)を見る。
  ここで信号が全く無ければ、その先のCNN構築・カットオフ検討に進む前に立ち止まるべき。

  Step 2（カットオフ別）は、--cutoff_hours を指定すれば同じロジックで実行できる。
  未指定（デフォルト）なら「全期間」を使う。

使い方（DIAMOND開発コホート、全期間）:
  python diamond_step1_duration_feasibility.py \
      --input_dir ~/Documents/DIAMOND/DIAMOND/Data-fix \
      --output_csv ~/Desktop/step1_dev_fullseries.csv

使い方（カットオフ指定、例: 24h）:
  python diamond_step1_duration_feasibility.py \
      --input_dir ~/Documents/DIAMOND/DIAMOND/Data-fix \
      --output_csv ~/Desktop/step1_dev_24h.csv \
      --cutoff_hours 24

使い方（順天堂外部データ、Excel）:
  python diamond_step1_duration_feasibility.py \
      --input_dir "/Volumes/Extreme Pro/Thop easy data/RCT-2_...Thopaz dataをexportしたexcel（全施設分）" \
      --output_csv ~/Desktop/step1_external_fullseries.csv \
      --file_type xlsx
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--file_type", choices=["csv", "xlsx"], default="csv")
    p.add_argument("--cutoff_hours", type=float, default=None, help="未指定なら全期間を使う")
    return p.parse_args()


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_col(df: pd.DataFrame, exact_set: set, contains: Optional[str] = None) -> Optional[str]:
    normalized = {c: normalize_colname(c) for c in df.columns}
    for c, n in normalized.items():
        if n in exact_set:
            return c
    if contains:
        for c, n in normalized.items():
            if contains in n:
                return c
    return None


def load_series(file_path: Path, file_type: str) -> Optional[pd.DataFrame]:
    try:
        if file_type == "xlsx":
            df = pd.read_excel(file_path, sheet_name="Data")
        else:
            for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
                for sep in [None, ";", ","]:
                    try:
                        df = pd.read_csv(file_path, encoding=enc, sep=sep, engine="python")
                        if len(df.columns) >= 2:
                            raise StopIteration  # break out with this df
                    except StopIteration:
                        break
                    except Exception:
                        continue
                else:
                    continue
                break
            else:
                return None
    except Exception:
        return None

    dt_col = resolve_col(df, {"date", "datetime", "timestamp"}, "date")
    air_col = resolve_col(df, {"airleak", "airleakmlmin", "airleakml", "leak", "flow"}, "airleak")
    if dt_col is None or air_col is None:
        return None

    out = df[[dt_col, air_col]].copy()
    out.columns = ["Date", "Air leak"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Air leak"] = pd.to_numeric(out["Air leak"], errors="coerce")
    out = out.dropna().sort_values("Date").reset_index(drop=True)
    return out if not out.empty else None


def summarize_case(df: pd.DataFrame, cutoff_hours: Optional[float]) -> Optional[dict]:
    start = df["Date"].min()
    full_end = df["Date"].max()
    duration_hours = (full_end - start).total_seconds() / 3600.0

    if cutoff_hours is not None:
        cutoff_time = start + pd.Timedelta(hours=cutoff_hours)
        if full_end < cutoff_time:
            # このカットオフに到達していない症例。Step1では「全期間版」を推奨し、
            # カットオフ版ではこの扱いを別途検討する（除外 or 打ち切りとして扱う等）。
            return None
        window = df[df["Date"] <= cutoff_time]
    else:
        window = df

    if window.empty:
        return None

    leak = window["Air leak"].to_numpy(dtype=float)
    n = len(leak)

    # 単純な要約特徴量（重い学習をせず、まず「信号があるか」を見る）
    feats = {
        "n_points": n,
        "mean_leak": float(np.mean(leak)),
        "median_leak": float(np.median(leak)),
        "max_leak": float(np.max(leak)),
        "sd_leak": float(np.std(leak)),
        "cv_leak": float(np.std(leak) / np.mean(leak)) if np.mean(leak) != 0 else np.nan,
        "frac_time_gt100": float(np.mean(leak > 100)),
        "last_value": float(leak[-1]),
        "trend_slope": float(np.polyfit(np.arange(n), leak, 1)[0]) if n >= 3 else np.nan,
        "duration_hours_full": float(duration_hours),
    }
    return feats


def main() -> int:
    args = parse_args()
    input_dir = Path(os.path.expanduser(args.input_dir))

    if args.file_type == "xlsx":
        files = sorted(input_dir.rglob("*.xlsx"))
    else:
        files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in {".csv", ".txt", ".tsv"}])

    print(f"[INFO] found {len(files)} files")

    rows = []
    n_failed = 0
    n_cutoff_not_reached = 0
    for fp in files:
        df = load_series(fp, args.file_type)
        if df is None:
            n_failed += 1
            continue
        feats = summarize_case(df, args.cutoff_hours)
        if feats is None:
            n_cutoff_not_reached += 1
            continue
        feats["case_id"] = fp.stem
        feats["file_path"] = str(fp)
        rows.append(feats)

    result = pd.DataFrame(rows)
    result.to_csv(os.path.expanduser(args.output_csv), index=False)

    print(f"[INFO] n_readable_ok={len(result)}, n_read_failed={n_failed}, n_cutoff_not_reached={n_cutoff_not_reached}")

    if len(result) < 5:
        print("[WARN] サンプル数が少なすぎて相関の評価は困難です。")
        return 0

    y = result["duration_hours_full"].to_numpy(dtype=float)
    print("\n[Step1] 留置期間（実測、全期間ベース）との相関（Spearman）:")
    for col in ["mean_leak", "median_leak", "max_leak", "sd_leak", "cv_leak", "frac_time_gt100", "last_value", "trend_slope", "n_points"]:
        x = result[col].to_numpy(dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 5:
            continue
        rho, p = spearmanr(x[mask], y[mask])
        print(f"  {col:20s}: rho={rho:+.3f}  p={p:.4f}  (n={mask.sum()})")

    print(f"\n[OK] 出力: {args.output_csv}")
    print("[NEXT] 相関がある特徴量（|rho|が大きい）があれば、Step2（カットオフ別）に進む価値あり。")
    print("       全くrhoが出ない場合は、そもそもair leakトレンドと留置期間の関係が弱い可能性があり、")
    print("       特徴量設計・データ品質(Step0 QC)を先に見直すべき。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
