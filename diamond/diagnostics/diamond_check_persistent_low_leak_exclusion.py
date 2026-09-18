#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_check_persistent_low_leak_exclusion.py

目的:
  「単施設コホートのPOD1ルールAUC(0.895)が高すぎるのは、totalnoleak群の大半が
  そもそも一度も臨床的に意味のある気漏を経験していない"最初から明らかにセーフ"な
  症例だからではないか」という仮説(2026-08-19の考察)を実データで検証する。

  各症例の生時系列データについて、記録開始から--window_start_hours〜
  --window_end_hours（既定2h〜24h）の区間の値がすべて--low〜--high（既定0〜20 mL/min）
  に収まっていれば「実質気漏なし(persistent_low)」としてフラグを立てる。
  このフラグが立った症例を除外した残りのコホートで、POD1ルール
  （diamond_roc_pod1_flow_devcohort.pyと同一ロジック）のAUCを再計算し、
  除外前(0.895)と比較する。

  ⚠️ window内にデータが無い症例(記録がその時間帯に無い等)は「判定不能」として
  persistent_lowフラグを立てない(除外しない)。データが無いことを「気漏なし」と
  誤認しないための安全策。

出力:
  - persistent_lowフラグの内訳（totalleak/totalnoleak群別の該当率、%）
  - 除外前後のAUC比較（ブートストラップ95%CI付き）
  - 症例別の値はローカルCSVにのみ保存、チャット等には出力しない

使い方（例）:
  python diamond_check_persistent_low_leak_exclusion.py \
      --input_dir ~/Documents/DIAMOND/DIAMOND/Data-fix \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/scored_ge5 \
      --window_start_hours 2 --window_end_hours 24 --low 0 --high 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import re

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
    p.add_argument("--window_start_hours", type=float, default=2.0, help="判定窓の開始(記録開始から何時間後)")
    p.add_argument("--window_end_hours", type=float, default=24.0, help="判定窓の終了(記録開始から何時間後)")
    p.add_argument("--low", type=float, default=0.0, help="「実質気漏なし」とみなす流量の下限(mL/min)")
    p.add_argument("--high", type=float, default=20.0, help="「実質気漏なし」とみなす流量の上限(mL/min)")
    p.add_argument("--pod_day", type=int, default=1, help="POD1ルール算出に使うPOD日")
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


def check_persistent_low(
    df: pd.DataFrame, window_start_hours: float, window_end_hours: float, low: float, high: float,
) -> Optional[bool]:
    """記録開始からwindow_start_hours〜window_end_hoursの区間の値がすべて[low,high]に
    収まっていればTrue(実質気漏なし)。区間にデータが無ければNone(判定不能、除外しない)。"""
    start_time = df["Date"].min()
    w_start = start_time + pd.Timedelta(hours=window_start_hours)
    w_end = start_time + pd.Timedelta(hours=window_end_hours)
    sub = df[(df["Date"] >= w_start) & (df["Date"] <= w_end)]
    if sub.empty:
        return None
    return bool((sub["Air leak"] >= low).all() and (sub["Air leak"] <= high).all())


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


def report_auc(y: np.ndarray, x: np.ndarray, tag: str, n_boot: int, seed: int) -> Optional[float]:
    n = len(y)
    n_pos = int(y.sum())
    if n == 0 or len(np.unique(y)) < 2:
        print(f"[{tag}] n={n}, 陽性={n_pos} -> AUC計算不可")
        return None
    auc = float(roc_auc_score(y, x))
    ci_lo, ci_hi = bootstrap_auc_ci(y, x, n_boot, seed)
    print(f"[{tag}] n={n} (陽性{n_pos}, 陰性{n - n_pos}), AUC={auc:.3f} (95%CI {ci_lo:.3f}-{ci_hi:.3f})")
    return auc


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
        pod_mean = compute_pod_mean(clean, args.pod_day)
        persistent_low = check_persistent_low(
            clean, args.window_start_hours, args.window_end_hours, args.low, args.high,
        )
        records.append({
            "case_id": fp.stem, "label": label,
            "status": "ok" if pod_mean is not None else "pod_window_empty",
            "pod_mean_airleak": pod_mean,
            "persistent_low": persistent_low,
        })

    df_all = pd.DataFrame(records)
    status_path = output_dir / "dev_persistent_low_leak_case_status.csv"
    df_all.to_csv(status_path, index=False)
    print(f"[OK] 症例別ステータス(ローカルのみ、チャット非出力): {status_path}")

    scored = df_all[(df_all["status"] == "ok") & df_all["pod_mean_airleak"].notna()].copy()
    print(f"\n[INFO] POD{args.pod_day}ルール計算対象: n={len(scored)}")

    print("\n===== ①除外前(全体、既知の0.895の再確認) =====")
    y_all = scored["label"].astype(int).to_numpy()
    x_all = scored["pod_mean_airleak"].to_numpy(dtype=float)
    report_auc(y_all, x_all, "除外前", args.bootstrap_n, args.seed)

    print(
        f"\n===== ②「開始{args.window_start_hours:.0f}h〜{args.window_end_hours:.0f}hがずっと"
        f"{args.low:.0f}〜{args.high:.0f}mL/min」の症例を除外 ====="
    )
    n_determined = int(scored["persistent_low"].notna().sum())
    n_undetermined = int(scored["persistent_low"].isna().sum())
    print(f"[INFO] 判定可能: {n_determined}件, 判定不能(window内データなし、除外対象外): {n_undetermined}件")

    for lbl, lbl_name in [(0, "totalnoleak(陰性)"), (1, "totalleak(陽性)")]:
        sub = scored[scored["label"] == lbl]
        sub_determined = sub[sub["persistent_low"].notna()]
        n_flagged = int(sub_determined["persistent_low"].sum())
        n_group_determined = len(sub_determined)
        pct = (n_flagged / n_group_determined * 100) if n_group_determined else float("nan")
        print(f"  {lbl_name}: 判定可能{n_group_determined}件中、「実質気漏なし」該当={n_flagged}件 ({pct:.1f}%)")

    remaining = scored[scored["persistent_low"] != True].copy()  # noqa: E712 (NaN=判定不能は残す)
    n_excluded = len(scored) - len(remaining)
    print(f"[INFO] 除外: {len(scored)}例 -> {len(remaining)}例 ({n_excluded}例を除外)")

    y_rem = remaining["label"].astype(int).to_numpy()
    x_rem = remaining["pod_mean_airleak"].to_numpy(dtype=float)
    report_auc(y_rem, x_rem, "除外後", args.bootstrap_n, args.seed)

    print(
        "\n[結論の見方] 除外後のAUCが除外前(0.895)より下がり、分野一般で報告される中程度の値"
        "（目安0.6〜0.7程度）や外部実測値(0.594)に近づくほど、「単施設コホートの高いAUCは"
        "『最初から明らかにセーフな症例』を多く含むことによる見かけ上の高さである」"
        "という仮説の裏付けが強くなる。"
    )

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for y, x, color, label_txt in [
            (y_all, x_all, "#8b5cf6", f"除外前 (n={len(y_all)})"),
            (y_rem, x_rem, "#2a78d6", f"除外後 (n={len(y_rem)})"),
        ]:
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, x)
            auc = roc_auc_score(y, x)
            ax.plot(fpr, tpr, linewidth=2, color=color, label=f"{label_txt}, AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Reference (AUC=0.5)")
        ax.set_xlabel("1 - specificity (FPR)")
        ax.set_ylabel("sensitivity (TPR)")
        ax.set_title("Effect of excluding persistently low-leak cases")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal")
        fig.tight_layout()
        out_path = output_dir / "roc_persistent_low_leak_exclusion_comparison.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] 保存: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
