#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_roc_pod1_flow_devcohort.py

目的:
  単施設(DIAMOND)コホートについて、生のThopaz時系列ファイル(--input_dir、
  diamond_dev_cohort_cv.py/diamond_freeze_dev_model.pyが学習に使っているのと
  同じディレクトリ構成: totalleak/totalnoleak配下に症例ごとのcsv/txt/tsv、
  Date列とAir leak列を含む)から、POD1(既定)相当時点の平均流量を1症例1値で算出し、
  真のラベル(フォルダ名=totalleak→1/totalnoleak→0、AIモデルの学習に実際に
  使われているラベルそのもの)に対するROC曲線・AUCを算出する。

  外部(RCT)側のdiamond_roc_alflow_external.pyで使う[alflow](POD1時点の1点流量)と
  概念を揃えるため、POD1日の「9:00±1時間」window平均を既定にしている
  （diamond_external_validate.pyのcompute_pod_meanと同一ロジック、2026-08-19に
  単施設側にも展開）。

⚠️ 出力はAUC・n・陽性数などの集計値とROC画像のみ。症例単位の値はローカルCSVに
  保存するがチャット等には出力しない設計。

使い方（例）:
  python diamond_roc_pod1_flow_devcohort.py \
      --input_dir ~/Documents/DIAMOND/DIAMOND/Data-fix \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/scored_ge5 \
      --pod_day 1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


LABEL_POSITIVE_DIR = "totalleak"
LABEL_NEGATIVE_DIR = "totalnoleak"
TARGET_HOUR = 9
WINDOW_HOURS = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="diamond_dev_cohort_cv.pyと同じ生データディレクトリ")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pod_day", type=int, default=1, help="何POD目のwindow平均を取るか(既定1=POD1、外部alflowと概念を揃える)")
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
    """⚠️ 2026-08-19高速化: 旧版は毎ファイル5encoding×5sep(sep=None/python engineを
    最優先で試す最悪の順序)=最大25回、しかも最も遅い自動判定を必ず先に試していたため、
    1000件超のバッチで体感「固まった」ほど遅くなっていた(バグではなく性能問題)。
    同一バッチ内のファイルはほぼ同じ形式である前提で、
    ①直前に成功した(encoding,sep)をまず試す高速パス
    ②Cエンジン+主要区切り文字(明示指定、python engineのsep=Noneより桁違いに速い)
    ③本当にダメな場合のみ低速な自動判定にフォールバック
    の順に変更。結果は変えず、速度だけ改善する。"""
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
    """diamond_external_validate.py の compute_pod_mean と同一ロジック
    （外部側のalflowと概念を揃えるため、意図的に同じ計算式を単施設側にも複製）。"""
    start_time = df["Date"].min()
    target_date = (start_time + pd.Timedelta(days=pod_day)).date()
    target_dt = pd.Timestamp(target_date) + pd.Timedelta(hours=TARGET_HOUR)
    window_start = target_dt - pd.Timedelta(hours=WINDOW_HOURS)
    window_end = target_dt + pd.Timedelta(hours=WINDOW_HOURS)
    sub = df[(df["Date"] >= window_start) & (df["Date"] <= window_end)]
    if sub.empty:
        return None
    return float(sub["Air leak"].mean())


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
            records.append({"case_id": fp.stem, "status": "label_not_found_in_path"})
            continue
        df, enc, sep = try_read_table(fp)
        if df is None:
            records.append({"case_id": fp.stem, "status": "read_failed", "label": label})
            continue
        clean, status = clean_and_standardize(df)
        if clean is None:
            records.append({"case_id": fp.stem, "status": status, "label": label})
            continue
        pod_mean = compute_pod_mean(clean, args.pod_day)
        records.append({
            "case_id": fp.stem, "status": "ok" if pod_mean is not None else "pod_window_empty",
            "label": label, "pod_mean_airleak": pod_mean,
        })

    df_all = pd.DataFrame(records)
    status_path = output_dir / f"dev_pod{args.pod_day}_flow_case_status.csv"
    df_all.to_csv(status_path, index=False)
    print(f"[INFO] ステータス内訳: {df_all['status'].value_counts().to_dict()}")
    print(f"[OK] 症例別ステータス(ローカルのみ、チャット非出力): {status_path}")

    scored = df_all[(df_all["status"] == "ok") & df_all["pod_mean_airleak"].notna() & df_all["label"].notna()].copy()
    y_true = scored["label"].astype(int).to_numpy()
    y_score = scored["pod_mean_airleak"].to_numpy(dtype=float)
    n = len(y_true)
    n_pos = int(y_true.sum())
    print(f"[INFO] ROC計算対象: n={n}, 陽性(totalleak)={n_pos}")

    if n == 0 or len(np.unique(y_true)) < 2:
        print("[ERROR] 陽性/陰性が揃っていないためAUC計算不可（--input_dirやフォルダ構成を確認してください）")
        return 1

    auc = float(roc_auc_score(y_true, y_score))
    print(f"[結果] 単施設(DIAMOND)コホート: POD{args.pod_day}流量によるROC AUC={auc:.3f} (n={n}, 陽性{n_pos})")

    if HAS_MPL:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot(fpr, tpr, color="#2a78d6", linewidth=2,
                label=f"POD{args.pod_day} flow rule (AUC={auc:.3f}, n={n}, pos={n_pos})")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Reference (AUC=0.5)")
        ax.set_xlabel("1 - specificity (FPR)")
        ax.set_ylabel("sensitivity (TPR)")
        ax.set_title(f"Single-center (DIAMOND): label vs POD{args.pod_day} airflow rule")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_aspect("equal")
        fig.tight_layout()
        out_path = output_dir / f"roc_pod{args.pod_day}_flow_devcohort.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] 保存: {out_path}")
    else:
        print("[WARN] matplotlib未インストールのためROC画像は保存しません")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
