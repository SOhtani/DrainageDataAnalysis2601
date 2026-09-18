#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_dev_cohort_cv.py

目的:
  開発コホート(DIAMOND)の「モデル構築コホートでの精度」を、9:1の1回だけの
  分割（internal test陽性例わずか10例で結果が不安定）でなく、
  stratified k分割交差検証で評価する。全症例が漏れなく1回ずつheld-outとして
  使われるため（5-foldならheld-out陽性例は毎回約20例）、はるかに頑健な
  内部検証AUCが得られる。

  データ読込・ラベル付与・前処理（clean_and_standardize, extract_horizon_series,
  create_cnn_model等）は diamond_freeze_dev_model.py と完全に同一ロジックを
  用いる（依存を減らすため意図的にコピー）。

  ⚠️ 位置づけ（2026-08-16、本人との協議で確定）:
    この内部CV結果は、diamond_freeze_dev_model.pyで作った
    「開発コホート100%学習・凍結モデル」による外部検証AUC(0.681, 24hホライズン)
    を置き換えるものではない。外部検証は引き続き100%学習モデルの結果を用いる
    （外部コホートn=177と限られているため学習側の検出力を優先する）。
    本スクリプトの役割は、従来「in-sample train AUCのみ」で参考値扱いだった
    「モデル構築コホートでの精度」を、リークのない形で正式な内部検証結果として
    報告できるようにすることである。
    最終的な結果の見せ方は「内部CV(本スクリプト) → 外部検証(0.681)」の2段階。

  重要な設計（リーク防止）:
    各foldで、MinMaxScalerのfit・パディング長(max_rows)の決定は
    「そのfoldの学習側データのみ」から行う。held-out側には学習側で
    決めたscaler/max_rowsをそのまま適用する（外部検証と同じ考え方）。

使い方（例、主解析=24hホライズン、5-fold×1repeat）:
  python diamond_dev_cohort_cv.py \
      --input_dir ~/Documents/DIAMOND/DIAMOND/Data-fix \
      --output_dir ~/Documents/DIAMOND/DIAMOND/cv_24h \
      --horizon_hours 24 \
      --n_folds 5 --n_repeats 1

  安定性をさらに高めたい場合（fold分割の偶然性の影響をさらに減らす）:
  --n_repeats 3 （異なるfold分割を3回繰り返し、repeat間のばらつきも報告する）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam


LABEL_POSITIVE_DIR = "totalleak"
LABEL_NEGATIVE_DIR = "totalnoleak"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified k-fold CV for internal validation of the dev-cohort model "
        "(diamond_freeze_dev_model.py と同一パイプライン)."
    )
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--horizon_hours", type=int, default=24)
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_repeats", type=int, default=1, help="fold分割を複数seedで繰り返し安定性を高める(推奨2-3)")
    p.add_argument("--bootstrap_n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


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


def try_read_table(file_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]
    seps = [None, ",", "\t", ";", r"\s+"]
    for enc in encodings:
        for sep in seps:
            try:
                if sep is None:
                    df = pd.read_csv(file_path, encoding=enc, sep=None, engine="python")
                    sep_desc = "auto"
                else:
                    df = pd.read_csv(file_path, encoding=enc, sep=sep, engine="python")
                    sep_desc = repr(sep)
                if df is not None and len(df.columns) >= 2 and len(df) >= 1:
                    return df, enc, sep_desc
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
        return None, "no valid rows after cleaning"
    return out, "ok"


def extract_horizon_series(df: pd.DataFrame, horizon_hours: int) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """diamond_freeze_dev_model.pyと同一ロジック: horizon未到達症例は除外。
    ⚠️ この除外設計自体が選択バイアスの原因になりうる点は
    study-protocol-v1.md 4節Limitationsに既述（本スクリプトでは変更しない）。"""
    start_time = df["Date"].min()
    end_required = start_time + pd.Timedelta(hours=horizon_hours)
    end_time = df["Date"].max()
    if end_time < end_required:
        return None, {"reason": "drain_removed_before_horizon"}
    sub = df[df["Date"] <= end_required].copy()
    if sub.empty:
        return None, {"reason": "empty_horizon_slice"}
    arr = sub[["Air leak"]].to_numpy(dtype=np.float32)
    return arr, {"reason": "ok"}


def pad_or_truncate(item: np.ndarray, target_len: int) -> np.ndarray:
    if item.shape[0] < target_len:
        return np.pad(item, ((0, target_len - item.shape[0]), (0, 0)), "constant")
    return item[:target_len]


def create_cnn_model(input_shape: Tuple[int, int], learning_rate: float) -> tf.keras.Model:
    """diamond_freeze_dev_model.pyと同一アーキテクチャ"""
    model = Sequential()
    model.add(Conv1D(64, 3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_dev_cohort(input_dir: Path, recursive: bool, horizon_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = find_candidate_files(input_dir, recursive)
    records = []
    for file_path in sorted(files):
        label = detect_label_from_path(file_path)
        if label is None:
            continue
        raw_df, enc, sep = try_read_table(file_path)
        if raw_df is None:
            records.append({
                "file_path": str(file_path), "case_id": file_path.stem,
                "final_label": label, "tensor_status": "read_failed", "series": None,
            })
            continue
        df, status = clean_and_standardize(raw_df)
        if df is None:
            records.append({
                "file_path": str(file_path), "case_id": file_path.stem,
                "final_label": label, "tensor_status": status, "series": None,
            })
            continue
        series, info = extract_horizon_series(df, horizon_hours)
        records.append({
            "file_path": str(file_path), "case_id": file_path.stem, "final_label": int(label),
            "tensor_status": info["reason"], "series": series,
        })
    df_all = pd.DataFrame(records)
    included = df_all[df_all["tensor_status"] == "ok"].copy().reset_index(drop=True)
    excluded = df_all[df_all["tensor_status"] != "ok"].copy().reset_index(drop=True)
    return included, excluded


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
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
    return float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


def run_one_repeat(included: pd.DataFrame, args: argparse.Namespace, repeat_seed: int) -> pd.DataFrame:
    y = included["final_label"].to_numpy(dtype=int)
    series_list = included["series"].tolist()
    case_ids = included["case_id"].tolist()

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=repeat_seed)
    oof_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        set_seeds(repeat_seed * 1000 + fold_idx)  # fold毎に決定的、repeat間では変える

        train_series = [series_list[i] for i in train_idx]
        test_series = [series_list[i] for i in test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # リーク防止: max_rows・scalerは学習側(train_idx)のみから決定する
        max_rows = max(item.shape[0] for item in train_series)
        x_train_padded = np.array([pad_or_truncate(item, max_rows) for item in train_series], dtype=np.float32)
        scaler = MinMaxScaler()
        scaler.fit(x_train_padded.reshape(-1, 1))
        x_train_scaled = np.array([scaler.transform(item) for item in x_train_padded], dtype=np.float32)

        x_test_padded = np.array([pad_or_truncate(item, max_rows) for item in test_series], dtype=np.float32)
        x_test_scaled = np.array([scaler.transform(item) for item in x_test_padded], dtype=np.float32)

        model = create_cnn_model(input_shape=(x_train_scaled.shape[1], x_train_scaled.shape[2]), learning_rate=args.learning_rate)
        model.fit(x_train_scaled, y_train.astype(np.float32), epochs=args.epochs, batch_size=args.batch_size, verbose=0, shuffle=True)

        test_prob = model.predict(x_test_scaled, verbose=0).flatten()
        fold_auc = safe_auc(y_test, test_prob)
        print(
            f"[INFO] repeat_seed={repeat_seed} fold={fold_idx + 1}/{args.n_folds}: "
            f"n_test={len(test_idx)}(陽性{int(y_test.sum())}), fold_AUC={fold_auc:.3f}"
        )

        for local_i, global_i in enumerate(test_idx):
            oof_records.append({
                "repeat_seed": repeat_seed,
                "fold": fold_idx,
                "case_id": case_ids[global_i],
                "y_true": int(y[global_i]),
                "oof_prob": float(test_prob[local_i]),
                "fold_auc_reference": fold_auc,
            })

    return pd.DataFrame(oof_records)


def main() -> int:
    args = parse_args()
    input_dir = Path(os.path.expanduser(args.input_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    included, excluded = load_dev_cohort(input_dir, args.recursive, args.horizon_hours)
    print(f"[INFO] horizon={args.horizon_hours}h: n_included={len(included)}, n_excluded={len(excluded)}")
    if not excluded.empty:
        print(f"[INFO] excluded breakdown: {excluded['tensor_status'].value_counts().to_dict()}")
    print(f"[INFO] n_positive={int(included['final_label'].sum())}, n_negative={int((1 - included['final_label']).sum())}")
    if included.empty:
        raise RuntimeError("No cases reached the target horizon.")

    all_oof = []
    for r in range(args.n_repeats):
        repeat_seed = args.seed + r
        print(f"[INFO] ===== repeat {r + 1}/{args.n_repeats} (seed={repeat_seed}) =====")
        oof_df = run_one_repeat(included, args, repeat_seed)
        all_oof.append(oof_df)
    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv(output_dir / f"cv_oof_predictions_{args.horizon_hours}h.csv", index=False)

    # repeatごとのOOF AUC（全症例を1回ずつ含む）を算出し、repeat間の平均・ばらつきを見る
    per_repeat_summary = []
    for r in range(args.n_repeats):
        repeat_seed = args.seed + r
        sub = oof_all[oof_all["repeat_seed"] == repeat_seed]
        auc = safe_auc(sub["y_true"].to_numpy(), sub["oof_prob"].to_numpy())
        per_repeat_summary.append({
            "repeat_seed": repeat_seed, "oof_auc": auc,
            "n": len(sub), "n_positive": int(sub["y_true"].sum()),
        })
    per_repeat_df = pd.DataFrame(per_repeat_summary)
    per_repeat_df.to_csv(output_dir / f"cv_per_repeat_summary_{args.horizon_hours}h.csv", index=False)

    # メイン指標: 全repeat・全foldのOOF予測をプールしたAUC + bootstrap CI
    # (n_repeats=1ならこれがそのまま「k-fold CVのOOF AUC」＝本研究の内部検証AUC)
    pooled_auc = safe_auc(oof_all["y_true"].to_numpy(), oof_all["oof_prob"].to_numpy())
    ci_lo, ci_hi = bootstrap_auc_ci(oof_all["y_true"].to_numpy(), oof_all["oof_prob"].to_numpy(), args.bootstrap_n, args.seed)

    fold_aucs = oof_all.drop_duplicates(subset=["repeat_seed", "fold"])["fold_auc_reference"].to_numpy(dtype=float)

    summary = {
        "horizon_hours": args.horizon_hours,
        "n_folds": args.n_folds,
        "n_repeats": args.n_repeats,
        "n_included": len(included),
        "n_positive": int(included["final_label"].sum()),
        "pooled_oof_auc": pooled_auc,
        "pooled_oof_auc_ci_lo": ci_lo,
        "pooled_oof_auc_ci_hi": ci_hi,
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_sd": float(np.std(fold_aucs)),
        "fold_auc_min": float(np.min(fold_aucs)),
        "fold_auc_max": float(np.max(fold_aucs)),
        "per_repeat_oof_auc_mean": float(per_repeat_df["oof_auc"].mean()),
        "per_repeat_oof_auc_sd": float(per_repeat_df["oof_auc"].std()) if len(per_repeat_df) > 1 else float("nan"),
    }
    with open(output_dir / f"cv_summary_{args.horizon_hours}h.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[結果] {args.horizon_hours}h ホライズン, {args.n_folds}-fold CV × {args.n_repeats} repeat")
    print(f"[結果] Pooled OOF AUC(=本研究の内部交差検証AUCとして報告する数値) = {pooled_auc:.3f} (95%CI {ci_lo:.3f}-{ci_hi:.3f})")
    print(f"[結果] Fold毎AUCの平均±SD = {summary['fold_auc_mean']:.3f} ± {summary['fold_auc_sd']:.3f} (range {summary['fold_auc_min']:.3f}-{summary['fold_auc_max']:.3f})")
    if args.n_repeats > 1:
        print(f"[結果] Repeat間のPooled OOF AUC平均±SD = {summary['per_repeat_oof_auc_mean']:.3f} ± {summary['per_repeat_oof_auc_sd']:.3f}")
    print(f"[OK] 保存: {output_dir / f'cv_summary_{args.horizon_hours}h.json'}")
    print(
        "[NEXT] このPooled OOF AUCを、既存の外部検証AUC(0.681, 24h, 開発コホート100%学習の凍結モデル)"
        "と並べて『内部CV → 外部検証』の2段階（＋外部の36h探索的解析0.720）として報告してください。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
