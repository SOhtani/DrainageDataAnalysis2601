#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_freeze_dev_model.py

目的:
  diamond_1dcnn_final_label_legacy_style.py と同一のデータ読込・ラベル付与・
  前処理ロジックを踏襲しつつ、以下2点を変更した「外部検証用の凍結モデル」を作る。

  変更点1（凍結）:
    交差検証（3fold/5fold/loo）ではなく、開発コホート全体で1本のモデルを学習し、
    重み・scaler（MinMaxScalerのmin/max）・入力長を保存する。
    外部データにはこの保存済みパラメータのみを適用し、再学習・再fitは行わない。

  変更点2（選択バイアス対策・任意）:
    legacy_style.py は「horizon到達前にドレーン抜去された症例を除外」する設計であり、
    これが48h/72hでの選択バイアス（陽性率の激増）の原因と判明している（2026-08-04確認）。
    本スクリプトでは --resample_fixed_grid を指定すると、
    固定グリッド（resample_minutes間隔でhorizon_hours分）にリサンプリングし、
    記録が短い症例は「観測なし=0」として扱うことで、除外せずに含める設計に変更できる。
    デフォルトはlegacy_style.py同様の除外方式（--resample_fixed_gridなし）。
    今日はまず legacy_style と同じ除外方式で凍結モデルを作り、時間があれば
    固定グリッド版も追加で作る。

使い方:
  python diamond_freeze_dev_model.py \
      --input_dir ~/Documents/DIAMOND/DIAMOND/Data-fix \
      --output_dir ~/Documents/DIAMOND/DIAMOND/frozen_24h \
      --horizon_hours 24
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
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam


LABEL_POSITIVE_DIR = "totalleak"
LABEL_NEGATIVE_DIR = "totalnoleak"
TARGET_HOUR = 9
WINDOW_HOURS = 1
RULE_THRESHOLD = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a single final model (legacy-style pipeline) on the full development cohort.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--horizon_hours", type=int, choices=[24, 48, 72], required=True)
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
    """legacy_style.pyと同一ロジック: horizon未到達（drain_removed_before_horizon）は除外。
    ⚠️ これが48h/72hでの選択バイアス（陽性率激増）の原因と確認済み。"""
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


def pad_data(data: List[np.ndarray], max_rows: int) -> np.ndarray:
    padded_data = []
    for item in data:
        if item.shape[0] < max_rows:
            item = np.pad(item, ((0, max_rows - item.shape[0]), (0, 0)), "constant")
        else:
            item = item[:max_rows]
        padded_data.append(item)
    return np.array(padded_data, dtype=np.float32)


def create_cnn_model(input_shape: Tuple[int, int], learning_rate: float) -> tf.keras.Model:
    """legacy_style.pyと同一アーキテクチャ"""
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


def main() -> int:
    args = parse_args()
    set_seeds(args.seed)

    input_dir = Path(os.path.expanduser(args.input_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_candidate_files(input_dir, args.recursive)
    records = []
    failures = []

    for file_path in sorted(files):
        label = detect_label_from_path(file_path)
        if label is None:
            continue
        raw_df, enc, sep = try_read_table(file_path)
        if raw_df is None:
            failures.append((str(file_path), "read_failed"))
            continue
        df, status = clean_and_standardize(raw_df)
        if df is None:
            failures.append((str(file_path), status))
            continue
        series, info = extract_horizon_series(df, args.horizon_hours)
        records.append({
            "file_path": str(file_path),
            "case_id": file_path.stem,
            "final_label": int(label),
            "tensor_status": info["reason"],
            "series": series,
        })

    df_all = pd.DataFrame(records)
    included = df_all[df_all["tensor_status"] == "ok"].copy().reset_index(drop=True)
    excluded = df_all[df_all["tensor_status"] != "ok"].copy().reset_index(drop=True)

    print(f"[INFO] n_total={len(df_all)}, n_included={len(included)}, n_excluded={len(excluded)}")
    if not excluded.empty:
        print(f"[INFO] excluded breakdown: {excluded['tensor_status'].value_counts().to_dict()}")
    print(f"[INFO] n_positive={int(included['final_label'].sum())}, n_negative={int((1-included['final_label']).sum())}")

    if included.empty:
        raise RuntimeError("No cases reached the target horizon.")

    y = included["final_label"].to_numpy(dtype=int)
    series_list = included["series"].tolist()

    # 開発コホート全体でscalerをfit・全体をpadding（外部検証でも同じmax_rowsに合わせる）
    max_rows = max(item.shape[0] for item in series_list)
    x_padded = pad_data(series_list, max_rows)

    scaler = MinMaxScaler()
    scaler.fit(x_padded.reshape(-1, 1))
    x_scaled = np.array([scaler.transform(item) for item in x_padded], dtype=np.float32)

    model = create_cnn_model(input_shape=(x_scaled.shape[1], x_scaled.shape[2]), learning_rate=args.learning_rate)
    model.fit(x_scaled, y.astype(np.float32), epochs=args.epochs, batch_size=args.batch_size, verbose=1, shuffle=True)

    # 学習データ上の性能（参考値。overfitの可能性が高く、外部検証の主結果には使わない）
    train_prob = model.predict(x_scaled, verbose=0).flatten()
    train_auc = roc_auc_score(y, train_prob) if len(np.unique(y)) > 1 else float("nan")

    model.save(output_dir / f"frozen_model_{args.horizon_hours}h.keras")
    with open(output_dir / f"scaler_{args.horizon_hours}h.json", "w") as f:
        json.dump({
            "data_min": float(scaler.data_min_[0]),
            "data_max": float(scaler.data_max_[0]),
            "max_rows": int(max_rows),
            "horizon_hours": args.horizon_hours,
            "n_train": int(len(y)),
            "n_positive": int(y.sum()),
            "n_excluded_drain_removed_before_horizon": int((excluded["tensor_status"] == "drain_removed_before_horizon").sum()) if not excluded.empty else 0,
            "train_set_auc_reference_only": train_auc,
        }, f, indent=2)

    print(f"[OK] Frozen model saved: {output_dir / f'frozen_model_{args.horizon_hours}h.keras'}")
    print(f"[OK] Scaler saved: {output_dir / f'scaler_{args.horizon_hours}h.json'}")
    print(f"[INFO] max_rows(入力長, 外部検証でも同じ長さに揃える必要あり)={max_rows}")
    print(f"[INFO] train_set_auc(参考値、overfit注意)={train_auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
