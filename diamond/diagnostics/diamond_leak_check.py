#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_leak_check.py

Astra提案の情報リーク検査（PLAN_C_RECONCILIATION.md参照）:
「最初の24時間を同一に保ったまま、24時間以降のデータを削除・変更・追加しても、
24h用入力と予測スコアが変わらないことを確認する」

合成データで以下を検証する:
  1. extract_horizon_series(df, 24h) の出力が、24h以降のデータを変えても不変か
  2. モデルの前処理〜予測(ai_probability)が、24h以降のデータを変えても不変か
  3. 意図的に24h未到達にしたデータが、正しく "drain_removed_before_horizon" として
     除外されるか（＝「対象に含めるか」の判定は将来情報を見てよいが、「スコア自体」は
     見てはいけない、という区別を確認する）

使い方:
  python3 diamond_leak_check.py --model_dir ~/Documents/DIAMOND/DIAMOND/frozen_24h --horizon_hours 24
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense


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


def pad_or_truncate(item: np.ndarray, target_len: int) -> np.ndarray:
    if item.shape[0] < target_len:
        return np.pad(item, ((0, target_len - item.shape[0]), (0, 0)), "constant")
    return item[:target_len]


def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model


def make_series(start, hours, interval_min, value_fn):
    """value_fn(t_hours) -> air leak value at that hour offset."""
    times = pd.date_range(start, periods=int(hours * 60 / interval_min), freq=f"{interval_min}min")
    hours_offset = np.array([(t - start).total_seconds() / 3600.0 for t in times])
    values = value_fn(hours_offset)
    return pd.DataFrame({"Date": times, "Air leak": values})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--horizon_hours", type=int, default=24)
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    scaler_path = os.path.join(model_dir, f"scaler_{args.horizon_hours}h.json")
    weights_path = os.path.join(model_dir, f"frozen_weights_{args.horizon_hours}h.h5")
    if not os.path.exists(scaler_path) or not os.path.exists(weights_path):
        sys.exit(f"ERROR: モデルファイルが見つかりません: {scaler_path} / {weights_path}")

    with open(scaler_path) as f:
        scaler_info = json.load(f)
    data_min, data_max, max_rows = scaler_info["data_min"], scaler_info["data_max"], scaler_info["max_rows"]
    print(f"[INFO] scaler: data_min={data_min}, data_max={data_max}, max_rows={max_rows}")

    model = create_cnn_model(input_shape=(max_rows, 1))
    model.load_weights(weights_path)
    print(f"[INFO] loaded weights from {weights_path}\n")

    start = pd.Timestamp("2026-01-01 00:00:00")

    def base_pattern(h):
        # 0-24hはランダムに見える(が再現可能な)パターン。実データの気漏に近いスケール(0-300ml/min)。
        rng = np.random.default_rng(42)
        return 150 + 100 * np.sin(h / 3.0) + rng.normal(0, 10, size=h.shape)

    def score(df, label):
        series, status = extract_horizon_series(df, args.horizon_hours)
        if series is None:
            print(f"  [{label}] status={status} (除外)")
            return status, None, None
        x = pad_or_truncate(series, max_rows)
        x_scaled = (x - data_min) / (data_max - data_min if data_max != data_min else 1.0)
        prob = model.predict(x_scaled[np.newaxis, ...], verbose=0)[0, 0]
        print(f"  [{label}] status=ok, n_rows(24h切り出し後)={series.shape[0]}, ai_probability={prob:.10f}")
        return status, series, prob

    print("=== テスト1: 24h以降のデータを変えても、24h入力とスコアが不変か ===")
    # バージョンA: 0-30h、24h以降は「変動パターン1」
    dfA = make_series(start, 30, 10, lambda h: np.where(h <= 24, base_pattern(h), 500 + 50 * h))
    # バージョンB: 0-24hはAと全く同じ、24h以降は「全く違うパターン」(ゼロ、または巨大な値)
    dfB = make_series(start, 30, 10, lambda h: np.where(h <= 24, base_pattern(h), 0.0))
    # バージョンC: 0-24hはAと全く同じ、24h以降はさらに違う(spiky)パターン、かつ48hまで延長
    dfC = make_series(start, 48, 10, lambda h: np.where(h <= 24, base_pattern(h), 9999.0))

    statusA, seriesA, probA = score(dfA, "A: 24h以降=線形増加パターン, 30hまで")
    statusB, seriesB, probB = score(dfB, "B: 24h以降=ゼロ,          30hまで")
    statusC, seriesC, probC = score(dfC, "C: 24h以降=9999固定,      48hまで")

    ok1 = np.array_equal(seriesA, seriesB) and np.array_equal(seriesA, seriesC)
    ok2 = (probA == probB == probC)
    print(f"\n  -> 24h切り出し配列は3バージョンで完全一致か: {ok1}")
    print(f"  -> ai_probabilityは3バージョンで完全一致か:   {ok2} (A={probA:.10f}, B={probB:.10f}, C={probC:.10f})")

    print("\n=== テスト2: 意図的に24h未到達にしたデータが正しく除外されるか ===")
    df_short = make_series(start, 20, 10, base_pattern)  # 20hで打ち切り、24hに届かない
    status_short, _, _ = score(df_short, "D: 20hで打ち切り(24h未到達)")
    ok3 = (status_short == "drain_removed_before_horizon")
    print(f"\n  -> 正しく'drain_removed_before_horizon'として除外されたか: {ok3}")

    print("\n" + "=" * 60)
    if ok1 and ok2 and ok3:
        print("[PASS] 情報リークは検出されませんでした。")
        print("       ・24h以降のデータをどう変えても24h入力/予測スコアは不変")
        print("       ・24h未到達の判定(除外基準)は正しく機能している")
        return 0
    else:
        print("[FAIL] 情報リークの疑いがあります。上記の不一致箇所を確認してください。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
