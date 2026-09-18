#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_diagnose_label_vs_scored_cohort.py

目的:
  「患者背景表(Table1)・PAL陽性率51.0%(98/192)を出したコホート/ラベル」と、
  「AI予測AUCを出したコホート/ラベル(diamond_score_external_predictions.py)」が
  本当に同じ集団・同じtrue_pal定義を使っているかを直接突合して確認する。

  結論を先に言うと、コードを読む限り両者は同じ true_pal_labels.csv
  （diamond_build_true_pal_labels.pyの1本の出力ファイル）を参照しており、
  ラベルの定義・計算方法自体は完全に同一。ただし各horizonのAI予測ファイル
  (external_predictions_{horizon}.csv)が192例全員をカバーしているとは限らず、
  「予測が無い/IDが突合できない症例」がいれば、AUC評価に使われるn(n_scored)は
  192例より少なくなりうる。本スクリプトはその差分を実データで確認する。

  出力: 各horizonについて
    - labels_csv側のコホート数(=Table1/PAL率51.0%の母数)と真陽性数
    - predictions側で突合・スコアリングできた数(n_scored)と真陽性数
    - labels_csv側にいるがpredictions側で突合できなかった症例数・一覧(CSV)
    - 両者が一致する部分集合だけで見たPAL陽性率(母数の違いによる見かけの
      陽性率変化を切り分けるため)

使い方（例）:
  python diamond_diagnose_label_vs_scored_cohort.py \
      --labels_csv ~/Documents/DIAMOND/DIAMOND/split_models_external/true_pal_labels.csv \
      --predictions_dir ~/Documents/DIAMOND/DIAMOND/split_models_external \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/diagnostics \
      --horizons full 12 18 24
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True, help="diamond_build_true_pal_labels.py の出力")
    p.add_argument("--predictions_dir", required=True, help="external_predictions_{horizon}.csv があるディレクトリ")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--horizons", nargs="+", default=["full", "12", "18", "24"])
    return p.parse_args()


def normalize_case_id(raw_id: object) -> Tuple[Optional[str], Optional[str], str]:
    """diamond_build_true_pal_labels.py / diamond_score_external_predictions.py と同一ロジック。"""
    s = str(raw_id).strip()
    cleaned = re.sub(r"[（(].*?[）)]", "", s)
    cleaned = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", "", cleaned).strip()
    m = re.match(r"^(\d+)\s*[-_]\s*(\d+)\s*([a-zA-Z]?)", cleaned)
    if not m:
        return None, None, cleaned
    facility = str(int(m.group(1)))
    case_no = str(int(m.group(2)))
    suffix = m.group(3).lower()
    loose_key = f"{facility}-{case_no}"
    strict_key = f"{loose_key}{suffix}"
    return strict_key, loose_key, cleaned


def predictions_filename(horizon: str) -> str:
    return f"external_predictions_{horizon}.csv" if horizon == "full" else f"external_predictions_{horizon}h.csv"


def add_case_id_keys(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    strict, loose, cleaned = [], [], []
    for v in df[id_col]:
        sk, lk, cl = normalize_case_id(v)
        strict.append(sk)
        loose.append(lk)
        cleaned.append(cl)
    df = df.copy()
    df["_strict_key"] = strict
    df["_loose_key"] = loose
    df["_cleaned_id"] = cleaned
    return df


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels_csv).expanduser()
    predictions_dir = Path(args.predictions_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_path)
    if "case_id_strict_key" not in labels.columns:
        raise RuntimeError("labels_csvに case_id_strict_key 列がありません。true_pal_labels.csvを指定してください。")

    n_cohort = len(labels)
    n_determined = int(labels["true_pal"].notna().sum())
    n_positive = int((labels["true_pal"] == 1).sum())
    pos_rate = n_positive / n_determined if n_determined else float("nan")
    print("===== labels_csv側(=Table1・PAL陽性率の母数) =====")
    print(f"[INFO] コホート総数(rand非欠損+dsdecod=1に絞り込み済み): {n_cohort}")
    print(f"[INFO] true_pal判定可能: {n_determined}, 陽性: {n_positive} ({pos_rate:.1%})")
    print(
        "[NOTE] このn/陽性率が、diamond_compare_cohorts_stats.pyが報告した"
        "「PALあり(%) 98/192 (51.0%)」の直接の出所です。\n"
    )

    summary_rows = []
    for horizon in args.horizons:
        pred_path = predictions_dir / predictions_filename(horizon)
        if not pred_path.exists():
            print(f"[WARN] horizon={horizon}: {pred_path} が見つかりません（スキップ）")
            continue
        preds = pd.read_csv(pred_path)
        preds = add_case_id_keys(preds, "case_id")

        lab = labels.rename(columns={
            "case_id_strict_key": "_strict_key",
            "case_id_loose_key": "_loose_key",
        })

        lab_strict = lab.dropna(subset=["_strict_key"]).drop_duplicates("_strict_key", keep=False)
        merged_strict = preds.merge(lab_strict, on="_strict_key", suffixes=("", "_label"), how="inner")
        matched_ids = set(merged_strict["case_id"])

        remaining_preds = preds[~preds["case_id"].isin(matched_ids)]
        lab_loose = lab.dropna(subset=["_loose_key"])
        loose_counts = lab_loose["_loose_key"].value_counts()
        lab_loose_unique = lab_loose[lab_loose["_loose_key"].isin(loose_counts[loose_counts == 1].index)]
        merged_loose = remaining_preds.merge(lab_loose_unique, on="_loose_key", suffixes=("", "_label"), how="inner")

        merged = pd.concat([merged_strict, merged_loose], ignore_index=True, sort=False)

        merged_scored = merged.dropna(subset=["true_pal"])
        merged_scored = merged_scored[merged_scored["true_pal"].isin([0, 1])]
        n_scored = len(merged_scored)
        n_scored_pos = int(merged_scored["true_pal"].astype(int).sum())
        scored_pos_rate = n_scored_pos / n_scored if n_scored else float("nan")

        # labels_csv側のsubjidのうち、このhorizonの予測に突合できなかった症例
        matched_strict_keys = set(merged["_strict_key"].dropna()) if "_strict_key" in merged.columns else set()
        labels_not_matched = lab[~lab["_strict_key"].isin(matched_strict_keys)]

        print(f"===== horizon={horizon} =====")
        print(f"[INFO] 予測ファイル症例数: {len(preds)} / labels突合成功: {len(merged)} / true_pal判定可能でスコアリング対象: {n_scored}")
        print(f"[INFO] スコアリング対象内の陽性率: {n_scored_pos}/{n_scored} ({scored_pos_rate:.1%})  "
              f"※全体母数({n_cohort}例)での陽性率{pos_rate:.1%}と比較してください")
        print(f"[INFO] labels_csv側にいるがこのhorizonの予測と突合できなかった症例数: {len(labels_not_matched)}")

        if len(labels_not_matched):
            miss_path = output_dir / f"labels_not_matched_to_predictions_{horizon}.csv"
            labels_not_matched[["subjid", "true_pal", "case_id_cleaned"]].to_csv(miss_path, index=False)
            print(f"[OK] 突合できなかった症例一覧を保存: {miss_path}（手動確認用）")

        summary_rows.append({
            "horizon": horizon,
            "cohort_n_labels_csv": n_cohort,
            "cohort_n_determined_labels_csv": n_determined,
            "cohort_n_positive_labels_csv": n_positive,
            "cohort_pos_rate_labels_csv": pos_rate,
            "predictions_n": len(preds),
            "matched_n": len(merged),
            "scored_n": n_scored,
            "scored_n_positive": n_scored_pos,
            "scored_pos_rate": scored_pos_rate,
            "n_labels_not_matched_to_predictions": len(labels_not_matched),
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "label_vs_scored_cohort_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[OK] 全horizon横断サマリを保存: {summary_path}")
        print(summary_df.to_string(index=False))
    else:
        print("[ERROR] 1件もhorizonを処理できませんでした。")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
