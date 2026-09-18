#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_inspect_redcap_key_columns.py

目的:
  diamond_build_true_pal_labels.py をスクリーニングログ除外後(213症例)で実行したところ、
  「判定可能=0」（全例でsurgedatがNaN扱い）という結果になった(2026-08-16)。
  また「侵襲的処置の内訳」も想定(2例/5例/1例)から大きく乖離(5例/69例/1例)している。

  原因切り分けのため、diamond_build_true_pal_labels.py / diamond_table1_external.py が
  使う主要列について、①どのイベントに実際にデータが存在するか(非欠損件数)
  ②日付列(surgedat/alendat/chandat)がpd.to_datetimeでパースできているか、
  を集計値のみで報告する。患者を特定できる値・日付そのものの値は一切出力しない
  （件数・パターン分類のみ）。

  仮説A: surgedatは除外した「スクリーニングログ」イベント側にのみ存在し、
        残り7イベント側には無い（→そのイベントは実は基本情報を持つ主要イベントで、
        自動除外ロジックの見直しが必要）。
  仮説B: surgedatは残り7イベント側にも存在するが、日付書式(全角/和暦等)が
        pd.to_datetimeでパースできず、全例NaTになっている（→パース処理の修正が必要）。

使い方:
  python diamond_inspect_redcap_key_columns.py \
      --redcap_csv "/Volumes/Extreme Pro/Red Cap data/ThopazRCT2_raw.csv"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--redcap_csv", required=True)
    p.add_argument("--subjid_col", default="subjid")
    p.add_argument("--event_col", default="redcap_event_name")
    p.add_argument("--encoding", default="utf-8")
    return p.parse_args()


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    if wanted in df.columns:
        return wanted
    norm_wanted = normalize_colname(wanted)
    if not norm_wanted:
        return None
    for c in df.columns:
        if normalize_colname(c) == norm_wanted:
            return c
    return None


def read_redcap_csv(path: Path, encoding: str) -> pd.DataFrame:
    tried = [encoding, "utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_err = None
    for enc in dict.fromkeys(tried):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"{path} を読めるエンコーディングが見つかりませんでした: {last_err}")


# diamond_table1_external.py / diamond_build_true_pal_labels.py が使う主要列
TARGET_COLUMNS = [
    "subjid",
    "surgedat", "alendat", "chandat",
    "modifi___3", "modifi___4", "modifi___5",
    "rand", "facilities",
    "age2", "sex2", "surgtime", "bloodloss",
    "adjust_cig_result", "adjust_3", "primdise", "surg", "appro",
    "hight", "height", "weight",
]

DATE_COLUMNS = ["surgedat", "alendat", "chandat"]


def classify_date_pattern(s: str) -> str:
    """日付文字列の値そのものは出さず、構造パターンだけ分類する。"""
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", s):
        return "ISO風(YYYY-MM-DD)"
    if re.match(r"^\d{4}/\d{1,2}/\d{1,2}", s):
        return "スラッシュ区切り(YYYY/MM/DD)"
    if re.search(r"[令和平成昭和大正]", s):
        return "和暦表記を含む"
    if re.search(r"[０-９]", s):  # 全角数字
        return "全角数字を含む"
    if re.match(r"^\s*$", s):
        return "空白のみ"
    return "その他パターン"


def main() -> int:
    args = parse_args()
    path = Path(args.redcap_csv).expanduser()
    df = read_redcap_csv(path, args.encoding)
    print(f"[INFO] 生データ読込: {len(df)}行, {len(df.columns)}列")

    subjid_col = resolve_col(df, args.subjid_col)
    event_col = resolve_col(df, args.event_col)
    if subjid_col is None or event_col is None:
        raise RuntimeError(f"subjid列またはevent列が見つかりません（subjid={subjid_col}, event={event_col}）")

    event_counts = df[event_col].value_counts()
    ranked_events = event_counts.index.tolist()
    labels = {ev: f"E{i+1}" for i, ev in enumerate(ranked_events)}

    print(f"[INFO] イベント数={len(ranked_events)}（行数の多い順にE1..E{len(ranked_events)}とラベル付け）")

    resolved_map = {}
    for wanted in TARGET_COLUMNS:
        resolved = resolve_col(df, wanted)
        resolved_map[wanted] = resolved
        if resolved is None:
            print(f"[INFO] 列 '{wanted}': データ全体に見つかりません")
            continue

        per_event_nonnull = {}
        for ev in ranked_events:
            sub = df.loc[df[event_col] == ev, resolved]
            n_nonnull = int(sub.notna().sum())
            if n_nonnull > 0:
                per_event_nonnull[labels[ev]] = n_nonnull

        if per_event_nonnull:
            detail = ", ".join(f"{lbl}={n}件" for lbl, n in per_event_nonnull.items())
            print(f"[INFO] 列 '{wanted}' -> '{resolved}': 非欠損はイベント[{detail}]")
        else:
            print(f"[WARN] 列 '{wanted}' -> '{resolved}': 全イベントで非欠損値が0件")

    print()
    print("[INFO] 日付列のパース可否チェック（値は出さず件数・パターンのみ）:")
    for wanted in DATE_COLUMNS:
        resolved = resolved_map.get(wanted)
        if resolved is None:
            continue
        raw = df[resolved].dropna().astype(str)
        raw = raw[raw.str.strip() != ""]
        n_raw_nonempty = len(raw)
        parsed = pd.to_datetime(raw, errors="coerce")
        n_parsed_ok = int(parsed.notna().sum())
        n_parse_failed = n_raw_nonempty - n_parsed_ok
        print(
            f"    '{wanted}': 生の非空文字列={n_raw_nonempty}件, "
            f"to_datetimeでパース成功={n_parsed_ok}件, 失敗={n_parse_failed}件"
        )
        if n_parse_failed > 0:
            failed_values = raw[parsed.isna()]
            pattern_counts = failed_values.map(classify_date_pattern).value_counts()
            for pattern, cnt in pattern_counts.items():
                print(f"        パース失敗の内訳: {pattern} = {cnt}件")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
