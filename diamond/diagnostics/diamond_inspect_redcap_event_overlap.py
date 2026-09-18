#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_inspect_redcap_event_overlap.py

目的:
  diamond_inspect_redcap_subjid.py (2026-08-16実行) で判明した手がかりを追う:
  - REDCap生データは8イベント構成（既知のコードブック記載=17フォーム・8イベントと一致）。
  - うち1イベントの行数(2354)が、生subjidの総ユニーク数(2354)と完全に一致していた。
  - 残り7イベントの行数(204〜381)は、想定していた研究コホート規模(申告182例、
    親試験目標最大199例)により近い。

  仮説: 最大のイベントは「スクリーニングログ」等、除外例も含む
  より大きな母集団の登録ログであり、残り7イベントが実際に
  ランダム化・追跡された研究コホート（の各時点）である可能性が高い。

  本スクリプトはこの仮説を集計値のみで検証する:
    - イベントごとのsubjidユニーク数（行数計上順ではなく大きい順にE1..E8とラベル付け、
      実際のイベント名は表示しない）
    - 最大イベント(E1)を除いた残り7イベントの和集合サイズ
      → これが182-199に近ければ、真の研究コホート候補として有力
    - その和集合のうち、E1にも含まれる割合（スクリーニングログが上位互換か確認）
    - 残り7イベント同士の重なり具合（全7イベントに共通して出現する症例数、
      1つ以上に出現する症例数=和集合、の対比）
    - 施設(facilities列があれば)ごとの、E1除外後の和集合の症例数分布
      （既知の「18施設・申告合計182例」との整合確認用）

  患者を特定できる値(生のsubjid文字列)は一切出力しない。

使い方:
  python diamond_inspect_redcap_event_overlap.py \
      --redcap_csv "/Volumes/Extreme Pro/Red Cap data/ThopazRCT2_raw.csv"
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--redcap_csv", required=True)
    p.add_argument("--subjid_col", default="subjid")
    p.add_argument("--event_col", default="redcap_event_name")
    p.add_argument("--facilities_col", default="facilities")
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


def main() -> int:
    args = parse_args()
    path = Path(args.redcap_csv).expanduser()
    df = read_redcap_csv(path, args.encoding)
    print(f"[INFO] 生データ読込: {len(df)}行, {len(df.columns)}列")

    subjid_col = resolve_col(df, args.subjid_col)
    event_col = resolve_col(df, args.event_col)
    if subjid_col is None or event_col is None:
        raise RuntimeError(f"subjid列またはevent列が見つかりません（subjid={subjid_col}, event={event_col}）")

    df = df[[subjid_col, event_col] + ([resolve_col(df, args.facilities_col)] if resolve_col(df, args.facilities_col) else [])].copy()
    df[subjid_col] = df[subjid_col].astype(str).str.strip()

    # イベントごとのsubjid集合を作り、行数が多い順にE1..E8とラベル付け（実イベント名は非表示）
    event_counts = df[event_col].value_counts()
    ranked_events = event_counts.index.tolist()  # 大きい順
    labels = {ev: f"E{i+1}" for i, ev in enumerate(ranked_events)}
    id_sets = {labels[ev]: set(df.loc[df[event_col] == ev, subjid_col]) for ev in ranked_events}

    print(f"[INFO] イベント数={len(ranked_events)}")
    for ev in ranked_events:
        lbl = labels[ev]
        print(f"    {lbl}: 行数={event_counts[ev]}, ユニークsubjid数={len(id_sets[lbl])}")

    all_union = set().union(*id_sets.values())
    print(f"[INFO] 全イベント合算の和集合(=生ユニーク総数の再確認)={len(all_union)}")

    if len(ranked_events) < 2:
        print("[WARN] イベントが1種類以下のため、以降の重なり分析はスキップします")
        return 0

    largest_label = labels[ranked_events[0]]
    other_labels = [labels[ev] for ev in ranked_events[1:]]

    other_union = set().union(*(id_sets[l] for l in other_labels))
    print(f"[INFO] {largest_label}(最大イベント)を除いた残り{len(other_labels)}イベントの和集合={len(other_union)}")

    overlap_with_largest = len(other_union & id_sets[largest_label])
    print(
        f"[INFO] その和集合のうち{largest_label}にも含まれる症例数="
        f"{overlap_with_largest}/{len(other_union)} ({overlap_with_largest/len(other_union):.1%})"
        f"（100%に近ければ「{largest_label}=残り7イベントを包含する、より大きな母集団の登録ログ」という仮説を支持）"
    )

    intersection_all_others = set.intersection(*(id_sets[l] for l in other_labels))
    print(
        f"[INFO] 残り{len(other_labels)}イベント全てに共通して出現する症例数={len(intersection_all_others)} "
        f"（全時点データが揃っている「完全追跡」症例の目安）"
    )

    # 残り7イベント同士のペアワイズ重なり（何%が共通か、行列でなく要約統計のみ）
    pair_overlaps = []
    for a, b in combinations(other_labels, 2):
        inter = len(id_sets[a] & id_sets[b])
        union = len(id_sets[a] | id_sets[b])
        pair_overlaps.append(inter / union if union else 0.0)
    if pair_overlaps:
        s = pd.Series(pair_overlaps)
        print(
            f"[INFO] 残り{len(other_labels)}イベント間のペアワイズJaccard重なり率: "
            f"min={s.min():.1%}, median={s.median():.1%}, max={s.max():.1%}"
            f"（全体的に高ければ、これら7イベントはほぼ同一の患者集団を指していると解釈できる）"
        )

    facilities_col = resolve_col(df, args.facilities_col)
    if facilities_col:
        fac_map = df.drop_duplicates(subset=[subjid_col]).set_index(subjid_col)[facilities_col]
        fac_for_other_union = fac_map.reindex(list(other_union)).dropna()
        print(f"[INFO] {largest_label}除外後コホート({len(other_union)}例)の施設別内訳(件数のみ):")
        vc = fac_for_other_union.value_counts().sort_index()
        for fac, cnt in vc.items():
            print(f"    施設コード{fac}: {cnt}例")
        print(f"[INFO] 施設数={fac_for_other_union.nunique()}（既知の18施設と比較してください）")
    else:
        print(f"[INFO] facilities列（--facilities_col='{args.facilities_col}'）が見つからず施設別内訳はスキップ")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
