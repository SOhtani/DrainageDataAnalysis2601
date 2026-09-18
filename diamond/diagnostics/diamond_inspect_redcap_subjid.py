#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_inspect_redcap_subjid.py

目的:
  diamond_table1_external.py を実行したところ、想定症例数(~182-199例)に対して
  対象症例数(REDCap全体)=2354 という大きく乖離した値が出た(2026-08-16)。
  原因を切り分けるための診断専用スクリプト。

  仮説（2026-08-09の外部検証作業で実際に見つかった問題と同種）:
    REDCapの縦持ちexportでは、'subjid'欄がイベント(フォーム)ごとに
    別の担当者が手入力している可能性があり、同一患者でも表記ゆれ
    (先頭ゼロ・前後の空白・接尾辞アルファベット・括弧注記等)によって
    異なる文字列として記録され、groupby(subjid)が同一患者を複数の
    別グループに分割してしまっている可能性が高い。

  本スクリプトは患者を特定できる値(生のsubjid文字列そのもの)は一切出力せず、
  以下の構造的な集計値のみを報告する:
    - 生データの総行数・総列数
    - 解決された subjid 列名（別の列に誤ってマッチしていないかの確認）
    - 生の subjid のユニーク数（=今回の2354の再現確認）
    - 表記ゆれ正規化後（先頭ゼロ除去・空白/括弧注記/丸数字除去、
      diamond_build_true_pal_labels.py の normalize_case_id と同一ロジック）の
      ユニーク数（loose_key・strict_key 両方）
      → これが 182-199 に近づけば、表記ゆれが原因と確定できる。
    - パターンに全くマッチしない(パース不能)行数
    - redcap_event_name 等のイベント列があれば、イベント名ごとの行数
      （イベント名は個人情報ではないため安全に表示可能）
    - 1患者(loose_key)あたりの行数の分布（min/max/中央値）

使い方:
  python diamond_inspect_redcap_subjid.py \
      --redcap_csv "/Volumes/Extreme Pro/Red Cap data/ThopazRCT2_raw.csv"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
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


def normalize_case_id(raw_id: object) -> tuple[Optional[str], Optional[str], str]:
    """diamond_build_true_pal_labels.py の normalize_case_id と同一ロジック（ID体系の一貫性のため複製）。
    "施設番号-症例番号[接尾辞]" 形式のIDを正規化する。
    戻り値: (strict_key, loose_key, cleaned_raw)"""
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


def main() -> int:
    args = parse_args()
    path = Path(args.redcap_csv).expanduser()
    df = read_redcap_csv(path, args.encoding)
    print(f"[INFO] 生データ読込: {len(df)}行, {len(df.columns)}列")

    subjid_col = resolve_col(df, args.subjid_col)
    if subjid_col is None:
        candidates = [c for c in df.columns if "subj" in normalize_colname(c)]
        raise RuntimeError(f"subjid列が見つかりません。候補列名: {candidates}")
    print(f"[INFO] 解決された subjid 列名: '{subjid_col}'")

    raw_series = df[subjid_col].dropna().astype(str).str.strip()
    n_raw_nonnull = len(raw_series)
    n_raw_unique = raw_series.nunique()
    print(f"[INFO] subjid非欠損行数={n_raw_nonnull}, 生のユニーク数={n_raw_unique}")

    # 表記ゆれ正規化（diamond_build_true_pal_labels.py と同一ロジック）
    strict_keys, loose_keys, unparsed_n = [], [], 0
    for v in raw_series:
        sk, lk, _ = normalize_case_id(v)
        strict_keys.append(sk)
        loose_keys.append(lk)
        if sk is None:
            unparsed_n += 1
    strict_series = pd.Series(strict_keys)
    loose_series = pd.Series(loose_keys)

    n_strict_unique = strict_series.dropna().nunique()
    n_loose_unique = loose_series.dropna().nunique()
    print(f"[INFO] 正規化後(strict_key=施設-症例番号+接尾辞)のユニーク数={n_strict_unique}")
    print(f"[INFO] 正規化後(loose_key=施設-症例番号のみ、接尾辞無視)のユニーク数={n_loose_unique}")
    print(f"[INFO] 施設-症例番号パターンにマッチしない(パース不能)行数={unparsed_n} ({unparsed_n/n_raw_nonnull:.1%})")

    if n_loose_unique < n_raw_unique:
        collapse_ratio = n_raw_unique / n_loose_unique if n_loose_unique else float("inf")
        print(
            f"[結論候補] 正規化により生のユニーク数({n_raw_unique})が"
            f"{n_loose_unique}まで減少（{collapse_ratio:.1f}倍の圧縮）。"
            f"表記ゆれによる同一患者の分裂が主因である可能性が高い。"
        )
    else:
        print(
            "[結論候補] 正規化してもユニーク数がほぼ変化していない。"
            "表記ゆれ以外の原因（例: このCSVが単一研究でなく複数プロジェクト/"
            "サブスタディを含む、subjidが実は患者単位でない別の識別子である等）を検討する必要あり。"
        )

    # 1患者(loose_key)あたりの生subjid値の分布（何種類の表記が同一患者に対応しているか）
    loose_valid = loose_series.dropna()
    if len(loose_valid):
        variants_per_patient = pd.Series(raw_series.values, index=loose_series.index)[loose_series.notna()].groupby(loose_valid).nunique()
        print(
            f"[INFO] loose_key(患者)あたりの生subjid表記バリエーション数: "
            f"min={variants_per_patient.min()}, median={variants_per_patient.median():.1f}, "
            f"max={variants_per_patient.max()}, 2種類以上={int((variants_per_patient > 1).sum())}/{len(variants_per_patient)}患者"
        )

    # 行数（groupby前）の分布 = 1患者あたり何行あるか
    if len(loose_valid):
        rows_per_loose = loose_series[loose_series.notna()].value_counts()
        print(
            f"[INFO] loose_key(患者)あたりの行数: "
            f"min={rows_per_loose.min()}, median={rows_per_loose.median():.1f}, max={rows_per_loose.max()}"
        )

    # イベント列（存在すれば）: イベント名は個人情報ではないため安全に表示
    event_col = resolve_col(df, args.event_col)
    if event_col:
        print(f"[INFO] イベント列 '{event_col}' の値分布(件数):")
        vc = df[event_col].value_counts(dropna=False)
        for name, cnt in vc.items():
            print(f"    {name}: {cnt}行")
    else:
        print(f"[INFO] イベント列（--event_col='{args.event_col}'）は見つかりませんでした")

    # subjidの文字列構造（値そのものは出さず、構造だけ）
    lengths = raw_series.str.len()
    print(f"[INFO] subjid文字列長: min={lengths.min()}, median={lengths.median():.0f}, max={lengths.max()}")
    has_space = raw_series.str.contains(r"\s").mean()
    has_letter = raw_series.str.contains(r"[a-zA-Z]").mean()
    has_paren = raw_series.str.contains(r"[（(）)]").mean()
    print(f"[INFO] 空白を含む割合={has_space:.1%}, 英字を含む割合={has_letter:.1%}, 括弧を含む割合={has_paren:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
