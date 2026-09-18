#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_inspect_dev_candidate_values.py

目的:
  diamond_inspect_dev_background_candidates.py の列名一覧をもとに、
  単施設(DIAMOND)コホートTable1の元データ候補を「値のレベル」で確認する。
  ただし患者個人を特定できる生データ（患者番号・生年月日そのもの等）は
  出力しない設計とする。

確認する内容:
  1. 「背景」系シート（DIAMOND case file.xlsxの'背景さまり'、
     データ解析用.xlsxの'背景'、データ解析用231022.xlsxの'背景'）を
     まるごと表示する。列名がFactor/Group/Overallで85行程度と、
     患者レベルではなく変数レベルの集計表に見えるため、
     **既存のTable1がここに眠っている可能性がある**。
  2. 本命候補（DIAMOND case file.xlsx の '本当の除外なし921例' シート）の
     Table1候補列について、患者番号等は出さず、
     件数・欠測数・（カテゴリ変数なら)度数、（連続変数なら)平均・SD・中央値のみ表示。
  3. 身長・体重・肺機能（'臨床因子追加'シート）の同様の要約。
  4. '本当の除外なし921例' と '臨床因子追加' が患者番号で
     どれだけ結合できるかを、件数だけで確認する（値は出さない）。

使い方:
  python diamond_inspect_dev_candidate_values.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CASE_FILE = Path("~/Documents/DIAMOND/DIAMOND case file.xlsx").expanduser()
ANALYSIS_FILE = Path("~/Documents/DIAMOND/データ解析用.xlsx").expanduser()
ANALYSIS_FILE_231022 = Path("~/Documents/DIAMOND/データ解析用231022.xlsx").expanduser()

CANDIDATE_TABLE1_COLS = [
    "肺瘻0or1", "肺瘻",
    "PRE-sex", "PRE-age", "PRE-age(>70)",
    "PRE-smoking or never", "PRE-smoking", "PRE-S.I.",
    "PRE-COPD", "PRE-IP", "PRE-DM", "PRE-anticoag/plt", "PRE-steroid", "PRE-PS",
    "疾患の種類(1)", "疾患の種類(2)", "主病巣側",
]

CLINICAL_FACTOR_COLS = [
    "身長(cm)", "体重(kg)", "PRE-VC", "PRE-FVC", "PRE-FVC(%)",
    "PRE-FEV1.0", "PRE-FEV1.0(%)",
    "INTRA-Leak test(0=leakなし、1=あり/消失、2=あり/消失確認なし)",
    "INTRA-sealant(0=なし、1=のりのみ、2=シートのみ、3=のり＋シート、4=縫縮あり)",
    "INTRA-adhesion",
]

JOIN_KEY_CANDIDATES = ["患者番号", "通し番号"]


def show_background_sheet(path: Path, sheet: str) -> None:
    print(f"\n===== 背景シート全体表示: {path.name} / {sheet} =====")
    if not path.exists():
        print("  [NOT FOUND]")
        return
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return
    with pd.option_context("display.max_rows", 200, "display.max_columns", 30, "display.width", 200):
        print(df.to_string())


def summarize_candidate_columns(path: Path, sheet: str, cols: list[str]) -> None:
    print(f"\n===== 候補列の要約（値は集計のみ、生データは表示しない）: {path.name} / {sheet} =====")
    if not path.exists():
        print("  [NOT FOUND]")
        return
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return
    print(f"  総行数={len(df)}")
    for c in cols:
        if c not in df.columns:
            print(f"  - {c}: [列が見つかりません]")
            continue
        s = df[c]
        n_missing = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s):
            print(
                f"  - {c}: 数値, n={int(s.notna().sum())}, 欠測={n_missing}, "
                f"平均={s.mean():.2f}, SD={s.std():.2f}, 中央値={s.median():.2f}"
            )
        else:
            vc = s.value_counts(dropna=True)
            print(f"  - {c}: カテゴリ, 欠測={n_missing}, 度数={vc.to_dict()}")


def check_join_feasibility(
    path_a: Path, sheet_a: str, path_b: Path, sheet_b: str, key_candidates: list[str]
) -> None:
    print(f"\n===== 結合可否チェック（件数のみ、値は出さない）: {sheet_a} <-> {sheet_b} =====")
    if not path_a.exists() or not path_b.exists():
        print("  [NOT FOUND]")
        return
    try:
        df_a = pd.read_excel(path_a, sheet_name=sheet_a)
        df_b = pd.read_excel(path_b, sheet_name=sheet_b)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return
    for key in key_candidates:
        if key in df_a.columns and key in df_b.columns:
            common = set(df_a[key].dropna()) & set(df_b[key].dropna())
            print(
                f"  - key='{key}': A側{int(df_a[key].notna().sum())}件 / "
                f"B側{int(df_b[key].notna().sum())}件 / 共通{len(common)}件"
            )
        else:
            print(f"  - key='{key}': 片方または両方に列がありません")


def main() -> int:
    # 1. 背景系シートの中身を全表示（既存Table1の可能性を確認）
    show_background_sheet(CASE_FILE, "背景さまり")
    show_background_sheet(ANALYSIS_FILE, "背景")
    show_background_sheet(ANALYSIS_FILE_231022, "背景")

    # 2. 本命候補（患者背景＋肺瘻ラベル）の要約
    summarize_candidate_columns(CASE_FILE, "本当の除外なし921例", CANDIDATE_TABLE1_COLS)

    # 3. 身長・体重・肺機能（BMI算出・追加変数用）の要約
    summarize_candidate_columns(CASE_FILE, "臨床因子追加", CLINICAL_FACTOR_COLS)

    # 4. 上記2つが患者番号で結合できるか
    check_join_feasibility(CASE_FILE, "本当の除外なし921例", CASE_FILE, "臨床因子追加", JOIN_KEY_CANDIDATES)

    print(
        "\n[NEXT] 上記の出力をこのまま貼ってください。「背景」シートが既存Table1かどうか、"
        "本命候補の列内容が妥当か、結合可否を確認して diamond_table1_dev.py を書きます。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
