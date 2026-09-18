#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_inspect_dev_background_candidates.py

目的:
  単施設(DIAMOND)コホートの患者背景データ(Table1)がどのファイルに
  入っているか特定するため、候補4ファイルの「シート名・列名・行数」だけを
  安全に一覧表示する（値そのものは表示しない＝出力を短く保つ）。

  実行結果をそのままチャットに貼っていただければ、どのファイルが
  Table1に使えるか（年齢・性別・術式などの列があるか）を判断し、
  diamond_table1_dev.py を書きます。

使い方:
  python diamond_inspect_dev_background_candidates.py
  （デフォルトのパスをそのまま使う。違う場所にあれば --files で上書き可）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_FILES = [
    "~/Documents/DIAMOND/classification_results.xlsx",
    "~/Documents/DIAMOND/DIAMOND case file.xlsx",
    "~/Documents/DIAMOND/データ解析用.xlsx",
    "~/Documents/DIAMOND/データ解析用231022.xlsx",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    p.add_argument("--max_cols_preview", type=int, default=60, help="列名を表示する上限数")
    return p.parse_args()


def inspect_file(path: Path, max_cols_preview: int) -> None:
    print(f"\n===== {path} =====")
    if not path.exists():
        print("  [NOT FOUND]")
        return
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"  [ERROR] 開けませんでした: {e}")
        return
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, nrows=5)  # 先頭5行だけで十分（列名・型確認用）
            full_shape_df = pd.read_excel(xls, sheet_name=sheet, usecols=[0])
            n_rows = len(full_shape_df)
        except Exception as e:
            print(f"  - sheet='{sheet}': [ERROR] {e}")
            continue
        cols = df.columns.tolist()
        cols_preview = cols[:max_cols_preview]
        more = f" ...(+{len(cols) - max_cols_preview}列)" if len(cols) > max_cols_preview else ""
        print(f"  - sheet='{sheet}': 行数={n_rows}, 列数={len(cols)}")
        print(f"      列名: {cols_preview}{more}")


def main() -> int:
    args = parse_args()
    for f in args.files:
        inspect_file(Path(f).expanduser(), args.max_cols_preview)
    print(
        "\n[NEXT] 上記の列名一覧をこのままチャットに貼ってください。"
        "年齢/性別/術式/BMI(身長体重)/出血量/手術時間らしき列があるファイルを"
        "Table1の元データとして diamond_table1_dev.py を書きます。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
