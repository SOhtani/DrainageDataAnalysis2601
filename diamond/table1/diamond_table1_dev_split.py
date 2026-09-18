#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_table1_dev_split.py

目的:
  9:1分割モデル(full/12h/18h/24h)の学習で使われた「単施設9(train 90%)」
  「単施設1(internal test 10%)」それぞれについて、患者背景表(Table1)を作る。

  ⚠️ 背景（2026-08-16）: 9:1分割・モデル学習を行った元スクリプト
  (`diamond_external_validate_generic.py` 相当)はこのワークスペースのgit管理下に
  保存されておらず、正確な分割再現ができない。ただし学習成果物
  `~/Documents/DIAMOND/DIAMOND/split_models/internal_test_predictions_{horizon}.csv`
  （列: case_id, label, ai_probability）にinternal test(10%)の症例IDが残っている
  ことを確認済み（24hで93行=ヘッダ+92件、既知のinternal_test_n=92と一致）。
  本スクリプトはこのファイルを使い、
    - internal test(単施設1) = case_idがinternal_test_predictions_{horizon}.csvに
      含まれる症例
    - train(単施設9)         = diamond_table1_dev.pyの母集団(n=921、背景データ完備例)
      のうち、internal testに含まれない残り
  として2群のTable1を作る。

  ⚠️ 重要な制約: 「train(単施設9)」は「その horizon でモデル学習に実際に使われた
  90%」と完全には一致しない可能性がある。モデル学習コホートの母集団
  (horizon別 n_included=913〜1020)と本Table1の母集団(n=921、背景データ完備例)は
  別の絞り込み基準によるため。internal test側はinternal_test_predictions_*.csvの
  症例と直接対応するため、より信頼できる。

  case_idの表記ゆれ（先頭ゼロ等）に対応するため、患者番号/通し番号との突合は
  完全一致→数値正規化した緩い一致、の2段階で行う。マッチしなかった件数を
  必ず報告する（想定92件からの乖離が大きい場合はID体系の見直しが必要）。

  患者を特定できる値(患者番号そのもの)は一切出力しない。件数・集計値のみ。

使い方（例）:
  python diamond_table1_dev_split.py \
      --base_csv ~/Documents/DIAMOND/DIAMOND/table1_source_base.csv \
      --factor_csv ~/Documents/DIAMOND/DIAMOND/table1_source_factor.csv \
      --internal_test_csv ~/Documents/DIAMOND/DIAMOND/split_models/internal_test_predictions_24h.csv \
      --horizon_label 24h \
      --output_dir ~/DIAMOND-local/output
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CASE_FILE_DEFAULT = "~/Documents/DIAMOND/DIAMOND case file.xlsx"
BASE_SHEET = "本当の除外なし921例"
FACTOR_SHEET = "臨床因子追加"

CONTINUOUS_VARS = [
    ("年齢", "PRE-age"),
    ("喫煙指数(Brinkman Index)", "PRE-S.I."),
]
CATEGORICAL_VARS = [
    ("性別", "PRE-sex"),
    ("喫煙(有無)", "PRE-smoking or never"),
    ("喫煙(現喫煙)", "PRE-smoking"),
    ("COPD", "PRE-COPD"),
    ("間質性肺炎", "PRE-IP"),
    ("糖尿病", "PRE-DM"),
    ("抗血栓薬内服", "PRE-anticoag/plt"),
    ("ステロイド", "PRE-steroid"),
    ("PS", "PRE-PS"),
    ("疾患の種類(1)", "疾患の種類(1)"),
    ("主病巣側", "主病巣側"),
    ("肺瘻(アウトカム, 0/1)", "肺瘻0or1"),
]
KEYWORD_CANDIDATES = {
    "continuous": ["手術時間", "出血量", "surgtime", "bloodloss"],
    "categorical": [
        "手術の種類", "Approach", "アプローチ", "肺切除範囲",
        "切除肺葉", "開胸法", "ﾘﾝﾊﾟ節郭清",
    ],
}
FACTOR_CONTINUOUS_VARS = [
    ("身長(cm)", "身長(cm)"),
    ("体重(kg)", "体重(kg)"),
    ("VC", "PRE-VC"),
    ("FVC", "PRE-FVC"),
    ("FVC(%)", "PRE-FVC(%)"),
    ("FEV1.0", "PRE-FEV1.0"),
    ("FEV1.0(%)", "PRE-FEV1.0(%)"),
]
FACTOR_CATEGORICAL_VARS = [
    ("術中Leak test", "INTRA-Leak test(0=leakなし、1=あり/消失、2=あり/消失確認なし)"),
    ("術中sealant", "INTRA-sealant(0=なし、1=のりのみ、2=シートのみ、3=のり＋シート、4=縫縮あり)"),
    ("胸腔内癒着", "INTRA-adhesion"),
]
JOIN_KEY_CANDIDATES = ["患者番号", "通し番号"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case_file", default=CASE_FILE_DEFAULT)
    p.add_argument("--base_csv", default=None)
    p.add_argument("--factor_csv", default=None)
    p.add_argument("--internal_test_csv", required=True, help="internal_test_predictions_{horizon}.csv")
    p.add_argument("--internal_test_id_col", default="case_id")
    p.add_argument("--horizon_label", default="24h", help="出力ファイル名に使うラベル(full/12h/18h/24h)")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def read_csv_robust(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "mac_roman"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"{path} を読めるエンコーディングが見つかりませんでした: {last_err}")


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


def find_by_keyword(df: pd.DataFrame, keyword: str) -> Optional[str]:
    for c in df.columns:
        if keyword.lower() in str(c).lower():
            return c
    return None


def classify_approach_from_text(text: object) -> Optional[str]:
    if pd.isna(text):
        return None
    t = str(text)
    if re.search(r"ロボット支援|RATS", t, re.IGNORECASE):
        return "ロボット支援下(RATS)"
    if re.search(r"完全胸腔鏡下|完全鏡視下|完全鏡視|cVATS", t, re.IGNORECASE):
        return "完全胸腔鏡下(cVATS相当)"
    if re.search(r"hVATS", t, re.IGNORECASE):
        return "胸腔鏡補助下(hVATS相当)"
    if re.search(r"(?<![a-zA-Z])VATS", t, re.IGNORECASE):
        return "完全胸腔鏡下(cVATS相当)"
    if re.search(r"胸腔鏡", t):
        return "胸腔鏡補助下(hVATS相当)"
    if re.search(r"開胸", t):
        return "開胸(thoracotomy)"
    return "不明(文字列からアプローチ判定不能)"


def classify_extent_from_text(text: object) -> Optional[str]:
    if pd.isna(text):
        return None
    t = str(text)
    if re.search(r"全摘", t):
        return "肺全摘(pneumonectomy)"
    if re.search(r"葉切除", t):
        return "肺葉切除(lobectomy)"
    if re.search(r"区域切除|亜区域切除|底区域切除|舌区域切除|大区域切除|S\d[a-zA-Z0-9+]*区域|S\d[a-zA-Z0-9+]*切除", t):
        return "区域切除(segmentectomy)"
    if re.search(r"部分切除", t):
        return "部分切除(wedge)"
    return "不明(文字列から切除範囲判定不能)"


def fix_percent_decimal_point(
    series: pd.Series, plausible_max: float = 300.0, accept_max: float = 200.0,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    fixed = s.copy()
    out_of_range = (s < 0) | (s > plausible_max)
    for idx in s[out_of_range].index:
        raw = s.loc[idx]
        candidate_recovered = None
        for divisor in (10.0, 100.0, 1000.0):
            candidate = raw / divisor
            if 0 <= candidate <= plausible_max:
                candidate_recovered = candidate
                break
        if candidate_recovered is None or candidate_recovered > accept_max:
            fixed.loc[idx] = np.nan
        else:
            fixed.loc[idx] = candidate_recovered
    return fixed


def summarize_continuous(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    n = int(s.notna().sum())
    missing = int(s.isna().sum())
    if n == 0:
        return {"n": 0, "missing": missing, "mean": np.nan, "sd": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": n, "missing": missing,
        "mean": float(s.mean()), "sd": float(s.std()),
        "median": float(s.median()), "min": float(s.min()), "max": float(s.max()),
    }


MAX_CATEGORIES_SHOWN = 15


def summarize_categorical(series: pd.Series, max_categories: int = MAX_CATEGORIES_SHOWN) -> pd.DataFrame:
    s = series.astype("object")
    missing = int(s.isna().sum())
    counts = s.value_counts(dropna=True)
    total = int(counts.sum())
    rows = []
    top = counts.iloc[:max_categories]
    rest = counts.iloc[max_categories:]
    for cat, n in top.items():
        rows.append({"category": cat, "n": int(n), "pct": (n / total * 100) if total else np.nan})
    if len(rest):
        rows.append({"category": f"その他({len(rest)}カテゴリ)", "n": int(rest.sum()), "pct": (rest.sum() / total * 100) if total else np.nan})
    rows.append({"category": "(missing)", "n": missing, "pct": np.nan})
    return pd.DataFrame(rows)


def build_table1_grouped(df: pd.DataFrame, continuous_vars, categorical_vars, group_col: str) -> pd.DataFrame:
    groups = sorted(df[group_col].dropna().unique().tolist())
    records = []
    for label, col in continuous_vars:
        resolved = resolve_col(df, col)
        if resolved is None:
            records.append({"variable": label, "col": col, "type": "continuous", "note": "列が見つかりません"})
            continue
        for g in groups:
            stat = summarize_continuous(df.loc[df[group_col] == g, resolved])
            records.append({"variable": label, "col": resolved, "group": g, "type": "continuous", **stat})
    for label, col in categorical_vars:
        resolved = resolve_col(df, col)
        if resolved is None:
            records.append({"variable": label, "col": col, "type": "categorical", "note": "列が見つかりません"})
            continue
        for g in groups:
            cat_df = summarize_categorical(df.loc[df[group_col] == g, resolved])
            for _, r in cat_df.iterrows():
                records.append({
                    "variable": label, "col": resolved, "group": g, "type": "categorical",
                    "category": r["category"], "n": r["n"], "pct": r["pct"],
                })
    return pd.DataFrame(records)


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = df.columns.tolist()
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)


def normalize_id_loose(v: object) -> Optional[str]:
    """先頭ゼロ・空白・小数点(.0)の表記ゆれを吸収した緩いID正規化。"""
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    return s


def normalize_id_int_part(v: object) -> Optional[str]:
    """2026-08-16追加: internal_test_predictions_*.csvのcase_idが
    "XXXXXX.XXXXXXX"(整数部6桁+小数部6-7桁)という形式であることが判明。
    構造確認のみで値は見ていないが、整数部が患者番号(6桁程度)、小数部は
    どこかでfloat型を経由した際に生じた誤差(浮動小数点ノイズ)である可能性が高い。
    この仮説に基づき、小数点より前の整数部だけを取り出して候補キーとする。"""
    if pd.isna(v):
        return None
    s = str(v).strip()
    int_part = s.split(".")[0]
    if re.fullmatch(r"\d+", int_part):
        return str(int(int_part))
    return None


def _load_excel_sheets(path: Path, sheet_names: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
    try:
        xls = pd.ExcelFile(path, engine="openpyxl", engine_kwargs={"read_only": True})
        result: Dict[str, Optional[pd.DataFrame]] = {}
        for name in sheet_names:
            result[name] = pd.read_excel(xls, sheet_name=name) if name in xls.sheet_names else None
        return result
    except TypeError:
        import openpyxl
        result = {}
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        try:
            for name in sheet_names:
                if name not in wb.sheetnames:
                    result[name] = None
                    continue
                ws = wb[name]
                row_iter = ws.iter_rows(values_only=True)
                header_raw = next(row_iter)
                header = [str(h) if h is not None else f"Unnamed_{i}" for i, h in enumerate(header_raw)]
                ncol = len(header)
                data = []
                for r in row_iter:
                    if len(r) < ncol:
                        r = tuple(r) + (None,) * (ncol - len(r))
                    elif len(r) > ncol:
                        r = r[:ncol]
                    data.append(r)
                result[name] = pd.DataFrame(data, columns=header)
        finally:
            wb.close()
        return result


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.base_csv and args.factor_csv:
        base = read_csv_robust(Path(args.base_csv).expanduser())
        factor = read_csv_robust(Path(args.factor_csv).expanduser())
    else:
        case_file = Path(args.case_file).expanduser()
        print(f"[INFO] {case_file.name} を読み込み中...")
        sheets = _load_excel_sheets(case_file, [BASE_SHEET, FACTOR_SHEET])
        base, factor = sheets.get(BASE_SHEET), sheets.get(FACTOR_SHEET)
        if base is None or factor is None:
            raise RuntimeError("必要なシートが見つかりません")

    base = base.dropna(how="all").reset_index(drop=True)
    factor = factor.dropna(how="all").reset_index(drop=True)
    print(f"[INFO] {BASE_SHEET}: {len(base)}行 / {FACTOR_SHEET}: {len(factor)}行")

    join_key = next((k for k in JOIN_KEY_CANDIDATES if k in base.columns and k in factor.columns), None)
    merged = base
    if join_key:
        wanted_factor_cols = [c for _, c in FACTOR_CONTINUOUS_VARS + FACTOR_CATEGORICAL_VARS if c in factor.columns]
        factor_sub = factor[[join_key] + wanted_factor_cols].drop_duplicates(subset=[join_key])
        merged = base.merge(factor_sub, on=join_key, how="left", suffixes=("", "_factor"))
        print(f"[INFO] '{join_key}'で結合完了")
    else:
        raise RuntimeError(f"結合キー({JOIN_KEY_CANDIDATES})が見つかりません")

    if "身長(cm)" in merged.columns and "体重(kg)" in merged.columns:
        h_m = pd.to_numeric(merged["身長(cm)"], errors="coerce") / 100.0
        w_kg = pd.to_numeric(merged["体重(kg)"], errors="coerce")
        merged["_bmi_computed"] = w_kg / (h_m ** 2)

    fixed_pct_vars = []
    for label, col in [("FVC(%)", "PRE-FVC(%)"), ("FEV1.0(%)", "PRE-FEV1.0(%)")]:
        resolved = resolve_col(merged, col)
        if resolved is None:
            continue
        merged[f"_{resolved}_fixed"] = fix_percent_decimal_point(merged[resolved])
        fixed_pct_vars.append((f"{label}(小数点補正後)", f"_{resolved}_fixed"))

    proc_col = find_by_keyword(merged, "手術術式")
    derived_categorical_vars = []
    if proc_col:
        merged["_derived_approach"] = merged[proc_col].apply(classify_approach_from_text)
        merged["_derived_extent"] = merged[proc_col].apply(classify_extent_from_text)
        derived_categorical_vars = [
            ("アプローチ(術式文字列から推定)", "_derived_approach"),
            ("切除範囲(術式文字列から推定)", "_derived_extent"),
        ]

    extra_continuous = [(f"{kw}(自動検出:{c})", c) for kw in KEYWORD_CANDIDATES["continuous"] if (c := find_by_keyword(base, kw))]
    extra_categorical = [(f"{kw}(自動検出:{c})", c) for kw in KEYWORD_CANDIDATES["categorical"] if (c := find_by_keyword(base, kw))]

    # --- internal test症例IDとの突合 ---
    # 2026-08-16: 実データ確認の結果、internal_test_predictions_*.csvのcase_idは
    # 患者番号/通し番号とは異なる独自の形式("XXXXXX.XXXXXXX"、整数部6桁+小数部6-7桁)
    # であることが判明。整数部6桁を患者番号相当の候補キーとして追加で試す
    # (normalize_id_int_part)。患者番号・通し番号のどちらとも、また複数の正規化方式で
    # 突合を試し、最も一致率が高い組み合わせを採用する。
    it_df = read_csv_robust(Path(args.internal_test_csv).expanduser())
    it_id_col = resolve_col(it_df, args.internal_test_id_col)
    if it_id_col is None:
        raise RuntimeError(f"internal_test_csvに'{args.internal_test_id_col}'列が見つかりません: {it_df.columns.tolist()}")
    it_id_series = it_df[it_id_col].dropna()
    print(f"[INFO] internal_test_csv: {len(it_df)}行, ユニークID={it_id_series.astype(str).str.strip().nunique()}件")

    id_columns_to_try = []
    for cand in JOIN_KEY_CANDIDATES:
        resolved = resolve_col(merged, cand)
        if resolved and resolved not in [c for c, _ in id_columns_to_try]:
            id_columns_to_try.append((resolved, cand))

    normalizers = [
        ("厳密一致", lambda v: str(v).strip()),
        ("緩い一致(先頭ゼロ/.0除去)", normalize_id_loose),
        ("整数部一致(case_idの小数点以前)", normalize_id_int_part),
    ]

    best = None  # (n_matched, id_col_resolved, id_col_label, norm_label, is_match_series)
    for id_col_resolved, id_col_label in id_columns_to_try:
        merged_id_raw_str = merged[id_col_resolved].astype(str).str.strip()
        for norm_label, norm_fn in normalizers:
            it_keys = set(v for v in (norm_fn(x) for x in it_id_series) if v is not None)
            if not it_keys:
                continue
            merged_keys = merged[id_col_resolved].apply(norm_fn)
            is_match = merged_keys.isin(it_keys) & merged_keys.notna()
            n = int(is_match.sum())
            print(f"[INFO] 突合試行: ID列={id_col_label}, 方式={norm_label} -> {n}/{len(it_id_series)}件一致")
            if best is None or n > best[0]:
                best = (n, id_col_resolved, id_col_label, norm_label, is_match)

    n_matched, best_id_col, best_id_label, best_norm_label, is_internal_test = best
    n_expected = it_id_series.astype(str).str.strip().nunique()
    print(
        f"[INFO] 最良の組み合わせを採用: ID列={best_id_label}, 方式={best_norm_label} -> "
        f"{n_matched}/{n_expected}件一致（Table1母集団 n={len(merged)}）"
    )
    if n_matched < n_expected * 0.8:
        print(
            "[WARN] 一致率が80%未満です。case_idの体系（患者番号/通し番号のどちらか、"
            "または別のID体系）が想定と異なる可能性があります。"
            "--internal_test_id_col の指定や、internal_test_csvのID形式を確認してください。"
        )

    merged["_split_group"] = np.where(is_internal_test, f"単施設1(internal test, {args.horizon_label})", f"単施設9(train, {args.horizon_label}推定)")
    print(merged["_split_group"].value_counts().rename("n").to_string())

    continuous_vars = CONTINUOUS_VARS + extra_continuous + FACTOR_CONTINUOUS_VARS + fixed_pct_vars
    if "_bmi_computed" in merged.columns:
        continuous_vars = continuous_vars + [("BMI(算出値)", "_bmi_computed")]
    categorical_vars = CATEGORICAL_VARS + extra_categorical + FACTOR_CATEGORICAL_VARS + derived_categorical_vars

    table1 = build_table1_grouped(merged, continuous_vars, categorical_vars, "_split_group")
    csv_path = output_dir / f"table1_dev_split_{args.horizon_label}.csv"
    md_path = output_dir / f"table1_dev_split_{args.horizon_label}.md"
    table1.to_csv(csv_path, index=False)
    header = (
        f"# 単施設(DIAMOND)コホート Table1: train(単施設9) vs internal test(単施設1) — horizon={args.horizon_label}\n\n"
        f"internal test症例の特定: `{args.internal_test_csv}` とのID突合（{n_matched}/{n_expected}件一致）。\n\n"
        f"⚠️ train側はTable1母集団(n={len(merged)}、背景データ完備例)からinternal testを除いた残りであり、"
        f"その horizon で実際にモデル学習に使われた90%集団と完全に同一である保証はない"
        f"（モデル学習コホートの母集団と本Table1母集団の絞り込み基準が異なるため）。\n\n"
    )
    md_path.write_text(header + markdown_table(table1), encoding="utf-8")
    print(f"[OK] 保存: {csv_path} / {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
