#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_compare_cohorts_stats.py

目的:
  単施設(DIAMOND)コホートと外部(順天堂RCT-2)コホートを、患者単位の実データから
  直接比較し、統計学的検定（連続変数=Mann-Whitney U検定、カテゴリ変数=カイ二乗検定
  [期待度数<5のセルがあればFisher正確検定に自動切替]）を行う。

  入力は以下2つのCSV（いずれも diamond_table1_dev.py / diamond_table1_external.py の
  --dump_patient_csv オプションで事前に生成しておく、患者単位ワイド形式データ）:
    --dev_csv:      diamond_table1_dev.py --dump_patient_csv の出力
    --external_csv: diamond_table1_external.py --dump_patient_csv の出力
                     （--labels_csv も渡して true_pal を含めておくこと）

  出力は集計値（n・中央値・IQR・%・検定統計量・p値）のみ。患者単位の行は
  一切表示・保存しない（プライバシー制約）。

⚠️ 定義が完全には一致しない変数について（本スクリプトが機械的に検定するのみで、
  臨床的な解釈は使用者が行うこと）:
  - 喫煙歴「あり」: 単施設は生の喫煙歴（既往/現在）、外部はBrinkman指数≧100
    （必ずしも同じ患者集合を指さない）。
  - アプローチ「RATS」: 2026-08-19、cVATSとhVATSを外部側で区別できない問題を回避するため
    「cVATS or RATS」から**RATS単独**に定義変更。単施設は構造化Approach列(0=hVATS/1=cVATS/
    2=RATS/3=開胸)のRATS(2)、外部REDCapのapproフィールド(1=開胸/2=胸腔鏡補助下/3=ロボット)
    のロボット(3)はいずれも一意に判定できるため、この変数はcVATS/hVATSのような曖昧さが無い。
  - PS（Performance Status）・FVC: 外部(RCT-2)のREDCapに項目自体が存在しないため
    検定不可（対比不可として出力）。
  - PAL: 単施設は代理定義（肺瘻の有無）、外部は親試験一次エンドポイント準拠の
    真のPAL定義。母集団も異なる（外部はPOD1エアリーク100-1000mL/minの選択集団）。

使い方（例）:
  python diamond_compare_cohorts_stats.py \
      --dev_csv ~/DIAMOND-local/output/dev_cohort_patient_level.csv \
      --external_csv ~/DIAMOND-local/output/external_cohort_patient_level.csv
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dev_csv", required=True, help="diamond_table1_dev.py --dump_patient_csv の出力")
    p.add_argument("--external_csv", required=True, help="diamond_table1_external.py --dump_patient_csv の出力")
    return p.parse_args()


def fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def normalize_token(v: object) -> str:
    if pd.isna(v):
        return ""
    # ⚠️ 2026-08-18修正: 単施設側の生値が全角括弧（喫煙していた（既往）等）、
    # 外部側は基本半角のため、幅の違いだけで一致判定に失敗していた
    # （実データで「喫煙歴あり」が0/921という明らかな異常値になり発覚）。
    # NFKC正規化で全角/半角の違いを吸収してから比較する。
    s = unicodedata.normalize("NFKC", str(v)).strip()
    # "1.0" 等の浮動小数表記を整数風に正規化（"1" と一致させるため）
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s.lower()


def is_positive(value: object, positive_tokens: Sequence[str]) -> Optional[bool]:
    """value が欠損ならNone、positive_tokens(正規化済み)に一致すればTrue、非欠損で不一致ならFalse。"""
    if pd.isna(value):
        return None
    tok = normalize_token(value)
    norm_positives = {normalize_token(t) for t in positive_tokens}
    return tok in norm_positives


def continuous_row(label: str, dev: pd.Series, ext: pd.Series, note: str = "") -> dict:
    dev_n = pd.to_numeric(dev, errors="coerce").dropna()
    ext_n = pd.to_numeric(ext, errors="coerce").dropna()
    row = {
        "variable": label,
        "dev_n": len(dev_n), "dev_median": dev_n.median() if len(dev_n) else np.nan,
        "dev_q1": dev_n.quantile(0.25) if len(dev_n) else np.nan,
        "dev_q3": dev_n.quantile(0.75) if len(dev_n) else np.nan,
        "ext_n": len(ext_n), "ext_median": ext_n.median() if len(ext_n) else np.nan,
        "ext_q1": ext_n.quantile(0.25) if len(ext_n) else np.nan,
        "ext_q3": ext_n.quantile(0.75) if len(ext_n) else np.nan,
        "test": "Mann-Whitney U", "statistic": np.nan, "p_value": np.nan, "note": note,
    }
    if note:
        row["test"] = "対比不可"
        return row
    if not HAS_SCIPY:
        row["note"] = "scipy未インストールのためp値計算不可（pip install scipyが必要）"
        return row
    if len(dev_n) < 1 or len(ext_n) < 1:
        row["note"] = "いずれかの群でn=0のため検定不可"
        return row
    stat, p = scipy_stats.mannwhitneyu(dev_n, ext_n, alternative="two-sided")
    row["statistic"] = stat
    row["p_value"] = p
    return row


def categorical_row(
    label: str,
    dev: pd.Series, ext: pd.Series,
    dev_positive_tokens: Sequence[str], ext_positive_tokens: Sequence[str],
    note: str = "",
) -> dict:
    row = {
        "variable": label,
        "dev_n_pos": np.nan, "dev_n_total": np.nan, "dev_pct": np.nan,
        "ext_n_pos": np.nan, "ext_n_total": np.nan, "ext_pct": np.nan,
        "test": "Chi-square", "statistic": np.nan, "p_value": np.nan, "note": note,
    }
    if note:
        row["test"] = "対比不可"
        return row

    dev_flags = dev.apply(lambda v: is_positive(v, dev_positive_tokens))
    ext_flags = ext.apply(lambda v: is_positive(v, ext_positive_tokens))
    dev_flags = dev_flags.dropna()
    ext_flags = ext_flags.dropna()

    dev_pos = int(dev_flags.sum())
    dev_total = len(dev_flags)
    ext_pos = int(ext_flags.sum())
    ext_total = len(ext_flags)

    row["dev_n_pos"] = dev_pos
    row["dev_n_total"] = dev_total
    row["dev_pct"] = (dev_pos / dev_total * 100) if dev_total else np.nan
    row["ext_n_pos"] = ext_pos
    row["ext_n_total"] = ext_total
    row["ext_pct"] = (ext_pos / ext_total * 100) if ext_total else np.nan

    if not HAS_SCIPY:
        row["note"] = (row["note"] + " " if row["note"] else "") + "scipy未インストールのためp値計算不可"
        return row
    if dev_total == 0 or ext_total == 0:
        row["note"] = (row["note"] + " " if row["note"] else "") + "いずれかの群でn=0のため検定不可"
        return row

    table = np.array([
        [dev_pos, dev_total - dev_pos],
        [ext_pos, ext_total - ext_pos],
    ])
    # 期待度数<5のセルがあれば正確検定(Fisher)に自動切替（小標本での近似誤差を避ける）
    expected = scipy_stats.contingency.expected_freq(table)
    if (expected < 5).any():
        odds, p = scipy_stats.fisher_exact(table)
        row["test"] = "Fisher正確検定(期待度数<5のため自動切替)"
        row["statistic"] = odds
        row["p_value"] = p
    else:
        chi2, p, dof, _ = scipy_stats.chi2_contingency(table, correction=True)
        row["test"] = "Chi-square(Yates補正)"
        row["statistic"] = chi2
        row["p_value"] = p
    return row


def print_continuous(row: dict) -> None:
    dev_str = (
        f"{row['dev_median']:.1f} ({row['dev_q1']:.1f}–{row['dev_q3']:.1f}, n={row['dev_n']})"
        if row["dev_n"] else "n/a"
    )
    ext_str = (
        f"{row['ext_median']:.1f} ({row['ext_q1']:.1f}–{row['ext_q3']:.1f}, n={row['ext_n']})"
        if row["ext_n"] else "n/a"
    )
    print(f"  {row['variable']:<30s} 単施設={dev_str:<32s} 外部={ext_str:<32s} "
          f"検定={row['test']:<12s} p={fmt_p(row['p_value'])}  {row['note']}")


def print_categorical(row: dict) -> None:
    dev_str = (
        f"{row['dev_n_pos']}/{row['dev_n_total']} ({row['dev_pct']:.1f}%)"
        if not pd.isna(row["dev_n_total"]) else "n/a"
    )
    ext_str = (
        f"{row['ext_n_pos']}/{row['ext_n_total']} ({row['ext_pct']:.1f}%)"
        if not pd.isna(row["ext_n_total"]) else "n/a"
    )
    print(f"  {row['variable']:<30s} 単施設={dev_str:<20s} 外部={ext_str:<20s} "
          f"検定={row['test']:<30s} p={fmt_p(row['p_value'])}  {row['note']}")


def main() -> int:
    args = parse_args()
    if not HAS_SCIPY:
        print(
            "[WARN] scipyがインストールされていません。p値は計算できません。"
            "'pip install scipy' を実行してから再実行してください（n・%・中央値等は計算します）。"
        )

    dev = pd.read_csv(Path(args.dev_csv).expanduser())
    ext = pd.read_csv(Path(args.external_csv).expanduser())
    print(f"[INFO] 単施設(dev)コホート: n={len(dev)} / 外部(external)コホート: n={len(ext)}")

    rows: List[dict] = []

    print("\n===== 連続変数（Mann-Whitney U検定） =====")
    r = continuous_row("年齢(歳)", dev["age"], ext["age"])
    rows.append(r); print_continuous(r)

    r = continuous_row("SI(Brinkman Index)", dev["si"], ext["si"])
    rows.append(r); print_continuous(r)

    r = continuous_row("FEV1.0(mL)", dev["fev1_ml"], ext["fev1_ml"])
    rows.append(r); print_continuous(r)

    r = continuous_row(
        "FVC(mL)", dev["fvc_ml"], ext.get("fvc_ml", pd.Series(dtype=float)),
        note="外部データなし(REDCapにFVC項目自体が存在しない。VCのみ存在、既知の限界)",
    )
    rows.append(r); print_continuous(r)

    print("\n===== カテゴリ変数（Chi-square / 必要ならFisher正確検定に自動切替） =====")

    r = categorical_row(
        "性別 男(%)", dev["sex"], ext["sex"],
        dev_positive_tokens=["男", "1", "male", "m"],
        ext_positive_tokens=["1"],  # REDCap sex2: 1=男
    )
    rows.append(r); print_categorical(r)

    r = categorical_row(
        "喫煙歴 あり(%) ※定義が異なる、注記参照",
        dev["smoking_ever"], ext["smoking_brinkman100"],
        # 2026-08-18修正: 実データ確認の結果、正しい生値は「喫煙していた」「喫煙している」
        # （括弧付き既往/現在の表記ではなく、先頭スペース込みだが正規化後は括弧なし）と判明。
        # 当初「喫煙していた(既往)」「喫煙している(現在)」という誤った値を想定しており、
        # 単施設側が0/921という明らかな異常値になっていた。
        dev_positive_tokens=["喫煙していた", "喫煙している", "1", "yes"],
        ext_positive_tokens=["2"],  # adjust_cig_result: 2=あり(Brinkman指数100以上)
        note="",
    )
    rows.append(r); print_categorical(r)

    r = categorical_row(
        "PS 0(%)", dev["ps"], ext.get("ps", pd.Series(dtype=object)),
        dev_positive_tokens=["0"], ext_positive_tokens=[],
        note="外部データなし(REDCapにPS項目自体が存在しない、既知の限界)",
    )
    rows.append(r); print_categorical(r)

    r = categorical_row(
        "COPD あり(%)", dev["copd"], ext["copd"],
        dev_positive_tokens=["あり", "1", "yes"],
        ext_positive_tokens=["1"],  # adjust_3: 1=あり
    )
    rows.append(r); print_categorical(r)

    r = categorical_row(
        "疾患の種類 肺癌(%)", dev["primary_disease"], ext["primary_disease"],
        dev_positive_tokens=["肺癌", "1", "原発性肺癌"],
        ext_positive_tokens=["1"],  # primdise: 1=原発性肺癌
    )
    rows.append(r); print_categorical(r)

    r = categorical_row(
        "アプローチ RATS(%)",
        dev["approach_struct_code"], ext["approach_code"],
        dev_positive_tokens=["2"],  # INTRA-Approach: 2=RATS
        ext_positive_tokens=["3"],  # appro: 3=ロボット
        note="",
    )
    rows.append(r); print_categorical(r)

    r = categorical_row(
        "切除範囲 葉切除以上(%)", dev["resection_extent"], ext["resection_extent"],
        dev_positive_tokens=["肺全摘(pneumonectomy)", "肺葉切除(lobectomy)"],
        ext_positive_tokens=["葉切除以上"],
    )
    rows.append(r); print_categorical(r)

    r = categorical_row(
        "術中癒着所見 あり(%)", dev["adhesion"], ext["adhesion"],
        dev_positive_tokens=["あり", "1", "yes"],
        ext_positive_tokens=["あり"],
    )
    rows.append(r); print_categorical(r)

    r = categorical_row(
        "PAL あり(%) ⚠️定義が大きく異なる(注記参照)",
        dev["pal_proxy"], ext["pal_true"],
        dev_positive_tokens=["1"],
        ext_positive_tokens=["1"],
    )
    rows.append(r); print_categorical(r)

    print(
        "\n[OK] 検定完了。⚠️喫煙歴・アプローチ・PALは両コホートで定義が完全には一致しないため、"
        "p値は参考値として解釈してください（患者単位の行は一切出力していません）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
