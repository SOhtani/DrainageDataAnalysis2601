#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_table1_external.py

目的:
  順天堂多施設Thopaz RCT-2 REDCap生データから、外部検証コホートの
  患者背景表(Table 1)を作成する。列名は
  thopaz-methods-2.2-2.3-draft-v1.md の変数辞書に準拠。

  --labels_csv（diamond_build_true_pal_labels.py の出力）を渡すと、
  true_pal（PAL+ / PAL-）でも層別した表を追加出力する。

  ⚠️ 重要な前提（2026-08-16、実データQCで判明。一度誤った仮説で修正し、後に訂正）:
    素朴に subjid でgroupbyすると対象症例数=2354となり、想定していた
    研究コホート規模(申告182例、親試験目標最大199例)から大きく乖離する。
    最初は「8イベントのうち1イベント（行数が全体subjidユニーク総数と完全一致）
    ＝除外例も含む、より大きな母集団の登録/スクリーニングログ」と判断し、
    このイベントを丸ごと除外する修正を入れた。
    しかし`diamond_inspect_redcap_key_columns.py`で列×イベントの非欠損件数を
    確認したところ、**その"最大イベント"こそが年齢(age2)・性別(sex2)・
    手術日(surgedat)・施設(facilities)・喫煙歴(adjust_cig_result)等、
    全登録患者に共通する基本情報を持つ主要イベントである**ことが判明し、
    イベント丸ごと除外は誤り（真の研究コホートの患者からもこれらの列が
    根こそぎ消えてしまう）と分かった。
    正しい絞り込み条件は、この主要イベント内で**割付群(rand)が記録されている
    症例のみに限定する**こと（rand非欠損=210例、既知の182例・199例と矛盾なく
    整合する包含関係）。本スクリプトは全イベントの情報を素直にsubjid単位で
    集約した上で、集約後にrand非欠損の症例だけに絞り込む
    （--include_unrandomized で絞り込みを無効化可能、
    --exclude_events で特定イベントを手動除外することも可能＝汎用オプションとして維持）。

⚠️ 出力は必ずファイル(CSV/Markdown)に保存する設計。
  ターミナルには行数・欠測率などの要約のみを出す
  （「背景因子の検索の出力が長すぎて貼れなかった」問題への対策）。

使い方（例）:
  python diamond_table1_external.py \
      --redcap_csv "/Volumes/Extreme Pro/Red Cap data/ThopazRCT2_raw.csv" \
      --labels_csv ~/Documents/DIAMOND/DIAMOND/split_models_external/true_pal_labels.csv \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--redcap_csv", required=True)
    p.add_argument("--labels_csv", default=None, help="diamond_build_true_pal_labels.py の出力（層別用、任意）")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--subjid_col", default="subjid")
    p.add_argument("--event_col", default="redcap_event_name")
    p.add_argument("--rand_col", default="rand", help="割付群列。この列が非欠損の症例のみを真のランダム化コホートとして対象とする")
    p.add_argument(
        "--include_unrandomized", action="store_true",
        help="randが欠損の症例(スクリーニングのみで終わった患者等)も対象症例数の集計に含める(非推奨、QC目的のみ)",
    )
    p.add_argument(
        "--dsdecod_col", default="dsdecod",
        help="「プロトコル治療完了報告」フォームの完了/中止フィールド(1=完了,2=中止)。"
             "randがちょうど210件と親試験報告値199件より多かったため2026-08-16に追加。"
             "既定でdsdecod=1(完了)のみに絞り込む",
    )
    p.add_argument(
        "--include_discontinued", action="store_true",
        help="プロトコル治療が中止(dsdecod=2)となった症例も対象症例数の集計に含める(非推奨、QC目的のみ)",
    )
    p.add_argument(
        "--exclude_events", default=None,
        help="手動で特定イベントを集約前に除外する(カンマ区切り、汎用オプション。通常は不要)",
    )
    p.add_argument("--encoding", default="utf-8")
    p.add_argument(
        "--dump_patient_csv", default=None,
        help="2026-08-18追加: 患者単位のワイド形式データ(必要列のみ)をこのパスにCSV保存する。"
             "diamond_compare_cohorts_stats.py で単施設コホートとの統計検定に使う。"
             "ローカルファイルのみに保存し、チャット等には出力しない設計。",
    )
    return p.parse_args()


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    if wanted in df.columns:
        return wanted
    norm_wanted = normalize_colname(wanted)
    for c in df.columns:
        if normalize_colname(c) == norm_wanted:
            return c
    return None


def read_redcap_csv(path: Path, encoding: str) -> pd.DataFrame:
    """diamond_table1_dev.py の read_csv_robust と同様、REDCapのCSVエクスポートも
    環境依存でエンコーディングが変わりうるため複数試す（--encodingで指定した値を最優先）。"""
    tried = [encoding, "utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_err = None
    for enc in dict.fromkeys(tried):  # 重複を除きつつ順序維持
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"{path} を読めるエンコーディングが見つかりませんでした: {last_err}")


def aggregate_redcap_long_to_wide(df: pd.DataFrame, subjid_col: str) -> pd.DataFrame:
    def first_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan
    return df.groupby(subjid_col, dropna=False).agg(first_non_null).reset_index()


# thopaz-methods-2.2-2.3-draft-v1.md の変数辞書に準拠。
# (表示名, REDCap列名, 型) の一覧。型: "continuous" | "categorical"
# 2026-08-16追加: 単施設Table1との対比用にコードブック確認済みの追加変数
#   （SI[Brinkman指数,計算]・FEV1.0[mL]・切除範囲[surg checkboxから導出]・癒着[adhesion]）。
#   PS/間質性肺炎/糖尿病/抗血栓薬/FVC/術中Leak test/sealantはRCT-2のREDCapに項目自体が
#   存在しないため取得不可（コードブック全文確認済み、2026-08-16）。
# 2026-08-18追加: 親試験本文Table1(ユーザー提供、Group A n=93/Group B n=106)との対比のため、
# コードブックで存在を再確認した追加変数。fev1p/vc/vcp/pao2/paco2は「術前呼吸機能検査
# (ddcd_5151)」インストゥルメント、tnm/thoracotomyはそれぞれ独立フィールド。いずれも
# 患者ごとに1回のみ記録される非repeatingインストゥルメントのため、他の患者背景変数と
# 同様「最初の非欠損値」で問題ない。
# ⚠️ DLCO(%)は術前セットには存在せず、「術後呼吸機能検査(ddcd_e5aa、術後3ヶ月±14日)」の
#   dlco2列のみコードブックに存在する＝術前ベースライン値としては取得不可（親論文Table1の
#   DLCO値がどのソースかは不明、本REDCapからは再現できない点を明記する）。
CONTINUOUS_VARS = [
    ("年齢", "age2"),
    ("手術時間(分)", "surgtime"),
    ("出血量(mL)", "bloodloss"),
    ("FEV1.0 (mL)", "fev1"),
    ("FEV1.0 (%)", "fev1p"),
    ("VC (mL)", "vc"),
    ("VC (%)", "vcp"),
    ("PaO2 (mmHg)", "pao2"),
    ("PaCO2 (mmHg)", "paco2"),
]
CATEGORICAL_VARS = [
    ("性別", "sex2"),
    ("喫煙(Brinkman指数カテゴリ、≧100=pack-years換算で5超に相当)", "adjust_cig_result"),
    ("肺気腫/COPD", "adjust_3"),
    ("原疾患", "primdise"),
    ("術式", "surg"),
    ("手術アプローチ", "appro"),
    ("割付群", "rand"),
    ("施設", "facilities"),
    ("TNM病期", "tnm"),
    ("同側開胸手術の既往", "thoracotomy"),
]
# 出力は生のREDCapコード値のまま(value_counts()、ラベル変換は行わない)。
# 手動翻訳用の参考(コードブック確認済み、2026-08-18):
#   tnm: 1=潜伏癌/2=0期/3=IA期/4=IA1期/5=IA2期/6=IA3期/7=IB期/8=IIA期/9=IIB期/
#        10=IIIA期/11=IIIB期/12=IIIC期/13=IV期/14=IVA期/15=IVB期
#   thoracotomy(同側開胸手術の既往): 1=あり/2=なし/9=不明

# REDCap checkbox列(surg___1〜___9)から手術術式を単一カテゴリに導出する。
# 1=一葉切除,2=二葉切除 → 「葉切除以上」/ 3=区域切除 / 4=部分(楔状)切除 / 9=その他。
# 優先順位は「より侵襲的な区分を優先」(diamond_table1_dev.pyの術式文字列推定と同じ設計思想)。
SURG_CHECKBOX_TO_EXTENT = [
    (["surg___1", "surg___2"], "葉切除以上"),
    (["surg___3"], "区域切除"),
    (["surg___4"], "部分切除"),
]


def is_checked(val: object) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in {"1", "1.0", "true", "checked", "yes", "y"}


def compute_derived_vars(wide: pd.DataFrame) -> pd.DataFrame:
    """SI(Brinkman指数, 計算値)・切除範囲(surg checkboxから導出)・癒着の有無(2値化)を追加する。"""
    wide = wide.copy()

    # SI = 1日本数 × 喫煙年数（REDCapのadjust_cig_resultは計算済みの2値カテゴリのみのため、
    # 単施設のPRE-S.I.と比較可能な連続値が別途必要）
    num_col = resolve_col(wide, "adjust_cig_num")
    yrs_col = resolve_col(wide, "adjust_cig_his")
    if num_col and yrs_col:
        wide["_si_computed"] = pd.to_numeric(wide[num_col], errors="coerce") * pd.to_numeric(wide[yrs_col], errors="coerce")

    # 切除範囲: surg___1(一葉)/___2(二葉)/___3(区域)/___4(部分)のcheckbox列から導出
    surg_cols = {c: resolve_col(wide, c) for c in ["surg___1", "surg___2", "surg___3", "surg___4", "surg___9"]}
    if any(surg_cols.values()):
        def extent_for_row(row) -> str:
            for cols, label in SURG_CHECKBOX_TO_EXTENT:
                for c in cols:
                    resolved = surg_cols.get(c)
                    if resolved and resolved in row.index and is_checked(row[resolved]):
                        return label
            oth = surg_cols.get("surg___9")
            if oth and oth in row.index and is_checked(row[oth]):
                return "その他"
            return "不明"
        wide["_derived_resection_extent"] = wide.apply(extent_for_row, axis=1)

    # 癒着: adhesion(1=なし,2=高度,3=中等度,4=軽度) を 有無の2値に変換
    adhesion_col = resolve_col(wide, "adhesion")
    if adhesion_col:
        adhesion_num = pd.to_numeric(wide[adhesion_col], errors="coerce")
        wide["_adhesion_present"] = adhesion_num.map(lambda v: "あり" if v in (2, 3, 4) else ("なし" if v == 1 else np.nan))

    return wide


def summarize_continuous(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    n = int(s.notna().sum())
    missing = int(s.isna().sum())
    if n == 0:
        return {
            "n": 0, "missing": missing, "mean": np.nan, "sd": np.nan,
            "median": np.nan, "q1": np.nan, "q3": np.nan, "min": np.nan, "max": np.nan,
        }
    return {
        "n": n, "missing": missing,
        "mean": float(s.mean()), "sd": float(s.std()),
        "median": float(s.median()), "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)),
        "min": float(s.min()), "max": float(s.max()),
    }


def summarize_categorical(series: pd.Series) -> pd.DataFrame:
    s = series.astype("object")
    missing = int(s.isna().sum())
    counts = s.value_counts(dropna=True)
    total = int(counts.sum())
    rows = []
    for cat, n in counts.items():
        rows.append({"category": cat, "n": int(n), "pct": (n / total * 100) if total else np.nan})
    rows.append({"category": "(missing)", "n": missing, "pct": np.nan})
    return pd.DataFrame(rows)


def build_table1(wide: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """group_colがNoneなら全体1列、指定があればgroup別に列を分けて出力する長形式の表を返す。"""
    groups = [None] if group_col is None else sorted(wide[group_col].dropna().unique().tolist())
    records = []

    for label, col in CONTINUOUS_VARS:
        resolved = resolve_col(wide, col)
        if resolved is None:
            records.append({"variable": label, "redcap_col": col, "group": "ALL", "type": "continuous", "note": "列が見つかりません"})
            continue
        if group_col is None:
            stat = summarize_continuous(wide[resolved])
            records.append({"variable": label, "redcap_col": resolved, "group": "ALL", "type": "continuous", **stat})
        else:
            for g in groups:
                sub = wide[wide[group_col] == g]
                stat = summarize_continuous(sub[resolved])
                records.append({"variable": label, "redcap_col": resolved, "group": str(g), "type": "continuous", **stat})

    for label, col in CATEGORICAL_VARS:
        resolved = resolve_col(wide, col)
        if resolved is None:
            records.append({"variable": label, "redcap_col": col, "group": "ALL", "type": "categorical", "note": "列が見つかりません"})
            continue
        if group_col is None:
            cat_df = summarize_categorical(wide[resolved])
            for _, r in cat_df.iterrows():
                records.append({
                    "variable": label, "redcap_col": resolved, "group": "ALL", "type": "categorical",
                    "category": r["category"], "n": r["n"], "pct": r["pct"],
                })
        else:
            for g in groups:
                sub = wide[wide[group_col] == g]
                cat_df = summarize_categorical(sub[resolved])
                for _, r in cat_df.iterrows():
                    records.append({
                        "variable": label, "redcap_col": resolved, "group": str(g), "type": "categorical",
                        "category": r["category"], "n": r["n"], "pct": r["pct"],
                    })

    return pd.DataFrame(records)


def format_median_iqr(stat: dict) -> str:
    if stat["n"] == 0 or pd.isna(stat["median"]):
        return "—"
    return f"{stat['median']:.1f} ({stat['q1']:.1f}–{stat['q3']:.1f})"


def mannwhitney_p(a: pd.Series, b: pd.Series) -> float:
    """Mann-Whitney U (Wilcoxon rank-sum)。親論文の連続変数比較(Supplemental Table 3等)が
    'Wilcoxon test'表記のため、median/IQR表示と対応する検定として採用。
    片群のn<1、または両群とも値が完全に同一(定数)でエラーになる場合はNaNを返す。"""
    a2 = pd.to_numeric(a, errors="coerce").dropna()
    b2 = pd.to_numeric(b, errors="coerce").dropna()
    if len(a2) < 1 or len(b2) < 1:
        return np.nan
    try:
        _, p = scipy_stats.mannwhitneyu(a2, b2, alternative="two-sided")
        return float(p)
    except ValueError:
        return np.nan


def build_continuous_group_pvalue_table(wide: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """CONTINUOUS_VARS(年齢・喫煙指数・呼吸機能等)について、group_col(通常はrand=割付群)の
    2群間でmedian(IQR)とMann-Whitney U検定のp値を1行1変数の横持ちで出す。
    親論文Supplemental Tableと同じ『中央値(IQR)・P value』形式に揃えている。
    3群以上の場合はp値計算をスキップし、group_labelsに含まれる全群のmedian(IQR)のみ出す。"""
    groups = sorted(wide[group_col].dropna().unique().tolist())
    records = []
    for label, col in CONTINUOUS_VARS:
        resolved = resolve_col(wide, col)
        if resolved is None:
            records.append({"variable": label, "redcap_col": col, "note": "列が見つかりません"})
            continue
        row = {"variable": label, "redcap_col": resolved}
        group_series = {}
        for g in groups:
            sub = wide[wide[group_col] == g][resolved]
            stat = summarize_continuous(sub)
            gname = str(g)
            row[f"n_group{gname}"] = stat["n"]
            row[f"median_iqr_group{gname}"] = format_median_iqr(stat)
            group_series[g] = sub
        if len(groups) == 2:
            p = mannwhitney_p(group_series[groups[0]], group_series[groups[1]])
            row["p_value"] = p
            row["test"] = "Mann-Whitney U"
        else:
            row["p_value"] = np.nan
            row["test"] = f"n_groups={len(groups)}(2群でないためp値スキップ)"
        records.append(row)
    return pd.DataFrame(records)


def compute_bmi(wide: pd.DataFrame) -> pd.DataFrame:
    height_col = resolve_col(wide, "hight") or resolve_col(wide, "height")
    weight_col = resolve_col(wide, "weight")
    if height_col is None or weight_col is None:
        print(f"[WARN] BMI算出用の身長/体重列が見つかりません（hight/height={height_col}, weight={weight_col}）")
        return wide
    h_m = pd.to_numeric(wide[height_col], errors="coerce") / 100.0
    w_kg = pd.to_numeric(wide[weight_col], errors="coerce")
    wide = wide.copy()
    wide["_bmi_computed"] = w_kg / (h_m ** 2)
    return wide


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # tabulateが無い環境向けの簡易フォールバック
        cols = df.columns.tolist()
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)


def main() -> int:
    args = parse_args()
    redcap_path = Path(args.redcap_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = read_redcap_csv(redcap_path, args.encoding)
    subjid_col = resolve_col(df_raw, args.subjid_col)
    if subjid_col is None:
        raise RuntimeError(f"subjid列が見つかりません。列名候補: {[c for c in df_raw.columns if 'subj' in normalize_colname(c)]}")

    n_unique_all = df_raw[subjid_col].dropna().astype(str).str.strip().nunique()
    print(f"[INFO] 全イベント合算のsubjidユニーク数(スクリーニングのみで終わった患者を含む): {n_unique_all}")

    df_filtered = df_raw
    if args.exclude_events:
        event_col = resolve_col(df_raw, args.event_col)
        excluded = [e.strip() for e in args.exclude_events.split(",") if e.strip()]
        if event_col:
            df_filtered = df_raw[~df_raw[event_col].isin(excluded)]
            print(f"[INFO] 手動指定によりイベントを除外: {len(excluded)}件 -> 除外後行数={len(df_filtered)}")
        else:
            print("[WARN] --exclude_eventsが指定されましたがevent列が見つからず適用できません")

    # 全イベントの情報をsubjid単位に集約する（「最初の非欠損値」採用のため、
    # ある列がどのイベントにあっても正しく1患者1行に統合される）。
    wide = aggregate_redcap_long_to_wide(df_filtered, subjid_col)
    wide = compute_bmi(wide)
    if "_bmi_computed" in wide.columns:
        CONTINUOUS_VARS.append(("BMI(算出値)", "_bmi_computed"))

    wide = compute_derived_vars(wide)
    if "_si_computed" in wide.columns:
        CONTINUOUS_VARS.append(("喫煙指数(Brinkman Index, 算出値)", "_si_computed"))
    if "_derived_resection_extent" in wide.columns:
        CATEGORICAL_VARS.append(("切除範囲(surg checkboxから導出)", "_derived_resection_extent"))
    if "_adhesion_present" in wide.columns:
        CATEGORICAL_VARS.append(("胸腔内癒着", "_adhesion_present"))

    # 真のランダム化コホートに限定: rand(割付群)が記録されている症例のみを対象とする。
    # スクリーニングのみで終わった患者（ランダム化前に除外・辞退等）はrandが欠損のため
    # 自然に除外される。
    rand_col = resolve_col(wide, args.rand_col)
    if rand_col is not None and not args.include_unrandomized:
        n_before = len(wide)
        wide = wide[wide[rand_col].notna()].reset_index(drop=True)
        print(
            f"[INFO] 割付群(rand)が記録されている症例のみに限定(=真のランダム化コホート): "
            f"{n_before} -> {len(wide)}"
        )
    elif rand_col is None:
        print(
            f"[WARN] 割付群列（--rand_col='{args.rand_col}'）が見つからず、"
            "ランダム化コホートへの絞り込みができません。全症例を対象とします。"
        )
    else:
        print("[INFO] --include_unrandomizedが指定されたため、rand非欠損によるコホート限定はスキップします")

    # 2026-08-16追加: rand非欠損=210件は親試験の報告値(199例)より多い。
    # コードブック確認の結果、「プロトコル治療完了報告」フォーム(dsdecod: 1=完了/2=中止)が
    # ランダム化された210例全員に存在することが判明。無作為化後の中止例(有害事象・同意撤回等)を
    # 除いた「完了」例のみが最終的な解析対象コホートに対応すると考えられるため、既定でdsdecod=1に絞る。
    dsdecod_col = resolve_col(wide, args.dsdecod_col)
    if dsdecod_col is not None and not args.include_discontinued:
        n_before = len(wide)
        dsdecod_num = pd.to_numeric(wide[dsdecod_col], errors="coerce")
        n_discontinued = int((dsdecod_num == 2).sum())
        n_missing_dsdecod = int(dsdecod_num.isna().sum())
        wide = wide[dsdecod_num == 1].reset_index(drop=True)
        print(
            f"[INFO] プロトコル治療完了(dsdecod=1)の症例のみに限定: {n_before} -> {len(wide)} "
            f"(中止={n_discontinued}件, dsdecod欠損={n_missing_dsdecod}件)"
        )
    elif dsdecod_col is None:
        print(
            f"[WARN] プロトコル治療完了報告列（--dsdecod_col='{args.dsdecod_col}'）が見つからず、"
            "完了/中止による絞り込みができません。ランダム化コホート全体を対象とします。"
        )
    else:
        print("[INFO] --include_discontinuedが指定されたため、dsdecodによるコホート限定はスキップします")

    print(f"[INFO] 対象症例数(Table1母数): {len(wide)}")

    table1_all = build_table1(wide, group_col=None)
    table1_all_path_csv = output_dir / "table1_external_all.csv"
    table1_all_path_md = output_dir / "table1_external_all.md"
    table1_all.to_csv(table1_all_path_csv, index=False)
    table1_all_path_md.write_text(markdown_table(table1_all), encoding="utf-8")
    print(f"[OK] Table1(全体)を保存: {table1_all_path_csv} / {table1_all_path_md}")

    wide_with_pal = wide
    if args.labels_csv:
        labels = pd.read_csv(Path(args.labels_csv).expanduser())
        if "subjid" not in labels.columns or "true_pal" not in labels.columns:
            print("[WARN] labels_csvにsubjid/true_pal列がないため層別表はスキップします")
        else:
            merged = wide.merge(labels[["subjid", "true_pal"]], left_on=subjid_col, right_on="subjid", how="left")
            wide_with_pal = merged
            n_pal_pos = int((merged["true_pal"] == 1).sum())
            n_pal_neg = int((merged["true_pal"] == 0).sum())
            n_pal_na = int(merged["true_pal"].isna().sum())
            print(f"[INFO] true_pal突合結果: PAL+={n_pal_pos}, PAL-={n_pal_neg}, 判定不能/未突合={n_pal_na}")

            table1_strat = build_table1(merged, group_col="true_pal")
            strat_path_csv = output_dir / "table1_external_by_true_pal.csv"
            strat_path_md = output_dir / "table1_external_by_true_pal.md"
            table1_strat.to_csv(strat_path_csv, index=False)
            strat_path_md.write_text(markdown_table(table1_strat), encoding="utf-8")
            print(f"[OK] Table1(PAL+/-層別)を保存: {strat_path_csv} / {strat_path_md}")
    else:
        print("[INFO] --labels_csv未指定のため層別表はスキップ（全体表のみ出力）")

    # 2026-08-18追加: diamond_compare_cohorts_stats.py（両コホート統計検定）用に、
    # 患者単位のワイド形式データを必要列だけ選んでCSV保存する（ローカルのみ、チャット非出力）。
    if args.dump_patient_csv:
        dump_cols_wanted = [
            ("age2", "age"),
            ("sex2", "sex"),
            ("adjust_cig_result", "smoking_brinkman100"),
            ("_si_computed", "si"),
            ("adjust_3", "copd"),
            ("fev1", "fev1_ml"),
            ("primdise", "primary_disease"),
            ("appro", "approach_code"),
            ("_derived_resection_extent", "resection_extent"),
            ("_adhesion_present", "adhesion"),
        ]
        dump_df = pd.DataFrame()
        for src_wanted, out_name in dump_cols_wanted:
            resolved = resolve_col(wide_with_pal, src_wanted)
            if resolved is not None:
                dump_df[out_name] = wide_with_pal[resolved]
            else:
                dump_df[out_name] = np.nan
                print(f"[WARN] --dump_patient_csv: 列'{src_wanted}'が見つからず{out_name}は全欠損になります")
        if "true_pal" in wide_with_pal.columns:
            dump_df["pal_true"] = wide_with_pal["true_pal"]
        else:
            dump_df["pal_true"] = np.nan
            print("[WARN] --dump_patient_csv: true_pal列が無いため(labels_csv未指定?) pal_trueは全欠損になります")
        dump_path = Path(args.dump_patient_csv).expanduser()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_df.to_csv(dump_path, index=False)
        print(f"[OK] 患者単位ワイドデータ(n={len(dump_df)})を保存: {dump_path}")

    if not args.labels_csv:
        print(
            "[NOTE] PAL+/-層別表が必要な場合は、先に diamond_build_true_pal_labels.py で "
            "true_pal_labels.csv を作成し、--labels_csv で渡してください。"
        )

    # 2026-08-18追加: 親論文本文Table1(Group A: -8cmH2O n=93 / Group B: -15cmH2O n=106)との
    # 直接比較用に、rand(割付群)で層別した表も出力する。
    if rand_col is not None and rand_col in wide.columns:
        rand_group_counts = wide[rand_col].value_counts(dropna=True).to_dict()
        print(f"[INFO] 割付群(rand)内訳(1=A群-8cmH2O/2=B群-15cmH2O): {rand_group_counts} "
              "(親論文Table1: Group A n=93 / Group B n=106)")
        table1_by_rand = build_table1(wide, group_col=rand_col)
        rand_path_csv = output_dir / "table1_external_by_rand_group.csv"
        rand_path_md = output_dir / "table1_external_by_rand_group.md"
        table1_by_rand.to_csv(rand_path_csv, index=False)
        rand_path_md.write_text(markdown_table(table1_by_rand), encoding="utf-8")
        print(f"[OK] Table1(割付群A/B層別)を保存: {rand_path_csv} / {rand_path_md}")

        # 2026-08-19追加: 連続変数(年齢・喫煙指数・呼吸機能等)のGroup A/B比較を
        # 『中央値(IQR)・Mann-Whitney U p値』の1行1変数コンパクト表として別出力。
        # 親論文Supplemental Table 1/3と同じ形式で突合しやすくするため。
        n_rand_groups = wide[rand_col].dropna().nunique()
        if n_rand_groups == 2:
            cont_pval = build_continuous_group_pvalue_table(wide, group_col=rand_col)
            cont_pval_path_csv = output_dir / "table1_external_continuous_baseline_pvalue.csv"
            cont_pval_path_md = output_dir / "table1_external_continuous_baseline_pvalue.md"
            cont_pval.to_csv(cont_pval_path_csv, index=False)
            header_note = (
                "group1 = Group A (-8 cmH2O), group2 = Group B (-15 cmH2O) を想定"
                "（rand列の実コード値は上のrand_group_countsログで要確認）。\n\n"
            )
            cont_pval_path_md.write_text(header_note + markdown_table(cont_pval), encoding="utf-8")
            print(f"[OK] 連続変数Group A/B比較(median/IQR/p値)を保存: {cont_pval_path_csv} / {cont_pval_path_md}")
        else:
            print(f"[WARN] rand列の非欠損カテゴリ数が{n_rand_groups}のため、連続変数p値表はスキップ(2群のみ対応)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
