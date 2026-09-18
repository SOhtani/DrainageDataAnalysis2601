#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_inspect_modifi_criterion2.py (v2, 2026-08-17)

目的:
  diamond_build_true_pal_labels.py のCriterion2(早期侵襲的処置)判定で、
  胸膜癒着術(modifi___4)の件数が親試験報告値(5例)に対し実データで80例と
  大きく乖離している(16倍)問題を、コードブック再照合で判明した手がかりを
  使って診断する。

  ⚠️ 2026-08-17追記: 親論文本文(Journal Pre-proof、ユーザー提供)のENDPOINTS節で、
  一次エンドポイントPALの定義が確定した:
    "PAL was defined as either a persistent air leak after POD 5 or the need
     for invasive procedure, including additional chest tube insertion,
     pleurodesis, or reoperation, **due to an air leak** until POD 4."
  「due to an air leak」＝エアーリークが理由の処置に限定、という条件がある。
  コードブック234番field [chanres](変更理由: 1=エアーリーク/2=エアーリーク以外)
  はこの条件に直接対応するが、現在のCriterion2判定コードはchanresを一切見て
  いない。chantim(変更時期: 1=二次登録前/2=二次登録後)も同様に未フィルタ。
  この2つが、pleurodesis過大カウント(80 vs 期待5)の主因である可能性が高い。

  出力は全て「件数・割合」の集計値のみ。患者ID・日付の生値は一切出力しない
  （プライバシー制約）。

  診断1: modifi___1〜9の生値分布(チェック済み件数、行レベル)。
  診断2: modifi___3/4/5(drain/pleurodesis/reop)各行の chantim(変更時期)内訳。
         現在未フィルタなので、二次登録前(1)の混入があれば過大カウントの一因。
  診断3: modifi___3/4/5各行の chanres(変更理由: エアーリーク/それ以外)内訳。
         【新規】親論文の定義「due to an air leak」に直接対応する最重要フィルタ。
         現在未フィルタなので、エアーリーク以外(2)の混入があれば過大カウントの一因。
  診断4: pleumeth(胸膜癒着術使用薬剤___1/2/3/9)は、コードブック上modifi___4=1の
         記録に付随して入力される設計(分岐ロジック確認済み)。modifi___4=1行の
         うちpleumethが実際に入力されている割合を見る、独立の妥当性チェック。
  診断5: modifi___3/4/5各行のchandat-surgedat日数分布(バケット化、生日付非出力)。
  診断6: topazchanyn(患者単位「一次登録後の変更有無」)とmodifi___3/4/5存在の整合性。

使い方（例）:
  python diamond_inspect_modifi_criterion2.py \
      --redcap_csv "/Volumes/Extreme Pro/Red Cap data/ThopazRCT2_raw.csv" \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CRITERION2_KEYS = {"3": "追加ドレーン挿入", "4": "胸膜癒着術", "5": "再手術"}
EXPECTED_COUNTS = {"3": 2, "4": 5, "5": 1}  # 親試験報告値（参考表示のみ）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--redcap_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--subjid_col", default="subjid")
    p.add_argument("--surgedat_col", default="surgedat")
    p.add_argument("--chandat_col", default="chandat")
    p.add_argument("--chantim_col", default="chantim", help="変更時期(1=二次登録前/2=二次登録後)")
    p.add_argument("--chanres_col", default="chanres", help="変更理由(1=エアーリーク/2=エアーリーク以外)")
    p.add_argument("--topazchanyn_col", default="topazchanyn", help="一次登録後の変更有無(1=あり/2=なし)")
    p.add_argument("--modifi_prefix", default="modifi___", help="modifi___1〜9 の共通プレフィックス")
    p.add_argument("--pleumeth_prefix", default="pleumeth___", help="pleumeth___1/2/3/9 の共通プレフィックス")
    p.add_argument("--rand_col", default="rand")
    p.add_argument("--dsdecod_col", default="dsdecod")
    p.add_argument("--include_unrandomized", action="store_true")
    p.add_argument("--include_discontinued", action="store_true")
    p.add_argument("--encoding", default="utf-8")
    return p.parse_args()


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(name).strip().lower())


def resolve_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    if wanted in df.columns:
        return wanted
    norm_wanted = normalize_colname(wanted)
    for c in df.columns:
        if normalize_colname(c) == norm_wanted:
            return c
    return None


def resolve_checkbox_cols(df: pd.DataFrame, prefix: str) -> dict:
    """prefix + 数字 の形の列を全て見つける(例: modifi___1, modifi___2, ...)。"""
    out = {}
    for c in df.columns:
        nc = normalize_colname(c)
        np_ = normalize_colname(prefix)
        if nc.startswith(np_):
            suffix = nc[len(np_):]
            if suffix.isdigit():
                out[suffix] = c
    return out


def is_checked(val: object) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in {"1", "1.0", "true", "checked", "yes", "y"}


def read_redcap_csv(path: Path, encoding: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        print(f"[WARN] encoding={encoding}で読めなかったためcp932を試します")
        return pd.read_csv(path, encoding="cp932", low_memory=False)


def first_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan


def radio_label(v: object, code_map: dict) -> str:
    """radio型フィールド(1/2等)の値を安全にラベル化する(コードは選択肢番号であり患者情報ではない)。"""
    if pd.isna(v):
        return "欠損/不明"
    s = str(v).strip()
    for code, label in code_map.items():
        if s in {code, f"{code}.0"}:
            return label
    return f"欠損/不明(生値={s!r})"


def main() -> int:
    args = parse_args()
    redcap_path = Path(args.redcap_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = read_redcap_csv(redcap_path, args.encoding)
    print(f"[INFO] REDCap生データ読込: {len(df_raw)}行, {len(df_raw.columns)}列")

    subjid_col = resolve_col(df_raw, args.subjid_col)
    if subjid_col is None:
        raise RuntimeError("subjid列が見つかりません")

    # --- コホート限定: rand非欠損 → dsdecod=1（diamond_build_true_pal_labels.pyと同一ロジック） ---
    wide = df_raw.groupby(subjid_col, dropna=False).agg(first_non_null).reset_index()
    print(f"[INFO] subjid単位集約(全イベント): {len(wide)}症例")

    rand_col = resolve_col(wide, args.rand_col)
    if rand_col and not args.include_unrandomized:
        n_before = len(wide)
        wide = wide[wide[rand_col].notna()].reset_index(drop=True)
        print(f"[INFO] rand非欠損に限定: {n_before} -> {len(wide)}")

    dsdecod_col = resolve_col(wide, args.dsdecod_col)
    if dsdecod_col and not args.include_discontinued:
        n_before = len(wide)
        dsdecod_num = pd.to_numeric(wide[dsdecod_col], errors="coerce")
        wide = wide[dsdecod_num == 1].reset_index(drop=True)
        print(f"[INFO] dsdecod=1(完了)に限定: {n_before} -> {len(wide)}")

    cohort_ids = set(wide[subjid_col])
    print(f"[INFO] 最終コホート症例数: {len(cohort_ids)}")

    df_long = df_raw[df_raw[subjid_col].isin(cohort_ids)].copy()
    print(f"[INFO] コホートに属する全イベント行数: {len(df_long)}")

    # --- 列解決 ---
    surgedat_col = resolve_col(wide, args.surgedat_col)
    chandat_col = resolve_col(df_long, args.chandat_col)
    chantim_col = resolve_col(df_long, args.chantim_col)
    chanres_col = resolve_col(df_long, args.chanres_col)
    topazchanyn_col = resolve_col(wide, args.topazchanyn_col)
    modifi_cols = resolve_checkbox_cols(df_long, args.modifi_prefix)
    pleumeth_cols = resolve_checkbox_cols(df_long, args.pleumeth_prefix)

    print(
        f"[INFO] 列解決: chandat={chandat_col}, chantim={chantim_col}, "
        f"chanres={chanres_col}, topazchanyn={topazchanyn_col}"
    )
    print(f"[INFO] modifi___番号一覧: {sorted(modifi_cols.keys())}")
    print(f"[INFO] pleumeth___番号一覧: {sorted(pleumeth_cols.keys())}")

    if not modifi_cols:
        raise RuntimeError("modifi___N列が1つも見つかりませんでした。--modifi_prefixを確認してください。")

    surgedat_by_patient = dict(zip(wide[subjid_col], pd.to_datetime(wide[surgedat_col], errors="coerce"))) if surgedat_col else {}

    # ============================================================
    # 診断1: modifi___1〜9 の生値分布（行レベル、チェック済み件数のみ）
    # ============================================================
    print("\n===== 診断1: modifi___1〜9 チェック済み行数（コホート全イベント行ベース） =====")
    for num in sorted(modifi_cols.keys(), key=int):
        col = modifi_cols[num]
        n_checked = int(df_long[col].apply(is_checked).sum())
        n_nonnull = int(df_long[col].notna().sum())
        uniques = df_long[col].dropna().astype(str).str.strip().str.lower().value_counts().to_dict()
        label = CRITERION2_KEYS.get(num, "")
        print(f"  modifi___{num} {label}: チェック済み行数={n_checked} / 非欠損行数={n_nonnull} / 値表記内訳={uniques}")

    # 以降の診断はCriterion2の3成分(drain=3/pleurodesis=4/reop=5)についてループする
    rows_by_key = {}
    for num, label in CRITERION2_KEYS.items():
        col = modifi_cols.get(num)
        if col is None:
            print(f"\n[WARN] modifi___{num}({label})列が見つからないためスキップ")
            continue
        rows_by_key[num] = df_long[df_long[col].apply(is_checked)].copy()

    # ============================================================
    # 診断2: chantim(変更時期: 1=二次登録前/2=二次登録後) 内訳
    # ============================================================
    if chantim_col:
        print("\n===== 診断2: modifi___3/4/5=1 各行の chantim(変更時期)内訳 =====")
        for num, label in CRITERION2_KEYS.items():
            rows = rows_by_key.get(num)
            if rows is None:
                continue
            counts = rows[chantim_col].apply(
                lambda v: radio_label(v, {"1": "1_二次登録前", "2": "2_二次登録後"})
            ).value_counts().to_dict()
            print(f"  modifi___{num}({label}, 期待値{EXPECTED_COUNTS[num]}例, 実データ行数{len(rows)}): {counts}")
        print("  ※現在のCriterion2判定コードはchantimを見ていない。「1_二次登録前」の混入が"
              "過大カウントの一因である可能性が高い。")
    else:
        print("\n[WARN] 診断2スキップ: chantim列が見つかりません")

    # ============================================================
    # 診断3【新規・最重要】: chanres(変更理由: 1=エアーリーク/2=エアーリーク以外) 内訳
    # 親論文の一次エンドポイント定義「due to an air leak until POD 4」に直接対応
    # ============================================================
    if chanres_col:
        print("\n===== 診断3【最重要】: modifi___3/4/5=1 各行の chanres(変更理由)内訳 =====")
        print("  （親論文定義: 侵襲的処置は'due to an air leak'である必要がある。エアーリーク以外の")
        print("   理由による処置は本来PAL判定に含めるべきではない）")
        for num, label in CRITERION2_KEYS.items():
            rows = rows_by_key.get(num)
            if rows is None:
                continue
            counts = rows[chanres_col].apply(
                lambda v: radio_label(v, {"1": "1_エアーリークが理由", "2": "2_エアーリーク以外が理由"})
            ).value_counts().to_dict()
            print(f"  modifi___{num}({label}, 期待値{EXPECTED_COUNTS[num]}例, 実データ行数{len(rows)}): {counts}")
        print("  ※現在のCriterion2判定コードはchanresを一切見ていない。"
              "「2_エアーリーク以外が理由」の混入が過大カウントの主因である可能性が最も高い。")
    else:
        print("\n[WARN] 診断3スキップ: chanres列が見つかりません")

    # ============================================================
    # 診断4: modifi___4=1 の行における pleumeth(使用薬剤)整合性
    # ============================================================
    m4_rows = rows_by_key.get("4")
    if m4_rows is not None and pleumeth_cols:
        print("\n===== 診断4: modifi___4=1 の行における pleumeth(胸膜癒着術使用薬剤)整合性 =====")
        any_pleumeth = pd.Series(False, index=m4_rows.index)
        for num, col in pleumeth_cols.items():
            any_pleumeth = any_pleumeth | m4_rows[col].apply(is_checked)
        n_with_drug = int(any_pleumeth.sum())
        n_total_m4 = len(m4_rows)
        pct = (n_with_drug / n_total_m4 * 100) if n_total_m4 else float("nan")
        print(f"  modifi___4=1の行数: {n_total_m4}")
        print(f"  うちpleumeth(使用薬剤)が1つ以上チェックされている行数: {n_with_drug} ({pct:.1f}%)")
        print(f"  うちpleumethが全て空欄/未チェックの行数: {n_total_m4 - n_with_drug} ({100 - pct:.1f}%)")
        print("  ※コードブックの分岐ロジック上、真の胸膜癒着術ならpleumethが入力される設計。"
              "この割合が低い場合、modifi___4=1の多くが実際には癒着術ではない可能性が高い。")
    else:
        print("\n[WARN] 診断4スキップ: modifi___4またはpleumeth列が見つかりません")

    # ============================================================
    # 診断5: chandat-surgedat 日数分布（バケット化、生日付非出力）
    # ============================================================
    if chandat_col and surgedat_by_patient:
        print("\n===== 診断5: modifi___3/4/5=1 各行の chandat-surgedat 日数分布（バケット） =====")

        def bucket(d):
            if pd.isna(d):
                return "chandat欠損"
            if d < 0:
                return "負の値(手術前?)"
            if d <= 4:
                return "0-4日(Criterion2該当範囲)"
            if d <= 10:
                return "5-10日"
            return "11日以上"

        for num, label in CRITERION2_KEYS.items():
            rows = rows_by_key.get(num)
            if rows is None:
                continue
            rows = rows.copy()
            rows["_surgedat"] = rows[subjid_col].map(surgedat_by_patient)
            rows["_chandat"] = pd.to_datetime(rows[chandat_col], errors="coerce")
            rows["_days"] = (rows["_chandat"] - rows["_surgedat"]).dt.days
            counts = rows["_days"].apply(bucket).value_counts().to_dict()
            print(f"  modifi___{num}({label}): {counts}")
    else:
        print("\n[WARN] 診断5スキップ: 必要な列が見つかりません")

    # ============================================================
    # 診断6: topazchanyn(患者単位「変更有無」)とmodifi___3/4/5存在の整合性
    # ============================================================
    if topazchanyn_col:
        print("\n===== 診断6: topazchanyn(患者単位、一次登録後の変更有無)との整合性 =====")
        topazchanyn_num = pd.to_numeric(wide[topazchanyn_col], errors="coerce")
        for num, label in CRITERION2_KEYS.items():
            rows = rows_by_key.get(num)
            if rows is None:
                continue
            patients_with_flag = set(rows[subjid_col])
            has_flag = wide[subjid_col].isin(patients_with_flag)
            n_total = int(has_flag.sum())
            n_yes = int((has_flag & (topazchanyn_num == 1)).sum())
            n_no = int((has_flag & (topazchanyn_num == 2)).sum())
            n_missing = int((has_flag & topazchanyn_num.isna()).sum())
            print(
                f"  modifi___{num}({label})を持つ患者数={n_total} / "
                f"topazchanyn=1(あり)={n_yes} / topazchanyn=2(なし,矛盾)={n_no} / 欠損={n_missing}"
            )
        print("  ※topazchanyn=2(変更なし)なのにmodifi行が存在する場合は矛盾＝データ不整合の証拠。")
    else:
        print("\n[WARN] 診断6スキップ: topazchanyn列が見つかりません")

    print("\n[OK] 診断完了。上記の集計結果を確認してください（患者ID・日付の生値は一切出力していません）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
